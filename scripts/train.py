import copy
import os
import resource
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from minisora.data import DMLabTrajectoryDataset, MinecraftTrajectoryDataset, ResumableDistributedSampler
from minisora.models import DiTPipeline, build_condition_mask, build_dit
from minisora.train import LinearWarmupCosineLR, LinearWarmupLR, Timer, TrainIO
from minisora.utils import all_reduce_mean, ensure_soft_limit, format_numel, get_model_numel


@dataclass
class FullTrainingArguments:
    # dataset
    data_root: str = "train_data"
    dataset_type: str = "minecraft"
    dataset_limit: Optional[int] = None

    # model
    model_type: str = "small"
    grad_checkpoint: bool = False

    # training
    plugin: str = "zero"
    zero: int = 1
    precision: str = "bf16"
    grad_clip: float = 1.0
    batch_size: int = 32
    gradient_accumulation: int = 1
    num_workers: int = 16
    pin_memory: bool = True
    lr: float = 1e-4
    weight_decay: float = 0.0
    eps: float = 1e-8
    warmup_steps: int = 1000
    lr_total_steps: int = 200_000
    epochs: int = 2000
    seed: int = 42
    outputs: str = "outputs"
    experiment_name: str = "minisora"
    wandb: bool = True
    wandb_project: str = "minisora"
    record_time: bool = False
    load: Optional[str] = None
    load_optimizer: bool = True
    ckpt_every: int = 10000
    log_video_every: int = 1000

    # flow matching
    num_train_timesteps: int = 1000
    weighting_scheme: str = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29
    precondition_outputs: bool = True
    condition_continuation_prob: float = 0.03
    condition_random_prob: float = 0.02

    # validation
    validation_seed: int = 1234
    validation_batches: int = 1
    validation_num_inference_steps: int = 28
    validation_fps: int = 6


def log_validation(transformer, scheduler, cfg, train_io, step, height, width, num_frames):
    device = get_current_device()
    eval_model = transformer
    was_training = eval_model.training
    eval_model.eval()
    pipeline = DiTPipeline(
        transformer=eval_model,
        scheduler=scheduler,
    ).to(device=device)
    pipeline.set_progress_bar_config(disable=False)

    generator = torch.Generator(device=device).manual_seed(cfg.validation_seed + step)

    with torch.inference_mode():
        for sample_idx in range(cfg.validation_batches):
            output = pipeline(
                batch_size=1,
                num_inference_steps=cfg.validation_num_inference_steps,
                generator=generator,
                height=height,
                width=width,
                num_frames=num_frames,
            )
            latents = output.latents[0].detach().cpu()
            train_io.log_video(
                f"validation/sample_{sample_idx}",
                latents,
                -1.0,
                1.0,
                cfg.validation_fps,
                step,
            )

    eval_model.train(was_training)
    del pipeline
    free_memory()


def main(cfg):
    ensure_soft_limit()
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(cfg.seed)
    np.random.seed(cfg.seed + dist.get_rank())
    coordinator = DistCoordinator()
    device = get_current_device()

    precision = cfg.precision.lower()
    if precision == "bf16":
        weight_dtype = torch.bfloat16
    elif precision == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(f"Unknown precision {cfg.precision}.")

    if cfg.batch_size % cfg.gradient_accumulation != 0:
        raise ValueError("`batch_size` must be divisible by `gradient_accumulation`.")

    # ==============================
    # Initialize Booster
    # ==============================
    booster_kwargs = {}
    if cfg.plugin == "zero":
        plugin = LowLevelZeroPlugin(
            stage=cfg.zero,
            precision=cfg.precision,
            max_norm=cfg.grad_clip,
        )
    elif cfg.plugin == "ddp":
        plugin = TorchDDPPlugin()
    else:
        raise ValueError(f"Unknown plugin {cfg.plugin}.")
    booster = Booster(plugin=plugin, **booster_kwargs)

    # ==============================
    # Initialize Model and Optimizer
    # ==============================
    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=cfg.num_train_timesteps)
    noise_scheduler.set_timesteps(cfg.num_train_timesteps)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    transformer = build_dit(
        cfg.model_type,
        in_channels=3,
        out_channels=3,
        attn_implementation="flash_attention_2",
    )

    train_io = TrainIO(cfg, coordinator, project="minisora")

    if cfg.grad_checkpoint:
        transformer.enable_gradient_checkpointing()

    total_params, trainable_params = get_model_numel(transformer)
    train_io.logger.info(
        "Model params: total=%s, trainable=%s",
        format_numel(total_params),
        format_numel(trainable_params),
    )

    # ==============================
    # Initialize Dataset and Dataloader
    # ==============================
    if cfg.dataset_type == "dmlab":
        num_frames = 20
        height = 64
        width = 64
        dataset_class = DMLabTrajectoryDataset
    elif cfg.dataset_type == "minecraft":
        num_frames = 20
        height = 128
        width = 128
        dataset_class = MinecraftTrajectoryDataset
    else:
        raise ValueError(f"Unknown dataset_type {cfg.dataset_type}.")
    dataset_kwargs = dict(
        roots=[f"{cfg.data_root}/{cfg.dataset_type}/train"],
        num_frames=num_frames,
        frame_stride=2,
        resize_hw=(height, width),
        include_actions=False,
        limit=cfg.dataset_limit,
    )
    dataset = dataset_class(**dataset_kwargs)
    local_batch = cfg.batch_size // cfg.gradient_accumulation
    dataloader = plugin.prepare_dataloader(
        dataset,
        batch_size=local_batch,
        shuffle=True,
        drop_last=True,
        seed=cfg.seed,
        pin_memory=cfg.pin_memory,
        num_workers=cfg.num_workers,
        distributed_sampler_cls=ResumableDistributedSampler,
    )
    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_batch_size"):
        dataloader.sampler.set_batch_size(local_batch)

    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, transformer.parameters()),
        adamw_mode=True,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        eps=cfg.eps,
    )
    if cfg.warmup_steps <= 0:
        lr_scheduler = None
    elif cfg.lr_total_steps is not None:
        lr_scheduler = LinearWarmupCosineLR(optimizer, warmup_steps=cfg.warmup_steps, total_steps=cfg.lr_total_steps)
    else:
        lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=cfg.warmup_steps)

    original_dtype = torch.get_default_dtype()
    if weight_dtype == torch.bfloat16:
        torch.set_default_dtype(torch.bfloat16)
    transformer, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=transformer,
        optimizer=optimizer,
        dataloader=dataloader,
        lr_scheduler=lr_scheduler,
    )
    torch.set_default_dtype(original_dtype)

    base_transformer = transformer.unwrap()
    patch_size = base_transformer.config.patch_size
    base_transformer.build_rope_cache(
        num_frames=num_frames,
        height=height,
        width=width,
        device=device,
        dtype=weight_dtype,
    )

    optimizer.zero_grad(set_to_none=True)

    train_io.logger.info(
        "Booster init max CUDA memory: %.2f MB",
        get_accelerator().max_memory_allocated() / 1024**2,
    )
    train_io.logger.info(
        "Booster init max CPU memory: %.2f MB",
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    )

    cfg_epochs = cfg.epochs
    start_epoch = start_step = start_global_step = 0
    if cfg.load:
        train_io.logger.info("Loading checkpoint from %s", cfg.load)
        start_epoch, start_step, start_global_step = train_io.load(
            booster,
            cfg.load,
            model=transformer,
            optimizer=optimizer if cfg.load_optimizer else None,
            lr_scheduler=lr_scheduler if cfg.load_optimizer else None,
        )
    train_io.logger.info("Training for %s epochs", cfg_epochs)

    transformer.train()
    global_step = start_global_step
    conditioning_task_probs = {
        "continuation": cfg.condition_continuation_prob,
        "random": cfg.condition_random_prob,
    }

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero(as_tuple=False)[0].item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    record_time = cfg.record_time
    timers = {}
    for key in ["forward", "backward"]:
        if record_time:
            timers[key] = Timer(key, coordinator=coordinator)
        else:
            timers[key] = nullcontext()

    coordinator.block_all()
    if coordinator.is_master():
        log_validation(transformer.unwrap(), noise_scheduler, cfg, train_io, global_step, height, width, num_frames)
    coordinator.block_all()

    for epoch in range(start_epoch, cfg_epochs):
        sampler_step = start_step if epoch == start_epoch else 0
        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch, step=sampler_step)

        epoch_len = len(dataloader)
        real_epoch_len = epoch_len + sampler_step
        dataloader_iter = iter(dataloader)

        train_io.logger.info(f"Starting epoch {epoch} from step {sampler_step} with {epoch_len} steps")
        pbar = tqdm(
            enumerate(dataloader_iter, start=sampler_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=sampler_step,
            total=real_epoch_len,
            file=sys.stdout,
        )
        for step, batch in pbar:
            if batch is None:
                continue

            timer_list = []

            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)  # (B, F, C, H, W)
            pixel_values = pixel_values.permute(0, 2, 1, 3, 4).contiguous()  # -> (B, C, F, H, W)
            model_input = pixel_values

            def calc_loss():
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                p_t, p_h, p_w = patch_size
                if model_input.shape[2] % p_t != 0 or model_input.shape[3] % p_h != 0 or model_input.shape[4] % p_w != 0:
                    raise ValueError("Input dimensions must be divisible by the DiT patch size.")
                post_patch_num_frames = model_input.shape[2] // p_t
                tokens_per_frame = (model_input.shape[3] // p_h) * (model_input.shape[4] // p_w)
                seq_len = post_patch_num_frames * tokens_per_frame

                # Sample random timesteps
                sample_batch = bsz
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=cfg.weighting_scheme,
                    batch_size=sample_batch,
                    logit_mean=cfg.logit_mean,
                    logit_std=cfg.logit_std,
                    mode_scale=cfg.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                indices = torch.clamp(indices, 0, noise_scheduler_copy.config.num_train_timesteps - 1)
                timesteps_table = noise_scheduler_copy.timesteps.to(device=model_input.device)
                sigmas_table = noise_scheduler_copy.sigmas.to(device=model_input.device, dtype=model_input.dtype)

                timesteps = timesteps_table[indices]
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                patch_is_noisy_mask = build_condition_mask(
                    batch_size=bsz,
                    post_patch_frames=post_patch_num_frames,
                    device=model_input.device,
                    task_probs=conditioning_task_probs,
                )
                if patch_is_noisy_mask is not None:
                    timestep_groups = timesteps[:, None].expand(bsz, post_patch_num_frames)
                    zero_groups = torch.zeros_like(timestep_groups)
                    timestep_groups = torch.where(patch_is_noisy_mask, timestep_groups, zero_groups)
                    timestep_tokens = timestep_groups.repeat_interleave(tokens_per_frame, dim=1)
                else:
                    timestep_tokens = timesteps[:, None].expand(bsz, seq_len)

                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                if patch_is_noisy_mask is not None:
                    is_noisy_mask = patch_is_noisy_mask.repeat_interleave(p_t, dim=1)
                    if not is_noisy_mask.all():
                        is_noisy_mask_bc = is_noisy_mask[:, None, :, None, None]
                        noisy_model_input = torch.where(is_noisy_mask_bc, noisy_model_input, model_input)
                        sigmas = torch.where(is_noisy_mask_bc, sigmas, torch.zeros_like(sigmas))

                # Predict the noise residual
                model_pred = transformer(noisy_model_input, timestep=timestep_tokens).sample

                # Follow: Section 5 of https://huggingface.co/papers/2206.00364.
                # Preconditioning of the model outputs.
                if cfg.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=cfg.weighting_scheme,
                    sigmas=sigmas,
                )

                # flow matching loss
                if cfg.precondition_outputs:
                    target = model_input
                else:
                    target = noise - model_input

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                return loss

            with timers["forward"] as forward_t:
                loss = calc_loss()
            if record_time:
                timer_list.append(forward_t)

            loss_scale = loss / cfg.gradient_accumulation
            should_update = ((step + 1) % cfg.gradient_accumulation) == 0

            with timers["backward"] as backward_t:
                booster.backward(loss_scale, optimizer)
                if should_update:
                    optimizer.step()
                    optimizer.zero_grad()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
            if record_time:
                timer_list.append(backward_t)

            reduced_loss = all_reduce_mean(loss.detach())
            log_dict = {
                "main/loss": reduced_loss.item(),
                "main/lr": optimizer.param_groups[0]["lr"],
                "main/global_step": global_step,
                "main/epoch": epoch,
                "main/step": step,
            }
            if record_time:
                for timer in timer_list:
                    log_dict[f"debug/{timer.name}"] = timer.elapsed_time
            train_io.log_dict(log_dict, step=global_step)
            pbar.set_postfix({"loss": reduced_loss.item(), "global_step": global_step})
            if record_time:
                log_str = f"Rank {dist.get_rank()} | Epoch {epoch} | Step {step} | "
                for timer in timer_list:
                    log_str += f"{timer.name}: {timer.elapsed_time:.3f}s | "
                print(log_str)

            if cfg.log_video_every > 0 and (global_step + 1) % cfg.log_video_every == 0:
                coordinator.block_all()
                if coordinator.is_master():
                    train_io.logger.info(f"Logging validation at epoch {epoch}, step {step + 1}, global_step {global_step + 1}")
                    log_validation(transformer.unwrap(), noise_scheduler, cfg, train_io, global_step, height, width, num_frames)
                coordinator.block_all()

            if cfg.ckpt_every > 0 and (global_step + 1) % cfg.ckpt_every == 0:
                coordinator.block_all()
                save_dir = train_io.save(
                    booster,
                    model=transformer,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    epoch=epoch,
                    step=step + 1,
                    global_step=global_step + 1,
                    batch_size=cfg.batch_size,
                )
                if coordinator.is_master():
                    pipeline = DiTPipeline(
                        transformer=transformer.unwrap(),
                        scheduler=noise_scheduler,
                    )
                    pipeline.save_pretrained(save_dir + "/pipeline")
                coordinator.block_all()
                if coordinator.is_master():
                    train_io.logger.info(f"Saved checkpoint at epoch {epoch}, step {step + 1}, global_step {global_step + 1} to {save_dir}")

            global_step += 1

        train_io.logger.info(f"Finished epoch {epoch}")
        start_step = 0

    train_io.close()


if __name__ == "__main__":
    parser = HfArgumentParser(FullTrainingArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        cfg = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        cfg = parser.parse_args_into_dataclasses()[0]
    main(cfg)
