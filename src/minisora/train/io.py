"""Training I/O helpers (logging, checkpoints, media export)."""

import json
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import write_video

import wandb


class TrainIO:
    """Utility wrapper for logging, checkpoints, and experiment bookkeeping."""

    def __init__(self, cfg, coordinator: DistCoordinator, project: str) -> None:
        self.coordinator = coordinator
        timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%m%d-%H%M")
        exp_name = f"{cfg.experiment_name}-{timestamp}"
        self.exp_dir = Path(cfg.outputs) / project / exp_name
        if coordinator.is_master():
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            with open(self.exp_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg.__dict__ if hasattr(cfg, "__dict__") else dict(cfg), f, indent=2)
        coordinator.block_all()

        self.logger = logging.getLogger(exp_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        if coordinator.is_master():
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            file_handler = logging.FileHandler(self.exp_dir / "train.log")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
            self.logger.addHandler(file_handler)
        else:
            self.logger.addHandler(logging.NullHandler())

        self.tb_writer: Optional[SummaryWriter] = None
        if coordinator.is_master():
            self.tb_writer = SummaryWriter(log_dir=self.exp_dir / "tensorboard")

        self.wandb_run = None
        if coordinator.is_master() and cfg.wandb:
            self.wandb_run = wandb.init(project=cfg.wandb_project, name=exp_name, config=cfg.__dict__, dir=str(self.exp_dir))

    def log_dict(self, metrics: dict, step: Optional[int] = None) -> None:
        if not self.coordinator.is_master():
            return
        if self.tb_writer is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
        if self.wandb_run is not None:
            wandb.log(metrics, step=step)

    def log_video(
        self,
        name: str,
        video: torch.Tensor,
        min_pix: float,
        max_pix: float,
        fps: int,
        step: Optional[int] = None,
    ) -> None:
        if not self.coordinator.is_master():
            return
        if video.ndim != 4:
            raise ValueError("Video tensor must have shape (C, T, H, W).")
        diff = max_pix - min_pix
        if diff <= 0:
            raise ValueError("`max_pix` must be greater than `min_pix` for logging.")
        video = video.detach().cpu()
        norm_video = torch.clamp((video - min_pix) / diff, 0.0, 1.0)
        if self.tb_writer is not None:
            self.tb_writer.add_video(name, norm_video.unsqueeze(0), global_step=step, fps=fps)
        media_dir = self.exp_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        video_uint8 = (norm_video * 255).to(torch.uint8)
        thwc = video_uint8.permute(1, 2, 3, 0).contiguous()
        name = name.replace("/", "_")
        write_video(str(media_dir / f"{step or 0}_{name}.mp4"), thwc, fps=fps)

    def save(
        self,
        booster,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        lr_scheduler: Optional[_LRScheduler],
        epoch: int,
        step: int,
        global_step: int,
        batch_size: int,
    ) -> str:
        save_dir = self.exp_dir / f"epoch{epoch}-global_step{global_step}"
        (save_dir / "model").mkdir(parents=True, exist_ok=True)
        booster.save_model(model, str(save_dir / "model"), shard=True)
        if optimizer is not None:
            booster.save_optimizer(optimizer, str(save_dir / "optimizer"), shard=True, size_per_shard=4096)
        if lr_scheduler is not None:
            booster.save_lr_scheduler(lr_scheduler, str(save_dir / "lr_scheduler"))
        if dist.get_rank() == 0:
            with open(save_dir / "running_states.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "step": step,
                        "global_step": global_step,
                        "batch_size": batch_size,
                    },
                    f,
                    indent=2,
                )
        dist.barrier()
        return str(save_dir)

    def load(
        self,
        booster,
        load_dir: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        lr_scheduler: Optional[_LRScheduler],
    ) -> Tuple[int, int, int]:
        load_path = Path(load_dir)
        states_path = load_path / "running_states.json"
        if not states_path.exists():
            raise FileNotFoundError(f"{states_path} does not exist.")
        booster.load_model(model, str(load_path / "model"))
        if optimizer is not None:
            booster.load_optimizer(optimizer, str(load_path / "optimizer"))
        if lr_scheduler is not None:
            booster.load_lr_scheduler(lr_scheduler, str(load_path / "lr_scheduler"))
        with open(states_path, "r", encoding="utf-8") as f:
            states = json.load(f)
        dist.barrier()
        return states["epoch"], states["step"], states["global_step"]

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb_run is not None:
            wandb.finish()
