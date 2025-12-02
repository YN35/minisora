"""Diffusion-style pipelines for the custom DiT models used in minisora."""
import importlib
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import os
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput, is_torch_xla_available, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_loading_utils import CONNECTED_PIPES_KEYS
from huggingface_hub import ModelCard, snapshot_download
from huggingface_hub.utils import validate_hf_hub_args

from .modeling_dit import DiTModel

logger = logging.get_logger(__name__)

if is_torch_xla_available():  # pragma: no cover - optional runtime dependency
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


@dataclass
class DiTPipelineOutput(BaseOutput):
    latents: torch.FloatTensor


def retrieve_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, int]:
    r"""Diffusers-style helper that retrieves the concrete timesteps from a scheduler.

    This mirrors :func:`diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps` to keep
    the implementation self-contained without importing additional modules.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be provided.")

    def _accepts(param_name: str) -> bool:
        return param_name in set(inspect.signature(scheduler.set_timesteps).parameters.keys())

    if timesteps is not None:
        if not _accepts("timesteps"):
            raise ValueError(f"{scheduler.__class__.__name__} does not accept custom `timesteps`.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        resolved_timesteps = scheduler.timesteps
        num_inference_steps = len(resolved_timesteps)
    elif sigmas is not None:
        if not _accepts("sigmas"):
            raise ValueError(f"{scheduler.__class__.__name__} does not accept custom `sigmas`.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        resolved_timesteps = scheduler.timesteps
        num_inference_steps = len(resolved_timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        resolved_timesteps = scheduler.timesteps

    return resolved_timesteps, num_inference_steps


class DiTPipeline(DiffusionPipeline):
    r"""Minimal video diffusion pipeline that wraps :class:`DiTModel`.

    Args:
        transformer: The DiT backbone responsible for predicting noise / flow updates.
        scheduler: Flow matcher scheduler controlling the denoising trajectory.
    """

    model_cpu_offload_seq = "transformer"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer: DiTModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler)

    def check_inputs(
        self,
        batch_size: int,
        callback_steps: int,
        output_type: str,
        num_frames: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("`batch_size` must be >= 1.")
        if callback_steps < 1:
            raise ValueError("`callback_steps` must be >= 1.")
        if output_type != "latent":
            raise ValueError("Only `output_type='latent'` is supported.")
        if num_frames < 1:
            raise ValueError("`num_frames` must be >= 1.")
        config_frames = getattr(self.transformer.config, "num_frames", None)
        if config_frames is not None and num_frames != config_frames:
            raise ValueError(f"`num_frames` must match the DiT config ({config_frames}).")

    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        shape = (batch_size, num_channels, num_frames, height, width)  # (B, C, F, H, W)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"`latents` must have shape {shape}, but received {tuple(latents.shape)}.")
            latents = latents.to(device=device, dtype=dtype)
        init_sigma = getattr(self.scheduler, "init_noise_sigma", 1.0)
        return latents * init_sigma

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        height: Optional[int] = 64,
        width: Optional[int] = 64,
        num_frames: Optional[int] = 16,
        output_type: str = "latent",
        return_dict: bool = True,
        callback_steps: int = 1,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        condition_latents: Optional[torch.FloatTensor] = None,
        condition_mask: Optional[torch.BoolTensor] = None,
        callback_on_step_end: Optional[Callable[["DiTPipeline", int, torch.Tensor, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ) -> Union[DiTPipelineOutput, Tuple]:
        """Standard diffusers-style denoising call.

        Args:
            batch_size: Number of videos to sample.
            num_inference_steps: Scheduler steps to take.
            generator: Optional random generator(s).
            latents: Optional custom initial latents.
            height: Video height in pixels.
            width: Video width in pixels.
            num_frames: Number of frames per sample.
            output_type: Currently only ``"latent"`` is supported.
            return_dict: When ``True`` return :class:`DiTPipelineOutput`.
            callback_steps: Frequency for calling ``callback_on_step_end``.
            timesteps / sigmas: Optional custom timestep schedules.
            condition_latents: Latents to fix for context frames.
            condition_mask: Boolean mask selecting the conditioned frames.
            callback_on_step_end: Optional hook invoked after each scheduler step.
            callback_on_step_end_tensor_inputs: Names of tensors provided to the hook.
        """
        self.check_inputs(
            batch_size=batch_size,
            callback_steps=callback_steps,
            output_type=output_type,
            num_frames=num_frames,
        )

        device = self._execution_device
        dtype = self.transformer.dtype

        num_channels = self.transformer.config.in_channels

        scheduler_config = getattr(self.scheduler, "config", {})
        if hasattr(scheduler_config, "get") and scheduler_config.get("use_dynamic_shifting", False):
            raise ValueError("`FlowMatchEulerDiscreteScheduler` with dynamic shifting is not supported in this pipeline.")

        timesteps_tensor, _ = retrieve_timesteps(
            self.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
            timesteps=timesteps,
            sigmas=sigmas,
        )

        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels=num_channels,
            num_frames=num_frames,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        conditioning_latents = None
        frame_condition_mask = None
        frame_condition_mask_bc = None
        token_condition_mask = None
        if condition_latents is not None or condition_mask is not None:
            if condition_latents is None or condition_mask is None:
                raise ValueError("Both `condition_latents` and `condition_mask` must be provided together.")
            expected_shape = latents.shape
            if condition_latents.shape != expected_shape:
                raise ValueError(f"`condition_latents` must have shape {expected_shape}, got {condition_latents.shape}.")
            if condition_mask.shape != (batch_size, num_frames):
                raise ValueError(
                    f"`condition_mask` must have shape (batch_size, num_frames)=({batch_size}, {num_frames}), " f"got {condition_mask.shape}."
                )
            conditioning_latents = condition_latents.to(device=device, dtype=dtype)
            frame_condition_mask = condition_mask.to(device=device, dtype=torch.bool)
            frame_condition_mask_bc = frame_condition_mask[:, None, :, None, None]
            latents = torch.where(frame_condition_mask_bc, conditioning_latents, latents)

            p_t, p_h, p_w = self.transformer.config.patch_size
            if num_frames % p_t != 0 or height % p_h != 0 or width % p_w != 0:
                raise ValueError("`condition_mask` is only supported when video dimensions align with the patch size.")
            post_patch_frames = num_frames // p_t
            tokens_per_frame = (height // p_h) * (width // p_w)
            patch_mask = frame_condition_mask.reshape(batch_size, post_patch_frames, p_t).any(dim=2)
            token_condition_mask = patch_mask.repeat_interleave(tokens_per_frame, dim=1)

        for step_index, timestep in enumerate(self.progress_bar(timesteps_tensor)):
            model_timestep = self._prepare_timestep_tensor(timestep, batch_size=batch_size, latents=latents)
            if token_condition_mask is not None:
                model_timestep_tokens = model_timestep[:, None].expand(batch_size, token_condition_mask.shape[1])
                timestep_zeros = torch.zeros_like(model_timestep_tokens)
                model_timestep_input = torch.where(token_condition_mask, timestep_zeros, model_timestep_tokens)
            else:
                model_timestep_input = model_timestep

            latent_model_input = latents

            noise_pred = self.transformer(
                latent_model_input,
                timestep=model_timestep_input,
            ).sample  # inputs stay in (B, C, F, H, W) order

            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
            if frame_condition_mask_bc is not None and conditioning_latents is not None:
                latents = torch.where(frame_condition_mask_bc, conditioning_latents, latents)
            if latents.dtype != latents_dtype:
                latents = latents.to(dtype=latents_dtype)

            if callback_on_step_end is not None and (step_index % max(callback_steps, 1) == 0 or step_index == len(timesteps_tensor) - 1):
                callback_kwargs: Dict[str, torch.Tensor] = {}
                available_tensors = {
                    "latents": latents,
                    "noise_pred": noise_pred,
                    "model_output": noise_pred,
                }
                for tensor_name in callback_on_step_end_tensor_inputs:
                    tensor = available_tensors.get(tensor_name)
                    if tensor is not None:
                        callback_kwargs[tensor_name] = tensor
                callback_outputs = callback_on_step_end(self, step_index, timestep, callback_kwargs)
                if isinstance(callback_outputs, dict) and "latents" in callback_outputs:
                    latents = callback_outputs["latents"]

            if XLA_AVAILABLE:
                xm.mark_step()

        images = latents
        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)

        return DiTPipelineOutput(latents=images)

    def _prepare_timestep_tensor(
        self,
        timestep: Union[int, float, torch.Tensor],
        batch_size: int,
        latents: torch.FloatTensor,
    ) -> torch.Tensor:
        device = latents.device
        if not torch.is_tensor(timestep):
            dtype = torch.float32 if isinstance(timestep, float) else torch.long
            timestep_tensor = torch.tensor([timestep], device=device, dtype=dtype)
        else:
            timestep_tensor = timestep.to(device=device)
            if timestep_tensor.ndim == 0:
                timestep_tensor = timestep_tensor[None]
        if timestep_tensor.shape[0] != batch_size:
            timestep_tensor = timestep_tensor.expand(batch_size)
        return timestep_tensor


    @classmethod
    @validate_hf_hub_args
    def download(cls, pretrained_model_name, **kwargs) -> Union[str, os.PathLike]:
        r"""
        Download and cache a PyTorch diffusion pipeline from pretrained pipeline weights.

        Parameters:
            pretrained_model_name (`str` or `os.PathLike`, *optional*):
                A string, the *repository id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                hosted on the Hub.
            custom_pipeline (`str`, *optional*):
                Can be either:

                    - A string, the *repository id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained
                      pipeline hosted on the Hub. The repository must contain a file called `pipeline.py` that defines
                      the custom pipeline.

                    - A string, the *file name* of a community pipeline hosted on GitHub under
                      [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file
                      names must match the file name and not the pipeline script (`clip_guided_stable_diffusion`
                      instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the
                      current `main` branch of GitHub.

                    - A path to a *directory* (`./my_pipeline_directory/`) containing a custom pipeline. The directory
                      must contain a file called `pipeline.py` that defines the custom pipeline.

                <Tip warning={true}>

                🧪 This is an experimental feature and may change in the future.

                </Tip>

                For more information on how to load and create custom pipelines, take a look at [How to contribute a
                community pipeline](https://huggingface.co/docs/diffusers/main/en/using-diffusers/contribute_pipeline).

            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            use_onnx (`bool`, *optional*, defaults to `False`):
                If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights
                will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is
                `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending
                with `.onnx` and `.pb`.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom pipelines and components defined on the Hub in their own files. This
                option should only be set to `True` for repositories you trust and in which you have read the code, as
                it will execute code present on the Hub on your local machine.

        Returns:
            `os.PathLike`:
                A path to the downloaded pipeline.

        <Tip>

        To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with
        `huggingface-cli login`.

        </Tip>

        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        if use_safetensors is None:
            use_safetensors = True

        allow_patterns = None
        ignore_patterns = None

        model_info_call_error: Optional[Exception] = None

        user_agent = {"pipeline_class": cls.__name__}
        if custom_pipeline is not None and not custom_pipeline.endswith(".py"):
            user_agent["custom_pipeline"] = custom_pipeline

        # download all allow_patterns - ignore_patterns
        try:
            cached_folder = snapshot_download(
                pretrained_model_name,
                cache_dir=cache_dir,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                user_agent=user_agent,
            )

            # retrieve pipeline class from local file
            cls_name = cls.load_config(os.path.join(cached_folder, "model_index.json")).get("_class_name", None)
            cls_name = cls_name[4:] if isinstance(cls_name, str) and cls_name.startswith("Flax") else cls_name

            diffusers_module = importlib.import_module(__name__.split(".")[0])
            pipeline_class = getattr(diffusers_module, cls_name, None) if isinstance(cls_name, str) else None

            if pipeline_class is not None and pipeline_class._load_connected_pipes:
                modelcard = ModelCard.load(os.path.join(cached_folder, "README.md"))
                connected_pipes = sum([getattr(modelcard.data, k, []) for k in CONNECTED_PIPES_KEYS], [])
                for connected_pipe_repo_id in connected_pipes:
                    download_kwargs = {
                        "cache_dir": cache_dir,
                        "force_download": force_download,
                        "proxies": proxies,
                        "local_files_only": local_files_only,
                        "token": token,
                        "variant": variant,
                        "use_safetensors": use_safetensors,
                    }
                    DiffusionPipeline.download(connected_pipe_repo_id, **download_kwargs)

            return cached_folder

        except FileNotFoundError:
            # Means we tried to load pipeline with `local_files_only=True` but the files have not been found in local cache.
            # This can happen in two cases:
            # 1. If the user passed `local_files_only=True`                    => we raise the error directly
            # 2. If we forced `local_files_only=True` when `model_info` failed => we raise the initial error
            if model_info_call_error is None:
                # 1. user passed `local_files_only=True`
                raise
            else:
                # 2. we forced `local_files_only=True` when `model_info` failed
                raise EnvironmentError(
                    f"Cannot load model {pretrained_model_name}: model is not cached locally and an error occurred"
                    " while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace"
                    " above."
                ) from model_info_call_error

__all__ = ["DiTPipeline", "DiTPipelineOutput"]
