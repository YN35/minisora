import resource
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.io import read_video, write_video


def ensure_soft_limit(min_limit: int = 8192) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < min_limit:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(min_limit, hard), hard))


def format_numel(numel: int) -> str:
    units = [("B", 1024**3), ("M", 1024**2), ("K", 1024)]
    for suffix, value in units:
        if numel >= value:
            return f"{numel / value:.2f} {suffix}"
    return str(numel)


def get_model_numel(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


def load_video_clip(
    video_path: Union[str, Path],
    *,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Load, resize, and normalize a video clip to [-1, 1] range with shape (B, C, T, H, W)."""
    if num_frames < 1:
        raise ValueError("`num_frames` must be >= 1.")

    video_path = Path(video_path)
    frames, _, _ = read_video(str(video_path), output_format="THWC")
    total_frames = frames.shape[0]
    if total_frames < num_frames:
        raise ValueError(f"{video_path} needs at least {num_frames} frames (found {total_frames}).")

    frames = frames[:num_frames]
    video = frames.permute(0, 3, 1, 2).float() / 255.0
    video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)
    video = video * 2.0 - 1.0
    video = video.permute(1, 0, 2, 3).contiguous()
    return video.unsqueeze(0).to(device=device, dtype=dtype)


def build_condition_from_clip(clip: torch.Tensor, frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract a prefix from a clip and build the matching condition mask."""
    if clip.ndim != 5:
        raise ValueError("`clip` must have shape (B, C, T, H, W).")
    total_frames = clip.shape[2]
    if not 0 < frames <= total_frames:
        raise ValueError("`frames` must be within [1, clip_T].")

    condition_latents = torch.zeros_like(clip)
    condition_latents[:, :, :frames] = clip[:, :, :frames]
    condition_mask = torch.zeros((clip.shape[0], total_frames), dtype=torch.bool, device=clip.device)
    condition_mask[:, :frames] = True
    return condition_latents, condition_mask


def build_condition_from_video(
    video_path: Union[str, Path],
    *,
    prefix_frames: int,
    total_frames: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Convenience helper that loads only the needed prefix frames and expands to the requested context length."""
    if prefix_frames < 1:
        raise ValueError("`prefix_frames` must be >= 1.")
    if total_frames < 1:
        raise ValueError("`total_frames` must be >= 1.")

    conditioning_frames = max(1, min(prefix_frames, total_frames))
    clip_prefix = load_video_clip(
        video_path,
        num_frames=conditioning_frames,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
    )
    prefix = clip_prefix.shape[2]
    full_clip = torch.zeros(
        (clip_prefix.shape[0], clip_prefix.shape[1], total_frames, height, width),
        device=clip_prefix.device,
        dtype=clip_prefix.dtype,
    )
    full_clip[:, :, :prefix] = clip_prefix
    condition_latents, condition_mask = build_condition_from_clip(full_clip, frames=prefix)
    return condition_latents, condition_mask, prefix


def latents_to_video_frames(latents: torch.Tensor) -> torch.Tensor:
    """Convert latents in [-1, 1] to uint8 video frames with shape (B, T, H, W, C)."""
    if latents.ndim != 5:
        raise ValueError("`latents` must have shape (B, C, T, H, W).")
    frames = latents.detach().cpu().clamp(-1, 1)
    frames = ((frames + 1) * 127.5).to(torch.uint8)
    return frames.permute(0, 2, 3, 4, 1).contiguous()


def add_condition_border(
    frames: torch.Tensor,
    condition_mask: torch.Tensor,
    border_px: int = 1,
    color: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Add a colored border around conditioning frames to visualize the prefix context."""
    if frames.ndim != 5:
        raise ValueError("`frames` must have shape (B, T, H, W, C).")
    if condition_mask.ndim != 2:
        raise ValueError("`condition_mask` must have shape (B, T).")
    if frames.shape[:2] != condition_mask.shape:
        raise ValueError("Batch or frame dimension mismatch between `frames` and `condition_mask`.")
    if border_px <= 0:
        return frames

    result = frames.clone()
    color_tensor = color if color is not None else torch.tensor([255, 0, 0], dtype=torch.uint8)
    color_tensor = color_tensor.to(device=result.device, dtype=torch.uint8)
    mask_cpu = condition_mask.to(device="cpu", dtype=torch.bool)

    batch_size, num_frames = mask_cpu.shape
    for batch_idx in range(batch_size):
        conditioned = torch.nonzero(mask_cpu[batch_idx], as_tuple=False).flatten()
        for frame_idx in conditioned.tolist():
            frame = result[batch_idx, frame_idx]
            frame[:border_px, :, :] = color_tensor
            frame[-border_px:, :, :] = color_tensor
            frame[:, :border_px, :] = color_tensor
            frame[:, -border_px:, :] = color_tensor
    return result


def save_latents_as_video(
    latents: torch.Tensor,
    output_path: Union[str, Path],
    fps: float,
    *,
    condition_mask: Optional[torch.Tensor] = None,
    border_px: int = 1,
    video_codec: str = "libx264",
    crf: str = "17",
) -> Path:
    """Convert latents into video frames (optionally annotate) and write them to disk."""
    frames = latents_to_video_frames(latents)
    if frames.shape[0] != 1:
        raise ValueError("`save_latents_as_video` currently supports batch_size=1.")
    if condition_mask is not None:
        frames = add_condition_border(frames, condition_mask, border_px=border_px)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(
        filename=str(output_path),
        video_array=frames[0],
        fps=float(fps),
        video_codec=video_codec,
        options={"crf": str(crf)},
    )
    return output_path
