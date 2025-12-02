"""Dataset utilities for TECO-formatted trajectories."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchcodec.decoders import SimpleVideoDecoder
from tqdm import tqdm


def _compute_frame_selection(
    total_frames: int,
    num_frames: int,
    frame_stride: int,
    source: Optional[str],
) -> Tuple[np.ndarray, int, int]:
    """Return sampled frame indices and contiguous decode window."""
    required = num_frames * frame_stride
    if total_frames < required:
        prefix = f"{source} " if source else ""
        raise ValueError(f"{prefix}has {total_frames} frames < required {required}.")
    max_start = total_frames - required
    start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
    decode_start = start
    decode_stop = start + required
    frame_indices = decode_start + np.arange(0, required, frame_stride, dtype=np.int64)
    return frame_indices, decode_start, decode_stop


def _normalize_clip(frames: torch.Tensor | np.ndarray, resize_hw: Optional[Tuple[int, int]]) -> torch.Tensor:
    """Map raw uint8 frames to (-1, 1) and optionally resize."""
    tensor = torch.as_tensor(frames)
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D frames tensor, got shape {tuple(tensor.shape)}.")
    if tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)
    tensor = 2.0 * (tensor / 255.0) - 1.0
    if tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    elif tensor.shape[1] != 3:
        raise ValueError(f"Frames tensor must have 3 channels, got shape {tuple(tensor.shape)}.")
    if resize_hw is not None:
        tensor = F.interpolate(tensor, size=resize_hw, mode="bilinear", align_corners=False)
    return tensor


@dataclass(frozen=True)
class _MinecraftSample:
    video_path: Path
    action_path: Optional[Path]
    num_frames: int


def _probe_video_num_frames(video_path: Path) -> int:
    decoder = SimpleVideoDecoder(str(video_path))
    num_frames = int(decoder.metadata.num_frames)
    if num_frames <= 0:
        raise ValueError(f"{video_path} contains no decodable frames.")
    return num_frames


def _cache_file_for_root(root: Path) -> Path:
    return root / ".minecraft_video_cache.csv"


def _load_video_cache(cache_path: Path) -> Dict[str, Tuple[int, int]]:
    if not cache_path.exists():
        return {}
    cache: Dict[str, Tuple[int, int]] = {}
    with cache_path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        required = {"relative_path", "num_frames", "mtime_ns"}
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(f"Invalid cache header in {cache_path}.")
        for row in reader:
            rel = row.get("relative_path")
            num_frames = row.get("num_frames")
            mtime_ns = row.get("mtime_ns")
            if rel is None or num_frames is None or mtime_ns is None:
                raise ValueError(f"Malformed cache row in {cache_path}.")
            cache[rel] = (int(num_frames), int(mtime_ns))
    return cache


def _save_video_cache(cache_path: Path, cache: Dict[str, Tuple[int, int]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", newline="") as file:
        fieldnames = ["relative_path", "num_frames", "mtime_ns"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rel_path in sorted(cache.keys()):
            num_frames, mtime_ns = cache[rel_path]
            writer.writerow({"relative_path": rel_path, "num_frames": num_frames, "mtime_ns": mtime_ns})


class DMLabTrajectoryDataset(Dataset):
    """Dataset wrapper for TECO-formatted DMLab trajectories stored as npz."""

    NPZ_EXTS = (".npz",)

    def __init__(
        self,
        roots: Sequence[str],
        num_frames: int,
        frame_stride: int = 1,
        resize_hw: Optional[Tuple[int, int]] = None,
        include_actions: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        if num_frames < 1:
            raise ValueError("`num_frames` must be >= 1.")
        if frame_stride < 1:
            raise ValueError("`frame_stride` must be >= 1.")
        if not roots:
            raise ValueError("`roots` must contain at least one path.")
        self.roots = [Path(root) for root in roots]
        for root in self.roots:
            if not root.exists():
                raise FileNotFoundError(f"Dataset root {root} does not exist.")
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.resize_hw = resize_hw
        self.include_actions = include_actions

        samples: List[Path] = []
        for root in self.roots:
            matches = sorted([path for ext in self.NPZ_EXTS for path in root.rglob(f"*{ext}")])
            samples.extend(matches)

        if not samples:
            raise ValueError(f"No dataset files found under {roots}.")
        if limit is not None:
            samples = samples[:limit]
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        path = self.samples[index]
        with np.load(path, allow_pickle=False) as data:
            if "video" not in data:
                raise KeyError(f"'video' array missing in {path}.")
            video = data["video"]
            actions = data["actions"] if "actions" in data and self.include_actions else None

        frame_indices, _, _ = _compute_frame_selection(video.shape[0], self.num_frames, self.frame_stride, source=str(path))
        clip = _normalize_clip(video[frame_indices], self.resize_hw)
        sample = {"pixel_values": clip}
        if actions is not None:
            if actions.shape[0] < frame_indices[-1] + 1:
                raise ValueError(f"{path} actions have only {actions.shape[0]} frames.")
            sample["actions"] = torch.from_numpy(actions[frame_indices]).long()
        return sample


class MinecraftTrajectoryDataset(Dataset):
    """Dataset wrapper for Minecraft trajectories (video + separate action npz)."""

    VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov")

    def __init__(
        self,
        roots: Sequence[str],
        num_frames: int,
        frame_stride: int = 1,
        resize_hw: Optional[Tuple[int, int]] = None,
        include_actions: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        if num_frames < 1:
            raise ValueError("`num_frames` must be >= 1.")
        if frame_stride < 1:
            raise ValueError("`frame_stride` must be >= 1.")
        if not roots:
            raise ValueError("`roots` must contain at least one path.")
        self.roots = [Path(root) for root in roots]
        for root in self.roots:
            if not root.exists():
                raise FileNotFoundError(f"Dataset root {root} does not exist.")
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.resize_hw = resize_hw
        self.include_actions = include_actions

        samples: List[_MinecraftSample] = []

        def iter_videos(root: Path) -> Iterable[Path]:
            for ext in self.VIDEO_EXTS:
                yield from root.rglob(f"*{ext}")

        for root in self.roots:
            cache_path = _cache_file_for_root(root)
            cache = _load_video_cache(cache_path)
            updated_cache = dict(cache)
            video_paths = sorted(iter_videos(root))
            if limit is not None:
                remaining = limit - len(samples)
                if remaining <= 0:
                    break
                if len(video_paths) > remaining:
                    video_paths = video_paths[:remaining]
            if not video_paths:
                continue
            for video_path in tqdm(video_paths, desc=str(root), unit="video"):
                action_path = video_path.with_suffix(".npz")
                if include_actions and not action_path.exists():
                    raise FileNotFoundError(f"Missing actions npz for {video_path}.")
                rel_key = video_path.relative_to(root).as_posix()
                mtime_ns = video_path.stat().st_mtime_ns
                cached_entry = cache.get(rel_key)
                if cached_entry is not None and cached_entry[1] == mtime_ns:
                    num_frames = cached_entry[0]
                else:
                    num_frames = _probe_video_num_frames(video_path)
                    updated_cache[rel_key] = (num_frames, mtime_ns)
                samples.append(
                    _MinecraftSample(
                        video_path=video_path,
                        action_path=action_path if action_path.exists() else None,
                        num_frames=num_frames,
                    )
                )
            if updated_cache != cache:
                _save_video_cache(cache_path, updated_cache)
            if limit is not None and len(samples) >= limit:
                break

        if not samples:
            raise ValueError(f"No video files found under {roots}.")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample_meta = self.samples[index]
        frame_indices, decode_start, decode_stop = _compute_frame_selection(
            sample_meta.num_frames, self.num_frames, self.frame_stride, source=str(sample_meta.video_path)
        )
        decoder = SimpleVideoDecoder(str(sample_meta.video_path))
        frames_batch = decoder.get_frames_in_range(decode_start, decode_stop, step=self.frame_stride)
        clip = _normalize_clip(frames_batch.data, self.resize_hw)
        sample = {"pixel_values": clip}

        if self.include_actions:
            action_path = sample_meta.action_path
            if action_path is None:
                raise ValueError(f"No actions npz associated with {sample_meta.video_path}.")
            with np.load(action_path, allow_pickle=False) as data:
                if "actions" not in data:
                    raise KeyError(f"'actions' not found in {action_path}.")
                actions = data["actions"]
            if actions.shape[0] < frame_indices[-1] + 1:
                raise ValueError(f"{action_path} has {actions.shape[0]} frames < required {frame_indices[-1] + 1}.")
            sample["actions"] = torch.from_numpy(actions[frame_indices]).long()
        return sample
