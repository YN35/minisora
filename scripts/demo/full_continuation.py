#!/usr/bin/env python3
"""Minimal video continuation demo for the DiT pipeline."""

from pathlib import Path

import torch

from minisora.models import DiTPipeline
from minisora.utils import build_condition_from_video, save_latents_as_video

INPUT_VIDEO = Path("assets/example_dmlab.mp4")
OUTPUT_PATH = Path("outputs/demo_continuation.mp4")
NUM_FRAMES = 20
PREFIX_FRAMES = 8
HEIGHT = 64
WIDTH = 64
NUM_STEPS = 28
FPS = 6
SEED = 1235
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32


def main() -> None:
    pipeline = DiTPipeline.from_pretrained("ramu0e/minisora-dmlab", torch_dtype=DTYPE).to(DEVICE)

    condition_latents, condition_mask, _ = build_condition_from_video(
        INPUT_VIDEO,
        prefix_frames=PREFIX_FRAMES,
        total_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        device=DEVICE,
        dtype=DTYPE,
    )

    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    output = pipeline(
        batch_size=condition_latents.shape[0],
        num_inference_steps=NUM_STEPS,
        generator=generator,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        condition_latents=condition_latents,
        condition_mask=condition_mask,
    )
    save_latents_as_video(output.latents, OUTPUT_PATH, fps=float(FPS), condition_mask=condition_mask)
    print(f"[continuation] saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
