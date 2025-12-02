#!/usr/bin/env python3
"""Minimal image-to-video demo for the DiT pipeline."""

from pathlib import Path

import torch

from minisora.models import DiTPipeline
from minisora.utils import save_latents_as_video


OUTPUT_PATH = Path("outputs/demo_i2v.mp4")
NUM_FRAMES = 20
HEIGHT = 64
WIDTH = 64
NUM_STEPS = 28
FPS = 6
SEED = 1234
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32


def main() -> None:
    pipeline = DiTPipeline.from_pretrained("ramu0e/minisora-dmlab", torch_dtype=DTYPE).to(DEVICE)

    output = pipeline(
        batch_size=1,
        num_inference_steps=NUM_STEPS,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
    )

    save_latents_as_video(output.latents, OUTPUT_PATH, fps=float(FPS))
    print(f"[vgen] saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
