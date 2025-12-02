# minisora: A minimal & Scalable PyTorch re-implementation of the OpenAI Sora training

<p align="center">
<img src="assets/logo.png" width="300" alt="minisora logo (placeholder)"/>
</p>

<p align="center">
<b>A Minimal & Scalable PyTorch Implementation of DiT Video Generation</b>
</p>

<p align="center">
<a href="https://huggingface.co/ramu0e/minisora-dmlab">
<img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow" alt="Hugging Face">
</a>
<a href="https://github.com/YN35/minisora">
<img src="https://img.shields.io/badge/GitHub-Repo-black" alt="GitHub">
</a>
<a href="https://pytorch.org/">
<img src="https://img.shields.io/badge/PyTorch-v2.0+-EE4C2C" alt="PyTorch">
</a>
<a href="https://www.google.com/search?q=LICENSE">
<img src="https://img.shields.io/badge/License-MIT-blue" alt="License">
</a>
</p>

-----

## 📖 Introduction

**minisora** is a minimalist, educational, yet scalable re-implementation of the training process behind OpenAI's Sora (Diffusion Transformers). This project aims to strip away the complexity of large-scale video generation codebases while maintaining the ability to train on multi-node clusters.

It leverages **ColossalAI** for distributed training efficiency and **Diffusers** for a standardized inference pipeline.

### ✨ Key Features

  * **🚀 Scalable Training**: Built on ColossalAI to support multi-node, multi-GPU training out of the box.
  * **🧩 Simple & Educational**: The codebase is designed to be readable and hackable, avoiding "spaghetti code" common in research repos.
  * **🎬 Video Continuation**: Supports not just text-to-video, but also extending existing video clips (autoregressive-style generation in latent space).
  * **🛠️ Modern Tooling**: Uses `uv` for fast dependency management and Docker for reproducible environments.

-----

## 🎥 Demos

<div align="center">
<table>
<tr>
<td align="center"><b>Unconditional Generation</b></td>
<td align="center"><b>Video Continuation</b></td>
</tr>
<tr>
<td align="center"><img src="assets/demo_i2v.gif" width="100%" alt="Random Video Generation"></td>
<td align="center"><img src="assets/demo_continuation.gif" width="100%" alt="Video Continuation"></td>
</tr>
</table>
</div>

-----

## 🏗️ Architecture

minisora implements a Latent Diffusion Transformer (DiT). It processes video latents as a sequence of patches, handling both spatial and temporal dimensions via attention mechanisms.

![Diffusion Transformer architecture](assets/architecture.png)

The library is organized to separate the model definition from the training logic:

  * `minisora/models`: Contains the DiT implementation and pipeline logic.
  * `minisora/data`: Data loading logic for DMLab and Minecraft datasets.
  * `scripts/`: Training and inference entry points.

-----


## ⬇️ Model Zoo

| Model Name | Dataset | Resolution | Frames | Download |
| :--- | :--- | :--- | :--- | :--- |
| **minisora-dmlab** | DeepMind Lab | $64 \times 64$ | 20 | [🤗 Hugging Face](https://huggingface.co/ramu0e/minisora-dmlab) |
| **minisora-minecraft** | Minecraft | $128 \times 128$ | 20 | *(Coming Soon)* |

-----

## 🚀 Quick Start

### Installation

We recommend using `uv` for lightning-fast dependency management.

```bash
git clone https://github.com/YN35/minisora
cd minisora

# Install dependencies including dev tools
uv sync --dev
```

### Inference (Python)

You can generate video using the pre-trained weights hosted on Hugging Face.

```python
from minisora.models import DiTPipeline

# Load the pipeline
pipeline = DiTPipeline.from_pretrained("ramu0e/minisora-dmlab")

# Run inference
output = pipeline(
    batch_size=1,
    num_inference_steps=28,
    height=64,
    width=64,
    num_frames=20,
)

# Access the latents or decode them
latents = output.latents  # shape: (B, C, F, H, W)
print(f"Generated video latents shape: {latents.shape}")
```

### Run Demos

```bash
# Random unconditional generation
uv run scripts/demo/full_vgen.py

# Continuation (fixing the first frame and generating the rest)
uv run scripts/demo/full_continuation.py
```

-----

## 🏋️ Training

We provide a containerized workflow to ensure reproducibility.

### 1\. Environment Setup (Docker)

Start the development container:

```bash
docker compose up -d
```

> **Tip:** You can mount your local data directories by editing `docker-compose.override.yml`:
>
> ```yaml
> services:
>   minisora:
>     volumes:
>       - .:/workspace/minisora
>       - /path/to/your/data:/data
> ```

### 2\. Data Preparation

Download the sample datasets (DMLab or Minecraft) to your data directory:

```bash
# Example: Downloading DMLab dataset
uv run bash scripts/download/dmlab.sh /data/minisora

# Example: Downloading Minecraft dataset
uv run bash scripts/download/minecraft.sh /data/minisora
```

### 3\. Run Training Job

Training is launched via `torchrun`. The following command starts a single-node training job.

```bash
# Set your GPU ID
export CUDA_VISIBLE_DEVICES=0

# Start training
nohup uv run torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  scripts/train.py --dataset_type=dmlab > outputs/train.log 2>&1 &
```

You can monitor the progress in `outputs/train.log`. Change `--dataset_type` to `minecraft` to train on the Minecraft dataset.

-----

## 🗓️ Todo & Roadmap

  - [x] Basic DiT Implementation
  - [x] Integration with Diffusers Pipeline
  - [x] Multi-node training with ColossalAI
  - [x] Video Continuation support

-----

## 🤝 Acknowledgements

  * **[ColossalAI](https://github.com/hpcaitech/ColossalAI)**: For making distributed training accessible.
  * **[Diffusers](https://github.com/huggingface/diffusers)**: For the robust diffusion pipeline structure.
  * **[DiT Paper](https://arxiv.org/abs/2212.09748)**: Scalable Diffusion Models with Transformers.

-----

## 📄 License

This project is licensed under the MIT License. See [LICENSE](https://www.google.com/search?q=LICENSE) for details.