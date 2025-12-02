FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV TZ=Asia/Tokyo \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    ninja-build \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    pkg-config \
    python3 \
    python3-venv \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- && \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv

RUN uv python install 3.11

WORKDIR /workspace/minisora

CMD ["/bin/bash"]
