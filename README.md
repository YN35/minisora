
# minisora

<p align="center">
  <a href="https://github.com/YN35/minisora">GitHub (リポジトリ)</a> |
  <a href="https://x.com/__ramu0e__">X (@__ramu0e__)</a> |
  <a href="https://huggingface.co/ramu0e/minisora-dmlab">Hugging Face (minisora-dmlab)</a>
</p>

<p align="center">
  <b>ランダム動画生成</b> &nbsp;&nbsp;|&nbsp;&nbsp; <b>動画の続き生成</b>
</p>
<p align="center">
  <img src="assets/demo_i2v.gif" width="45%" alt="ランダム動画生成デモ">
  <img src="assets/demo_continuation.gif" width="45%" alt="動画の続き生成デモ">
</p>

Minisora は、小さめの DiT ベース動画生成モデルを ColossalAI + Diffusers で学習・推論するためのリポジトリです。

## 特徴
- **マルチノード学習対応、スケール可能**
- **動画の補完生成にも対応**
- **できる限りシンプルで管理しやすいコードにするために様々な工夫**

---

## クイックスタート


```bash
git clone https://github.com/YN35/minisora
cd minisora

# 依存関係のインストール
uv sync --dev

# Docker コンテナ起動（学習・データ DL 用）
docker compose up -d

# ランダム動画生成デモ
uv run scripts/demo/full_vgen.py

# 先頭フレームを固定した続き生成デモ
uv run scripts/demo/full_continuation.py
```

`outputs/demo_i2v.mp4` と `outputs/demo_continuation.mp4` が生成される。

---

## 環境構築と学習フロー

このリポジトリで学習を行うまでの、ざっくりとした流れは次の通りです。

1. 依存関係のインストール（`uv` を利用）
2. Docker コンテナの起動
3. データセットのダウンロード
4. 学習ジョブの実行

以下では、上記の順番でコマンドと補足説明をまとめています。

---

## 1. 依存関係のインストール

`uv` を使って開発用の依存関係までまとめてインストールします。

```bash
git clone https://github.com/YN35/minisora
cd minisora
uv sync --dev
```

---

## 2. Docker コンテナの起動・停止

学習やデータダウンロードは Docker コンテナ内で行います。

コンテナの起動:

```bash
docker compose up -d
```

コンテナの停止:

```bash
docker compose down
```

コンテナ起動時に、ホスト側のデータディレクトリなどをマウントしたい場合は、
`docker-compose.override.yml` を編集／作成して調整します。

`docker-compose.override.yml` の例:

```yml
services:
  minisora:
    container_name: my-minisora
    volumes:
      - .:/workspace/minisora
      - /home/yn35/data:/data
      - /home/yn35/huggingface_cache:/root/.cache/huggingface
```

---

## 3. データセットのダウンロード

`/data/minisora` 以下にデータセットをダウンロードする例です。
`/data/minisora` 部分はマウントしているホスト側ディレクトリに合わせて変更してください。

```bash
uv run bash scripts/download/dmlab.sh /data/minisora
uv run bash scripts/download/minecraft.sh /data/minisora
```

---

## 4. 学習の実行

GPU を指定して学習ジョブを実行します。`CUDA_VISIBLE_DEVICES` は使用したい GPU ID
に応じて変更してください。

```bash
export CUDA_VISIBLE_DEVICES=3
nohup uv run torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  scripts/train.py --dataset_type=dmlab > outputs/train.log 2>&1 &
```

- ログは `outputs/train.log` に書き出されます。
- バックグラウンドで動作させたくない場合は `nohup` やリダイレクト部分を取り除いて実行してください。
- `dataset_type` は `dmlab` / `minecraft` のいずれかを指定できます（`scripts/train.py` を参照）。

---

## 5. 推論・動画生成

学習済みの DiT モデルを使って、Python から直接動画を生成できます。

### 5.1 Python からの最小実行例

学習済みウェイト `ramu0e/minisora-dmlab` を用いた、64x64・20 フレームの動画サンプル生成例です。

```python
from minisora.models import DiTPipeline

pipeline = DiTPipeline.from_pretrained("ramu0e/minisora-dmlab")

output = pipeline(
    batch_size=1,
    num_inference_steps=28,
    height=64,
    width=64,
    num_frames=20,
)
latents = output.latents  # shape: (B, C, F, H, W)
```
