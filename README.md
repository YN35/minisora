
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
  <video src="assets/demo_i2v.mp4" width="45%" controls loop muted playsinline>
    Your browser does not support the video tag.
  </video>
  <video src="assets/demo_continuation.mp4" width="45%" controls loop muted playsinline>
    Your browser does not support the video tag.
  </video>
</p>

Minisora は、小さめの DiT ベース動画生成モデルを ColossalAI + Diffusers で学習・推論するためのリポジトリです。

研究・検証用途の「ミニ Sora」的なポジションを目指しており、単一 GPU でも
「データ用意 → 学習 → 推論 → 可視化」まで一通り試せるように設計されています。

## 特徴

- DMLab / Minecraft などの軌跡データセットから動画生成モデルを学習
- 学習済みウェイト（例: `ramu0e/minisora-dmlab`）を使って、1 GPU から動画生成を試せる
- 先頭数フレームを与えて「続き」や、フレームマスクを使った補完生成ができる

---

## クイックスタート

「とりあえず動く動画生成を見たい」場合は、次のコマンドだけで OK です。

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

`outputs/demo_i2v.mp4` と `outputs/demo_continuation.mp4` が生成されます。

より細かい設定や学習ジョブの回し方は、以降のセクションで説明します。

以下では、環境構築 → データ取得 → 学習 → 推論（動画生成）の流れをまとめます。

---

## 環境構築と学習フロー

このリポジトリで学習を行うまでの、ざっくりとした流れは次の通りです。

1. 依存関係のインストール（`uv` を利用）
2. Docker コンテナの起動
3. データセットのダウンロード
4. 学習ジョブの実行

以下では、上記の順番でコマンドと補足説明をまとめています。

---

## 1. 事前準備

- CUDA / GPU が利用できる環境を前提としています。
- Python や `uv` はホスト側にインストールされているものを利用します。
- Docker / Docker Compose がインストールされている必要があります。

リポジトリの取得例（この minisora 用リポジトリ）:

```bash
git clone https://github.com/YN35/minisora
cd minisora  # 実際のクローン先ディレクトリ名に合わせてください
```

---

## 2. 依存関係のインストール

`uv` を使って開発用の依存関係までまとめてインストールします。

```bash
uv sync --dev
```

---

## 3. Docker コンテナの起動・停止

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

## 4. データセットのダウンロード

`/data/minisora` 以下にデータセットをダウンロードする例です。
`/data/minisora` 部分はマウントしているホスト側ディレクトリに合わせて変更してください。

```bash
uv run bash scripts/download/dmlab.sh /data/minisora
uv run bash scripts/download/minecraft.sh /data/minisora
```

---

## 5. 学習の実行

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

## 6. 推論・動画生成

学習済みの DiT モデルを使って、Python から直接動画を生成できます。

### 6.1 Python からの最小実行例

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

- `DiTPipeline.from_pretrained(...)` で Hugging Face Hub 上の `ramu0e/minisora-dmlab` を読み込みます。
- `pipeline(...)` を呼び出すと、拡散過程を通してランダムな動画の「潜在表現」 `latents` を生成します。
  - 形状は `(batch, channels=3, frames, height, width)` です。
  - 値の範囲はおおよそ `[-1, 1]` で、`minisora.utils.save_latents_as_video` を使ってそのまま動画として保存できます。

GPU を使う場合の例:

```python
import torch
from minisora.models import DiTPipeline
from minisora.utils import save_latents_as_video

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

pipeline = DiTPipeline.from_pretrained(
    "ramu0e/minisora-dmlab",
    torch_dtype=DTYPE,
).to(DEVICE)

output = pipeline(
    batch_size=1,
    num_inference_steps=28,
    height=64,
    width=64,
    num_frames=20,
)

# latents を MP4 動画として保存
save_latents_as_video(output.latents, "outputs/sample.mp4", fps=6.0)
```

### 6.2 `scripts/demo` によるデモ実行

リポジトリには推論用の最小デモスクリプトが用意されています。

- `scripts/demo/full_vgen.py`
  - ランダムノイズから動画を生成する最小デモです。
- `scripts/demo/full_continuation.py`
  - 先頭数フレームを固定し、その続きの動画を生成する「動画 continuation」のデモです。

実行例（コンテナ内などで、依存関係インストール済みを前提）:

```bash
uv run scripts/demo/full_vgen.py
uv run scripts/demo/full_continuation.py
```

- 出力動画はそれぞれ `outputs/demo_i2v.mp4`、`outputs/demo_continuation.mp4` に保存されます。
- continuation デモでは、`assets/example_dmlab.mp4` の先頭 `PREFIX_FRAMES` フレームをコンテキストとして使用します。

---

## 7. マスク付き条件生成（補完・続きの生成）

Minisora の `DiTPipeline` は、時間方向のフレームマスクを使った条件付き生成に対応しています。
コード上では `condition_latents` と `condition_mask` という 2 つの引数で制御します。

### 7.1 引数の役割

- `condition_latents`: 形状 `(B, C, F, H, W)` のテンソル
  - 動画の一部フレームに「既知の画素値」を埋め込んだテンソルです。
  - `[-1, 1]` の範囲で正規化された動画データをそのまま入れます（`minisora.utils.load_video_clip` などが出力する形式）。
- `condition_mask`: 形状 `(B, F)` の `torch.bool` テンソル
  - True のフレームは「条件として固定する（与えられた部分）」、False のフレームは「モデルが生成する部分」を表します。

`DiTPipeline.__call__` 内では、次のような処理が行われます。

- True になっているフレーム位置は、拡散過程の各ステップで必ず `condition_latents` の値に差し戻されます。
- モデルに渡す時間ステップもトークン単位（パッチ単位）でマスクされ、
  条件フレームは「ノイズがほとんど乗っていない状態」として扱われます。

これにより、

- 先頭数フレームだけを固定して「続き」を生成する continuation
- 離れた位置のフレームをいくつか固定して、その間を補完するような生成（時間方向の補完）

といった使い方が可能です（画素レベルの部分的インペインティングではなく、フレーム単位のマスクです）。

### 7.2 動画 continuation の具体例

`scripts/demo/full_continuation.py` では、`minisora.utils.build_condition_from_video` を使って
動画ファイルから条件テンソルとマスクを構築し、先頭フレームを固定した continuation を行っています。

簡略化したコードは次の通りです。

```python
from pathlib import Path
import torch

from minisora.models import DiTPipeline
from minisora.utils import build_condition_from_video, save_latents_as_video

INPUT_VIDEO = Path("assets/example_dmlab.mp4")
NUM_FRAMES = 20
PREFIX_FRAMES = 8
HEIGHT = 64
WIDTH = 64
FPS = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

pipeline = DiTPipeline.from_pretrained(
    "ramu0e/minisora-dmlab",
    torch_dtype=DTYPE,
).to(DEVICE)

# 入力動画から条件フレームとマスクを構築
condition_latents, condition_mask, prefix = build_condition_from_video(
    INPUT_VIDEO,
    prefix_frames=PREFIX_FRAMES,
    total_frames=NUM_FRAMES,
    height=HEIGHT,
    width=WIDTH,
    device=DEVICE,
    dtype=DTYPE,
)

generator = torch.Generator(device=DEVICE).manual_seed(1235)
output = pipeline(
    batch_size=condition_latents.shape[0],
    num_inference_steps=28,
    generator=generator,
    height=HEIGHT,
    width=WIDTH,
    num_frames=NUM_FRAMES,
    condition_latents=condition_latents,
    condition_mask=condition_mask,
)

# 条件フレームには赤枠を付けて保存
save_latents_as_video(output.latents, "outputs/demo_continuation.mp4", fps=float(FPS), condition_mask=condition_mask)
```

ポイント:

- `build_condition_from_video` は、指定した動画ファイルから `prefix_frames` 分だけ読み込み、
  それを先頭に埋め込んだ `condition_latents` と、対応する `condition_mask` を構築します。
- `condition_mask` の True になっているフレーム（ここでは先頭 `PREFIX_FRAMES` 枚）は、生成中も固定され続けます。
- `save_latents_as_video(..., condition_mask=condition_mask)` を指定すると、
  条件フレームに赤い枠が描かれ、どこまでがコンテキストかが分かりやすく可視化されます。

### 7.3 マスクを使った補完生成

`condition_mask` はフレームごとに任意の True/False パターンを取れるため、

- 先頭と末尾のフレームだけを固定し、その間をモデルに補完させる
- 間引きサンプリングしたフレームを固定し、その間の細かい時間ステップを補完する

といった「時間方向の補完生成」も同じインターフェースで表現できます。
その場合は、任意のテンソルから自前で `condition_latents` と `condition_mask` を構築して
`DiTPipeline` に渡してください（形状と値のスケールが通常の `latents` と揃っている必要があります）。
