"""Convenience builder for ``DiTModel`` with size presets."""

from dataclasses import dataclass
from typing import Dict

from .modeling_dit import DiTModel


@dataclass(frozen=True)
class DiTSpec:
    hidden_size: int
    num_layers: int
    num_heads: int
    freq_dim: int


DIT_MODEL_SPECS: Dict[str, DiTSpec] = {
    # Roughly aligned with common DiT naming conventions.
    "tiny": DiTSpec(hidden_size=256, num_layers=8, num_heads=4, freq_dim=128),
    "small": DiTSpec(hidden_size=384, num_layers=12, num_heads=6, freq_dim=128),
    "base": DiTSpec(hidden_size=768, num_layers=12, num_heads=12, freq_dim=256),
    "large": DiTSpec(hidden_size=1024, num_layers=24, num_heads=16, freq_dim=256),
    "xl": DiTSpec(hidden_size=1152, num_layers=28, num_heads=16, freq_dim=256),
}


def build_dit(
    model_type: str,
    in_channels: int = 3,
    out_channels: int = 3,
    attn_implementation: str = "flex_attention",
) -> DiTModel:
    """Build a :class:`DiTModel` while switching sizes via ``model_type``."""
    try:
        spec = DIT_MODEL_SPECS[model_type]
    except KeyError as exc:
        raise ValueError(f"model_type={model_type} is not supported") from exc

    if spec.hidden_size % spec.num_heads != 0:
        raise ValueError(f"hidden_size ({spec.hidden_size}) must be divisible by num_heads ({spec.num_heads}).")
    attention_head_dim = spec.hidden_size // spec.num_heads
    ffn_dim = int(spec.hidden_size * 4)

    return DiTModel(
        patch_size=(2, 4, 4),
        in_channels=in_channels,
        out_channels=out_channels,
        num_attention_heads=spec.num_heads,
        attention_head_dim=attention_head_dim,
        ffn_dim=ffn_dim,
        num_layers=spec.num_layers,
        attn_implementation=attn_implementation,
    )


# Wan1 / Wan2.1 系 → 時間×空間 = 4×8×8
# Wan2.2 → 時間×空間 = 4×16×16
# Open-Sora 1.x → 時間×空間 = 4×8×8
# Open-Sora 2.0 → 時間×空間 = 4×32×32

# 2x4x4 のばあい small で 10 時間で学習終了。
# これより小さくすると時間かかりすぎちゃう。
# 1x2x2 のばあい small で 150 時間で学習終了。
