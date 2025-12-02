import math
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class DiTTimeEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(self.time_proj.weight)
        timestep_proj = self.time_proj(self.act_fn(temb))

        return temb, timestep_proj


class DiTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_bias: bool = True,
        rms_norm_eps: float = 1e-6,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = False

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=attention_bias)

        # in small model, qk_norm is not used because it is too slow
        # self.q_norm = torch.nn.RMSNorm(self.head_dim, eps=rms_norm_eps)  # unlike olmo, only on the head dim!
        # self.k_norm = torch.nn.RMSNorm(self.head_dim, eps=rms_norm_eps)  # thus post q_norm does not need reshape
        self._attn_implementation = attn_implementation
        # Some attention backends (e.g., flash attention integration) expect a minimal config namespace.
        self.config = SimpleNamespace(_attn_implementation=attn_implementation)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        # key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = eager_attention_forward
        if self._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DiTTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn1 = DiTAttention(
            layer_idx=layer_idx,
            hidden_size=dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            attention_bias=False,
            rms_norm_eps=eps,
            attn_implementation=attn_implementation,
        )

        # 2. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (self.scale_shift_table.unsqueeze(0) + temb).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (self.scale_shift_table + temb).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output, _ = self.attn1(
            norm_hidden_states,
            position_embeddings=rotary_emb,
            **kwargs,
        )
        hidden_states = (hidden_states + attn_output * gate_msa).type_as(hidden_states)

        # 2. Feed-forward
        norm_hidden_states = (self.norm2(hidden_states) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states + ff_output * c_gate_msa).type_as(hidden_states)

        return hidden_states


class DiTModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """Diffusion Transformer backbone that follows the diffusers `ModelMixin` contract."""

    _supports_gradient_checkpointing = True
    _no_split_modules = ["DiTSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 16,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: int = 16,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 28,
        eps: float = 1e-6,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        attn_implementation: str = "flex_attention",
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = DiTTimeEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                DiTTransformerBlock(
                    layer_idx=i,
                    dim=inner_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_attention_heads,
                    eps=eps,
                    attn_implementation=attn_implementation,
                )
                for i in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=eps)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self._attn_implementation = attn_implementation

        self.gradient_checkpointing = False
        self._rope_cache_resolution: Optional[Tuple[int, int, int]] = None
        self.register_buffer("_rope_cache_cos", torch.tensor([], dtype=torch.float32), persistent=False)
        self.register_buffer("_rope_cache_sin", torch.tensor([], dtype=torch.float32), persistent=False)
        self._rope_cache_initialized = False

    def build_rope_cache(
        self,
        num_frames: int,
        height: int,
        width: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if num_frames <= 0 or height <= 0 or width <= 0:
            raise ValueError("num_frames, height and width must be positive to build RoPE cache.")
        if device is None:
            device = next(self.parameters()).device
        if isinstance(device, str):
            device = torch.device(device)
        if dtype is None:
            dtype = next(self.parameters()).dtype
        hidden_states = torch.zeros(
            1,
            self.config.in_channels,
            num_frames,
            height,
            width,
            device=device,
            dtype=dtype,
        )
        rope_cos, rope_sin = self.rope(hidden_states)
        rope_cos = rope_cos.to(device=device, dtype=dtype).detach()
        rope_sin = rope_sin.to(device=device, dtype=dtype).detach()
        self._rope_cache_resolution = (num_frames, height, width)
        self._rope_cache_cos = rope_cos
        self._rope_cache_sin = rope_sin
        self._rope_cache_initialized = True

    def _get_rotary_embeddings(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frames, height, width = hidden_states.shape[2:5]
        cache_key = (frames, height, width)

        if (
            self._rope_cache_initialized
            and self._rope_cache_resolution == cache_key
            and self._rope_cache_cos.device == hidden_states.device
            and self._rope_cache_cos.dtype == hidden_states.dtype
        ):
            return self._rope_cache_cos, self._rope_cache_sin

        rope_cos, rope_sin = self.rope(hidden_states)
        if rope_cos.dtype != hidden_states.dtype:
            rope_cos = rope_cos.to(dtype=hidden_states.dtype)
            rope_sin = rope_sin.to(dtype=hidden_states.dtype)
        return rope_cos, rope_sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rope_cos, rope_sin = self._get_rotary_embeddings(hidden_states)

        def _squeeze_rope(x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 4 and x.shape[2] == 1:
                return x.squeeze(2)
            return x

        rotary_emb = (_squeeze_rope(rope_cos), _squeeze_rope(rope_sin))

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj = self.condition_embedder(timestep, timestep_seq_len=ts_seq_len)
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    timestep_proj,
                    rotary_emb,
                )

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
