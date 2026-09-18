# Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/huggingface/transformers/blob/main/LICENSE.
#
# This modified file is released under the same license.


from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.functional import scaled_dot_product_attention
from transformers.utils import ModelOutput

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:  # pragma: no cover - optional runtime dependency
    flash_attn_varlen_func = None
from ..qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)

from ..qwen2.configuration_qwen2 import Qwen2Config as _Qwen2Config
from ..cache_utils.taylorseer import (
    cal_type,
    taylor_cache_init,
    derivative_approximation,
    taylor_formula,
)
from ...accelerator import enable_dynamo_flex_attention as _enable_dynamo_flex_attention


torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
if _enable_dynamo_flex_attention():
    # flex_attention = torch.compile(flex_attention) # , dynamic=True, mode='max-autotune'
    flex_attention = torch.compile(flex_attention)


def bounded_residual_merge(
    base_hidden: torch.Tensor,
    reviewed_hidden: torch.Tensor,
    *,
    residual_scale: float,
    alpha: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Merge a loop update without moving BAGEL far from its native state.

    The cap is computed independently for every token over the hidden width.
    ``alpha=0`` is deliberately an exact identity operation, which lets the
    sampler bypass the loop at low-noise timesteps without changing BAGEL's
    original depth-1 trajectory.
    """

    if float(residual_scale) < 0:
        raise ValueError("residual_scale must be non-negative")
    if float(alpha) == 0.0 or float(residual_scale) == 0.0:
        return base_hidden
    if base_hidden.shape != reviewed_hidden.shape:
        raise ValueError(
            "base_hidden and reviewed_hidden must have identical shapes, got "
            f"{tuple(base_hidden.shape)} and {tuple(reviewed_hidden.shape)}"
        )

    delta = reviewed_hidden - base_hidden
    base_rms = base_hidden.float().square().mean(dim=-1, keepdim=True).sqrt()
    delta_rms = delta.float().square().mean(dim=-1, keepdim=True).sqrt()
    cap = (float(residual_scale) * base_rms / (delta_rms + float(eps))).clamp(max=1.0)
    weight = (cap * float(alpha)).to(dtype=delta.dtype)
    return base_hidden + delta * weight


def _resolve_loop_state_positions(
    *,
    packed_query_sequence: torch.Tensor,
    packed_loop_semantic_token_indexes: Optional[torch.Tensor],
) -> torch.Tensor:
    """Resolve BAGEL's native pre-image semantic slots as recurrent state."""

    if (
        packed_loop_semantic_token_indexes is None
        or int(packed_loop_semantic_token_indexes.numel()) == 0
    ):
        raise ValueError(
            "cross-step loop state requires non-empty semantic token indexes; "
            "set loop_state_tokens >= 1 when preparing the VAE latent"
        )
    indexes = packed_loop_semantic_token_indexes
    indexes = indexes.to(device=packed_query_sequence.device, dtype=torch.long)
    if int(indexes.unique().numel()) != int(indexes.numel()):
        raise ValueError("loop semantic token indexes must be unique")
    if int(indexes.min()) < 0 or int(indexes.max()) >= int(
        packed_query_sequence.shape[0]
    ):
        raise ValueError("loop state positions are outside the query sequence")
    return indexes


def _sdpa_varlen_inference(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lens: torch.Tensor,
    key_value_lens: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    """Exact PyTorch fallback for FlashAttention's packed varlen contract.

    Cached decoding needs a bottom-right causal mask: a one-token query must
    see every preceding cached key.  ``scaled_dot_product_attention`` uses a
    top-left triangle when ``is_causal=True`` and query/key lengths differ, so
    construct the absolute-position mask explicitly instead.
    """

    query_lengths = [int(length) for length in query_lens.tolist()]
    key_lengths = [int(length) for length in key_value_lens.tolist()]
    if len(query_lengths) != len(key_lengths):
        raise ValueError("query_lens and key_value_lens must have equal batch size")
    if sum(query_lengths) != int(query.shape[0]):
        raise ValueError("query_lens do not cover the packed query tensor")
    if sum(key_lengths) != int(key.shape[0]) or tuple(key.shape) != tuple(value.shape):
        raise ValueError("key_value_lens do not cover aligned packed key/value tensors")
    if int(query.shape[-1]) != int(key.shape[-1]):
        raise ValueError("query and key head dimensions must match")

    outputs = []
    query_offset = 0
    key_offset = 0
    query_heads = int(query.shape[1])
    key_heads = int(key.shape[1])
    if query_heads % key_heads:
        raise ValueError(
            f"query heads ({query_heads}) must be divisible by KV heads ({key_heads})"
        )
    groups = query_heads // key_heads
    for query_length, key_length in zip(query_lengths, key_lengths):
        if query_length <= 0 or key_length < query_length:
            raise ValueError(
                "each packed sample requires 0 < query_length <= key_value_length"
            )
        sample_query = query[query_offset : query_offset + query_length]
        sample_key = key[key_offset : key_offset + key_length]
        sample_value = value[key_offset : key_offset + key_length]
        if groups > 1:
            sample_key = sample_key.repeat_interleave(groups, dim=1)
            sample_value = sample_value.repeat_interleave(groups, dim=1)

        attention_mask = None
        if bool(causal):
            cached_length = key_length - query_length
            query_positions = (
                torch.arange(query_length, device=query.device, dtype=torch.long)
                + cached_length
            )
            key_positions = torch.arange(
                key_length, device=query.device, dtype=torch.long
            )
            attention_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        sample_output = scaled_dot_product_attention(
            sample_query.transpose(0, 1).unsqueeze(0),
            sample_key.transpose(0, 1).unsqueeze(0),
            sample_value.transpose(0, 1).unsqueeze(0),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        outputs.append(sample_output.squeeze(0).transpose(0, 1))
        query_offset += query_length
        key_offset += key_length
    return torch.cat(outputs, dim=0)


class Qwen2Config(_Qwen2Config):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2Model`]. It is used to instantiate a
    Qwen2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-7B-beta [Qwen/Qwen2-7B-beta](https://huggingface.co/Qwen/Qwen2-7B-beta).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Qwen2Model, Qwen2Config

    >>> # Initializing a Qwen2 style configuration
    >>> configuration = Qwen2Config()

    >>> # Initializing a model from the Qwen2-7B style configuration
    >>> model = Qwen2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        is_causal=True,
        _attn_implementation="flash_attention_2",
        qk_norm=True,
        layer_module="Qwen2DecoderLayer",
        freeze_und=False,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            is_causal=is_causal,
            _attn_implementation=_attn_implementation,
            **kwargs,
        )
        self.qk_norm = qk_norm
        self.layer_module = layer_module
        self.freeze_und = freeze_und


class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0


@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    packed_query_sequence: torch.FloatTensor = None
    past_key_values: Optional[NaiveCache] = None
    loop_state_out: Optional[torch.FloatTensor] = None


def pad_sequence(tensor, pad_size):
    H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=1)


class PackedAttention(Qwen2Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        if self.config.qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask: List[torch.Tensor],
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ):
        packed_query_states = self.q_proj(packed_sequence).view(
            -1, self.num_heads, self.head_dim
        )
        packed_key_states = self.k_proj(packed_sequence).view(
            -1, self.num_key_value_heads, self.head_dim
        )
        packed_value_states = self.v_proj(packed_sequence).view(
            -1, self.num_key_value_heads, self.head_dim
        )

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states,
            packed_key_states,
            packed_cos,
            packed_sin,
            unsqueeze_dim=1,
        )

        if isinstance(attention_mask, List):
            packed_key_states = packed_key_states[:, :, None, :].repeat(
                1, 1, self.num_key_value_groups, 1
            )
            packed_key_states = packed_key_states.reshape(
                -1, self.num_heads, self.head_dim
            )
            packed_value_states = packed_value_states[:, :, None, :].repeat(
                1, 1, self.num_key_value_groups, 1
            )
            packed_value_states = packed_value_states.reshape(
                -1, self.num_heads, self.head_dim
            )

            unpacked_query_states = packed_query_states.transpose(0, 1).split(
                sample_lens, dim=1
            )
            unpacked_key_states = packed_key_states.transpose(0, 1).split(
                sample_lens, dim=1
            )
            unpacked_value_states = packed_value_states.transpose(0, 1).split(
                sample_lens, dim=1
            )
            upacked_attn_output = []
            for (
                query_states,
                key_states,
                value_states,
                attention_mask_per_sample,
            ) in zip(
                unpacked_query_states,
                unpacked_key_states,
                unpacked_value_states,
                attention_mask,
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0),
                        key_states.to(torch.bfloat16).unsqueeze(0),
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states = pad_sequence(
                packed_query_states.permute(1, 0, 2), pad_size
            )
            packed_key_states = pad_sequence(
                packed_key_states.permute(1, 0, 2), pad_size
            )
            packed_value_states = pad_sequence(
                packed_value_states.permute(1, 0, 2), pad_size
            )
            packed_attn_output = flex_attention(
                packed_query_states.unsqueeze(0),
                packed_key_states.unsqueeze(0),
                packed_value_states.unsqueeze(0),
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(
            -1, self.hidden_size
        )
        packed_attn_output = self.o_proj(packed_attn_output)

        return packed_attn_output

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
    ):
        packed_query_states = self.q_proj(packed_query_sequence).view(
            -1, self.num_heads, self.head_dim
        )
        packed_key_states = self.k_proj(packed_query_sequence).view(
            -1, self.num_key_value_heads, self.head_dim
        )
        packed_value_states = self.v_proj(packed_query_sequence).view(
            -1, self.num_key_value_heads, self.head_dim
        )

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states,
            packed_key_states,
            packed_cos,
            packed_sin,
            unsqueeze_dim=1,
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if (
            past_key_values is not None
            and past_key_values.key_cache[self.layer_idx] is not None
        ):
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(
                (seqlens, self.num_key_value_heads, self.head_dim)
            )
            merged_value_states = past_key_states.new_zeros(
                (seqlens, self.num_key_value_heads, self.head_dim)
            )
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(key_values_lens, dim=0), (1, 0)
        )

        if flash_attn_varlen_func is None:
            packed_attn_output = _sdpa_varlen_inference(
                query=packed_query_states,
                key=merged_key_states,
                value=merged_value_states,
                query_lens=query_lens,
                key_value_lens=key_values_lens,
                causal=bool(is_causal),
            )
        else:
            packed_attn_output = flash_attn_varlen_func(
                q=packed_query_states,
                k=merged_key_states,
                v=merged_value_states,
                cu_seqlens_q=cu_seqlens_q.to(torch.int32),
                cu_seqlens_k=cu_seqlens_k.to(torch.int32),
                max_seqlen_q=max(query_lens).item(),
                max_seqlen_k=max(key_values_lens).item(),
                causal=is_causal,
            )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output)

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


class PackedAttentionMoT(Qwen2Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        if self.config.qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.q_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()

        self.q_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj_moe_gen = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ):
        packed_query_states = packed_sequence.new_zeros(
            (packed_sequence.shape[0], self.num_heads * self.head_dim)
        )
        packed_key_states = packed_sequence.new_zeros(
            (packed_sequence.shape[0], self.num_key_value_heads * self.head_dim)
        )
        packed_value_states = packed_sequence.new_zeros(
            (packed_sequence.shape[0], self.num_key_value_heads * self.head_dim)
        )

        packed_sequence_und = packed_sequence[packed_und_token_indexes]
        packed_sequence_gen = packed_sequence[packed_gen_token_indexes]

        packed_query_states[packed_und_token_indexes] = self.q_proj(packed_sequence_und)
        packed_query_states[packed_gen_token_indexes] = self.q_proj_moe_gen(
            packed_sequence_gen
        )

        packed_key_states[packed_und_token_indexes] = self.k_proj(packed_sequence_und)
        packed_key_states[packed_gen_token_indexes] = self.k_proj_moe_gen(
            packed_sequence_gen
        )

        packed_value_states[packed_und_token_indexes] = self.v_proj(packed_sequence_und)
        packed_value_states[packed_gen_token_indexes] = self.v_proj_moe_gen(
            packed_sequence_gen
        )

        packed_query_states = packed_query_states.view(
            -1, self.num_heads, self.head_dim
        )
        packed_key_states = packed_key_states.view(
            -1, self.num_key_value_heads, self.head_dim
        )
        packed_value_states = packed_value_states.view(
            -1, self.num_key_value_heads, self.head_dim
        )
        if self.config.freeze_und:
            packed_value_states[packed_und_token_indexes] = packed_value_states[
                packed_und_token_indexes
            ].detach()

        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

        packed_query_states_[packed_und_token_indexes] = self.q_norm(
            packed_query_states[packed_und_token_indexes]
        )
        if self.config.freeze_und:
            packed_query_states_[packed_und_token_indexes] = packed_query_states_[
                packed_und_token_indexes
            ].detach()
        packed_query_states_[packed_gen_token_indexes] = self.q_norm_moe_gen(
            packed_query_states[packed_gen_token_indexes]
        )

        packed_key_states_[packed_und_token_indexes] = self.k_norm(
            packed_key_states[packed_und_token_indexes]
        )
        if self.config.freeze_und:
            packed_key_states_[packed_und_token_indexes] = packed_key_states_[
                packed_und_token_indexes
            ].detach()
        packed_key_states_[packed_gen_token_indexes] = self.k_norm_moe_gen(
            packed_key_states[packed_gen_token_indexes]
        )

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states_, packed_key_states_ = apply_rotary_pos_emb(
            packed_query_states_,
            packed_key_states_,
            packed_cos,
            packed_sin,
            unsqueeze_dim=1,
        )

        if isinstance(attention_mask, List):
            packed_key_states_ = packed_key_states_[:, :, None, :].repeat(
                1, 1, self.num_key_value_groups, 1
            )
            packed_key_states_ = packed_key_states_.reshape(
                -1, self.num_heads, self.head_dim
            )
            packed_value_states = packed_value_states[:, :, None, :].repeat(
                1, 1, self.num_key_value_groups, 1
            )
            packed_value_states = packed_value_states.reshape(
                -1, self.num_heads, self.head_dim
            )

            unpacked_query_states = packed_query_states_.transpose(0, 1).split(
                sample_lens, dim=1
            )
            unpacked_key_states = packed_key_states_.transpose(0, 1).split(
                sample_lens, dim=1
            )
            unpacked_value_states = packed_value_states.transpose(0, 1).split(
                sample_lens, dim=1
            )
            upacked_attn_output = []
            for (
                query_states,
                key_states,
                value_states,
                attention_mask_per_sample,
            ) in zip(
                unpacked_query_states,
                unpacked_key_states,
                unpacked_value_states,
                attention_mask,
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0),
                        key_states.to(torch.bfloat16).unsqueeze(0),
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states_ = pad_sequence(
                packed_query_states_.permute(1, 0, 2), pad_size
            )
            packed_key_states_ = pad_sequence(
                packed_key_states_.permute(1, 0, 2), pad_size
            )
            packed_value_states = pad_sequence(
                packed_value_states.permute(1, 0, 2), pad_size
            )
            packed_attn_output = flex_attention(
                packed_query_states_.unsqueeze(0),  # 1, num_head, L, head_dim
                packed_key_states_.unsqueeze(0),
                packed_value_states.unsqueeze(0),
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(
            -1, self.num_heads * self.head_dim
        )
        packed_attn_output_ = packed_attn_output.new_zeros(packed_attn_output.shape)
        packed_attn_output_[packed_und_token_indexes] = self.o_proj(
            packed_attn_output[packed_und_token_indexes]
        )
        packed_attn_output_[packed_gen_token_indexes] = self.o_proj_moe_gen(
            packed_attn_output[packed_gen_token_indexes]
        )

        return packed_attn_output_

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        packed_loop_semantic_token_indexes=None,
        loop_kv_prefix_hidden=None,
    ):
        if mode == "und":
            packed_query_states = self.q_proj(packed_query_sequence).view(
                -1, self.num_heads, self.head_dim
            )
            packed_key_states = self.k_proj(packed_query_sequence).view(
                -1, self.num_key_value_heads, self.head_dim
            )
            packed_value_states = self.v_proj(packed_query_sequence).view(
                -1, self.num_key_value_heads, self.head_dim
            )
            packed_query_states = self.q_norm(packed_query_states)
            packed_key_states = self.k_norm(packed_key_states)
        elif mode == "gen":
            packed_query_sequence = packed_query_sequence.to(torch.bfloat16)
            packed_query_states = packed_query_sequence.new_zeros(
                (packed_query_sequence.shape[0], self.num_heads * self.head_dim)
            )
            packed_key_states = packed_query_sequence.new_zeros(
                (
                    packed_query_sequence.shape[0],
                    self.num_key_value_heads * self.head_dim,
                )
            )
            packed_value_states = packed_query_sequence.new_zeros(
                (
                    packed_query_sequence.shape[0],
                    self.num_key_value_heads * self.head_dim,
                )
            )

            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]

            packed_query_states[packed_text_indexes] = self.q_proj(
                packed_text_query_sequence
            )
            packed_query_states[packed_vae_token_indexes] = self.q_proj_moe_gen(
                packed_vae_query_sequence
            )

            packed_key_states[packed_text_indexes] = self.k_proj(
                packed_text_query_sequence
            )
            packed_key_states[packed_vae_token_indexes] = self.k_proj_moe_gen(
                packed_vae_query_sequence
            )

            packed_value_states[packed_text_indexes] = self.v_proj(
                packed_text_query_sequence
            )
            packed_value_states[packed_vae_token_indexes] = self.v_proj_moe_gen(
                packed_vae_query_sequence
            )

            if loop_kv_prefix_hidden is not None:
                state_indexes = packed_loop_semantic_token_indexes
                if state_indexes is None or int(state_indexes.numel()) == 0:
                    raise ValueError("KV-prefix state requires semantic token indexes")
                state_indexes = state_indexes.to(
                    device=packed_query_sequence.device, dtype=torch.long
                )
                prefix_hidden = loop_kv_prefix_hidden.to(
                    device=packed_query_sequence.device,
                    dtype=packed_query_sequence.dtype,
                )
                if prefix_hidden.ndim != 2 or int(prefix_hidden.shape[0]) != int(
                    state_indexes.numel()
                ):
                    raise ValueError(
                        "loop_kv_prefix_hidden must have one hidden row per "
                        "semantic state token"
                    )
                if int(prefix_hidden.shape[-1]) != int(self.hidden_size):
                    raise ValueError("KV-prefix hidden width does not match BAGEL")
                # KV-channel ablation: the recurrent UND representation replaces
                # only K/V at the native semantic-prefix positions. Queries and
                # residual state slots stay on the ordinary UND update stream.
                packed_key_states = packed_key_states.clone()
                packed_value_states = packed_value_states.clone()
                packed_key_states[state_indexes] = self.k_proj(prefix_hidden)
                packed_value_states[state_indexes] = self.v_proj(prefix_hidden)

            packed_query_states = packed_query_states.view(
                -1, self.num_heads, self.head_dim
            )
            packed_key_states = packed_key_states.view(
                -1, self.num_key_value_heads, self.head_dim
            )
            packed_value_states = packed_value_states.view(
                -1, self.num_key_value_heads, self.head_dim
            )

            # Keep the inference numerics while avoiding read-then-overwrite
            # mutations on tensors that carry loop-LoRA autograd history.
            raw_query_states = packed_query_states.to(torch.float32)
            normalized_query_states = torch.zeros_like(raw_query_states)
            normalized_query_states[packed_text_indexes] = self.q_norm(
                raw_query_states[packed_text_indexes]
            )
            normalized_query_states[packed_vae_token_indexes] = self.q_norm_moe_gen(
                raw_query_states[packed_vae_token_indexes]
            )
            packed_query_states = normalized_query_states

            raw_key_states = packed_key_states.to(torch.float32)
            normalized_key_states = torch.zeros_like(raw_key_states)
            normalized_key_states[packed_text_indexes] = self.k_norm(
                raw_key_states[packed_text_indexes]
            )
            normalized_key_states[packed_vae_token_indexes] = self.k_norm_moe_gen(
                raw_key_states[packed_vae_token_indexes]
            )
            packed_key_states = normalized_key_states

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states,
            packed_key_states,
            packed_cos,
            packed_sin,
            unsqueeze_dim=1,
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if (
            past_key_values is not None
            and past_key_values.key_cache[self.layer_idx] is not None
        ):
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(
                size=[seqlens, self.num_key_value_heads, self.head_dim]
            )
            merged_value_states = past_key_states.new_zeros(
                size=[seqlens, self.num_key_value_heads, self.head_dim]
            )
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(key_values_lens, dim=0), (1, 0)
        )

        if flash_attn_varlen_func is None:
            packed_attn_output = _sdpa_varlen_inference(
                query=packed_query_states,
                key=merged_key_states,
                value=merged_value_states,
                query_lens=query_lens,
                key_value_lens=key_values_lens,
                causal=bool(is_causal),
            )
        else:
            packed_attn_output = flash_attn_varlen_func(
                q=packed_query_states,
                k=merged_key_states,
                v=merged_value_states,
                cu_seqlens_q=cu_seqlens_q.to(torch.int32),
                cu_seqlens_k=cu_seqlens_k.to(torch.int32),
                max_seqlen_q=max(query_lens).item(),
                max_seqlen_k=max(key_values_lens).item(),
                causal=is_causal,
            )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        if mode == "und":
            packed_attn_output = self.o_proj(packed_attn_output)
        elif mode == "gen":
            raw_attn_output = packed_attn_output
            routed_attn_output = torch.zeros_like(raw_attn_output)
            routed_attn_output[packed_text_indexes] = self.o_proj(
                raw_attn_output[packed_text_indexes]
            )
            routed_attn_output[packed_vae_token_indexes] = self.o_proj_moe_gen(
                raw_attn_output[packed_vae_token_indexes]
            )
            packed_attn_output = routed_attn_output

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = PackedAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence = self.input_layernorm(packed_sequence)

        # Self Attention
        packed_sequence = self.self_attn.forward_train(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
        )
        packed_sequence = residual + packed_sequence

        # Fully Connected
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)
        packed_sequence = self.mlp(packed_sequence)
        packed_sequence = residual + packed_sequence

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
    ) -> BaseNavitOutputWithPast:

        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn.forward_inference(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        packed_query_sequence = self.mlp(packed_query_sequence)
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


class Qwen2MoTDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: Optional[int] = None,
        attn_module: Optional[Qwen2Attention] = PackedAttentionMoT,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.freeze_und = config.freeze_und

        self.self_attn = attn_module(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_moe_gen = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_und_token_indexes] = self.input_layernorm(
            packed_sequence[packed_und_token_indexes]
        )
        packed_sequence_[packed_gen_token_indexes] = self.input_layernorm_moe_gen(
            packed_sequence[packed_gen_token_indexes]
        )

        # Self Attention
        packed_sequence_ = self.self_attn.forward_train(
            packed_sequence=packed_sequence_,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[
                packed_und_token_indexes
            ].detach()
        packed_sequence = residual + packed_sequence_

        # Fully Connected
        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_und_token_indexes] = self.mlp(
            self.post_attention_layernorm(packed_sequence[packed_und_token_indexes])
        )
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[
                packed_und_token_indexes
            ].detach()

        packed_sequence_[packed_gen_token_indexes] = self.mlp_moe_gen(
            self.post_attention_layernorm_moe_gen(
                packed_sequence[packed_gen_token_indexes]
            )
        )
        packed_sequence = residual + packed_sequence_

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        packed_loop_semantic_token_indexes=None,
        loop_kv_prefix_hidden=None,
    ) -> BaseNavitOutputWithPast:

        enable_taylorseer = getattr(self, "enable_taylorseer", False)

        if enable_taylorseer and self.current["type"] == "full":
            self.current["module"] = "total"
            taylor_cache_init(cache_dic=self.cache_dic, current=self.current)

        if not enable_taylorseer or (
            enable_taylorseer and self.current["type"] == "full"
        ):
            residual = packed_query_sequence
            if mode == "und":
                packed_query_sequence = self.input_layernorm(packed_query_sequence)
            elif mode == "gen":
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                packed_query_sequence_[packed_text_indexes] = self.input_layernorm(
                    packed_query_sequence[packed_text_indexes]
                )
                packed_query_sequence_[packed_vae_token_indexes] = (
                    self.input_layernorm_moe_gen(
                        packed_query_sequence[packed_vae_token_indexes]
                    )
                )
                packed_query_sequence = packed_query_sequence_

            normalized_loop_kv_prefix = None
            if loop_kv_prefix_hidden is not None:
                if mode != "gen":
                    raise ValueError("loop KV prefix is only valid in gen mode")
                normalized_loop_kv_prefix = self.input_layernorm(loop_kv_prefix_hidden)

            # Self Attention
            packed_query_sequence, past_key_values = self.self_attn.forward_inference(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                mode=mode,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_indexes=packed_text_indexes,
                packed_loop_semantic_token_indexes=(packed_loop_semantic_token_indexes),
                loop_kv_prefix_hidden=normalized_loop_kv_prefix,
            )
            packed_query_sequence = residual + packed_query_sequence

            # Fully Connected
            residual = packed_query_sequence
            if mode == "und":
                packed_query_sequence = self.post_attention_layernorm(
                    packed_query_sequence
                )
                packed_query_sequence = self.mlp(packed_query_sequence)
            elif mode == "gen":
                packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
                packed_vae_query_sequence = packed_query_sequence[
                    packed_vae_token_indexes
                ]
                packed_text_query_sequence = self.post_attention_layernorm(
                    packed_text_query_sequence
                ).to(torch.bfloat16)
                packed_vae_query_sequence = self.post_attention_layernorm_moe_gen(
                    packed_vae_query_sequence
                ).to(torch.bfloat16)

                packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(
                    torch.bfloat16
                )
                packed_query_sequence_[packed_text_indexes] = self.mlp(
                    packed_text_query_sequence
                )
                packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(
                    packed_vae_query_sequence
                )
                packed_query_sequence = packed_query_sequence_

            packed_query_sequence = residual + packed_query_sequence

        if enable_taylorseer:
            if self.current["type"] == "full":
                derivative_approximation(
                    cache_dic=self.cache_dic,
                    current=self.current,
                    feature=packed_query_sequence,
                )
            elif self.current["type"] == "Taylor":
                self.current["module"] = "total"
                packed_query_sequence = taylor_formula(
                    cache_dic=self.cache_dic, current=self.current
                )

        return packed_query_sequence, past_key_values


class Qwen2MoEDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = PackedAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence = self.input_layernorm(packed_sequence)

        # Self Attention
        packed_sequence = self.self_attn.forward_train(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
        )
        packed_sequence = residual + packed_sequence

        # Fully Connected
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)

        packed_sequence_new = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_und = self.mlp(packed_sequence[packed_und_token_indexes])
        packed_sequence_gen = self.mlp_moe_gen(
            packed_sequence[packed_gen_token_indexes]
        )
        packed_sequence_new[packed_und_token_indexes] = packed_sequence_und
        packed_sequence_new[packed_gen_token_indexes] = packed_sequence_gen

        packed_sequence = residual + packed_sequence_new

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:

        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn.forward_inference(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        if mode == "und":
            packed_query_sequence = self.mlp(packed_query_sequence)
        elif mode == "gen":
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(
                torch.bfloat16
            )
            packed_query_sequence_[packed_text_indexes] = self.mlp(
                packed_query_sequence[packed_text_indexes]
            )
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(
                packed_query_sequence[packed_vae_token_indexes]
            )
            packed_query_sequence = packed_query_sequence_
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


Decoder_layer_dict = {
    "Qwen2DecoderLayer": Qwen2DecoderLayer,
    "Qwen2MoEDecoderLayer": Qwen2MoEDecoderLayer,
    "Qwen2MoTDecoderLayer": partial(
        Qwen2MoTDecoderLayer, attn_module=PackedAttentionMoT
    ),
}


class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_moe = "Mo" in config.layer_module

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        layer_module = Decoder_layer_dict[config.layer_module]
        self.layers = nn.ModuleList(
            [
                layer_module(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_moe:
            self.norm_moe_gen = Qwen2RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # 2026-04-29: BAGEL's NaViT Qwen2 stack does NOT inherit HF's
        # standard `_gradient_checkpointing_func` machinery (that path
        # lives in `qwen_latent_cot/bagel/modeling/qwen2/modeling_qwen2.py`,
        # which BAGEL doesn't use). We add a tiny manual flag that
        # `forward_train` checks; see `gradient_checkpointing_enable`.
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def gradient_checkpointing_enable(self, **_kwargs) -> None:
        """Enable activation checkpointing on every decoder layer.

        Uses `use_reentrant=False` (the new default contract) which is
        the only mode compatible with FSDP `use_orig_params=True`.
        """
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ):
        if self.config.freeze_und:
            packed_sequence[packed_und_token_indexes] = packed_sequence[
                packed_und_token_indexes
            ].detach()

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_moe:
            assert packed_und_token_indexes is not None
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )

        use_grad_ckpt = bool(self.gradient_checkpointing) and self.training
        for layer_index, decoder_layer in enumerate(self.layers):
            if use_grad_ckpt:
                packed_sequence = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    packed_sequence=packed_sequence,
                    sample_lens=sample_lens,
                    attention_mask=attention_mask,
                    packed_position_embeddings=packed_position_embeddings,
                    use_reentrant=False,
                    **extra_inputs,
                )
            else:
                actual_layer = getattr(
                    decoder_layer, "_checkpoint_wrapped_module", decoder_layer
                )
                packed_sequence = actual_layer.forward_train(
                    packed_sequence=packed_sequence,
                    sample_lens=sample_lens,
                    attention_mask=attention_mask,
                    packed_position_embeddings=packed_position_embeddings,
                    **extra_inputs,
                )

        if self.use_moe:
            packed_sequence_ = torch.zeros_like(packed_sequence)
            packed_sequence_[packed_und_token_indexes] = self.norm(
                packed_sequence[packed_und_token_indexes]
            )
            if self.config.freeze_und:
                packed_sequence_[packed_und_token_indexes] = packed_sequence_[
                    packed_und_token_indexes
                ].detach()
            packed_sequence_[packed_gen_token_indexes] = self.norm_moe_gen(
                packed_sequence[packed_gen_token_indexes]
            )
            output = packed_sequence_
        else:
            output = self.norm(packed_sequence)
        return output

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        loop_start_layer: Optional[int] = None,
        loop_end_layer: Optional[int] = None,
        loop_state_in: Optional[torch.Tensor] = None,
        loop_state_scale: Optional[float] = None,
        loop_external_state: Optional[torch.Tensor] = None,
        loop_external_state_scale: Optional[float] = None,
        loop_residual_alpha: float = 1.0,
        packed_boundary_token_indexes: Optional[torch.Tensor] = None,
        packed_loop_semantic_token_indexes: Optional[torch.Tensor] = None,
        loop_state_mode: str = "semantic_token",
        return_loop_state: bool = False,
        within_step_loop_start: Optional[int] = None,
        within_step_loop_end: Optional[int] = None,
        within_step_loop_repeat: int = 1,
        within_step_loop_damping: float = 1.0,
    ) -> BaseNavitOutputWithPast:

        enable_taylorseer = getattr(self, "enable_taylorseer", False)
        if enable_taylorseer:
            cal_type(self.cache_dic, self.current)
            self.current["stream"] = "layers_stream"

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(
            packed_query_sequence, packed_query_position_ids.unsqueeze(0)
        )
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_query_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs.update(mode=mode)
            if mode == "gen":
                assert packed_vae_token_indexes is not None
                assert packed_text_indexes is not None
                extra_inputs.update(
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_text_indexes=packed_text_indexes,
                )

        def run_layer(
            layer_idx,
            hidden,
            *,
            checkpoint=False,
            loop_adapter_enabled=False,
            loop_kv_prefix_hidden=None,
            packed_loop_semantic_token_indexes=None,
        ):
            decoder_layer = self.layers[layer_idx]
            if enable_taylorseer:
                decoder_layer.current = self.current
                decoder_layer.cache_dic = self.cache_dic
                decoder_layer.enable_taylorseer = True
                self.current["layer"] = layer_idx
            actual_layer = getattr(
                decoder_layer, "_checkpoint_wrapped_module", decoder_layer
            )
            layer_kwargs = dict(
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                **extra_inputs,
            )
            if loop_kv_prefix_hidden is not None:
                layer_kwargs.update(
                    loop_kv_prefix_hidden=loop_kv_prefix_hidden,
                    packed_loop_semantic_token_indexes=(
                        packed_loop_semantic_token_indexes
                    ),
                )
            use_checkpoint = (
                bool(checkpoint)
                and bool(getattr(self, "gradient_checkpointing", False))
                and self.training
                and torch.is_grad_enabled()
            )
            if use_checkpoint:

                def checkpointed_layer(layer_hidden):
                    local_states = []
                    if loop_adapter_enabled:
                        for module in actual_layer.modules():
                            setter = getattr(module, "set_loop_enabled", None)
                            if callable(setter):
                                local_states.append(
                                    (
                                        module,
                                        bool(getattr(module, "loop_enabled", False)),
                                    )
                                )
                                setter(True)
                    try:
                        layer_output, _ = actual_layer.forward_inference(
                            packed_query_sequence=layer_hidden,
                            **layer_kwargs,
                        )
                        return layer_output
                    finally:
                        for module, was_enabled in local_states:
                            module.set_loop_enabled(was_enabled)

                hidden = torch.utils.checkpoint.checkpoint(
                    checkpointed_layer,
                    hidden,
                    use_reentrant=False,
                )
                cache = past_key_values
            else:
                hidden, cache = actual_layer.forward_inference(
                    packed_query_sequence=hidden,
                    **layer_kwargs,
                )
            return hidden, cache

        def normalize(hidden):
            if self.use_moe:
                if mode == "und":
                    return self.norm(hidden)
                if mode == "gen":
                    normalized = torch.zeros_like(hidden)
                    normalized[packed_text_indexes] = self.norm(
                        hidden[packed_text_indexes]
                    )
                    normalized[packed_vae_token_indexes] = self.norm_moe_gen(
                        hidden[packed_vae_token_indexes]
                    )
                    return normalized
                raise ValueError(f"unsupported MoT mode: {mode}")
            return self.norm(hidden)

        # Bootstrap semantics: the first state-active step has no incoming
        # state yet. A requested output state with a valid scale still takes
        # the state path (body runs natively, adapters enabled, state
        # extracted) so the recurrence can start mid-trajectory.
        state_active = (
            loop_state_in is not None
            or loop_external_state is not None
            or loop_state_scale is not None
        )
        loop_state_out = None

        loop_repeat = int(within_step_loop_repeat)
        if loop_repeat > 1 and not state_active:
            # Looped-MMDiT style: repeat a shared middle block inside one
            # denoising step. L=1 is exact parity with the native path.
            if within_step_loop_start is None or within_step_loop_end is None:
                raise ValueError(
                    "within_step_loop_repeat > 1 requires "
                    "within_step_loop_start and within_step_loop_end"
                )
            s = int(within_step_loop_start)
            e = int(within_step_loop_end)
            alpha = float(within_step_loop_damping)
            if not (0 <= s < e <= len(self.layers)):
                raise ValueError(
                    f"within_step_loop range [{s}, {e}) is invalid for "
                    f"{len(self.layers)} layers"
                )
            hidden = packed_query_sequence
            for layer_idx in range(0, s):
                hidden, past_key_values = run_layer(layer_idx, hidden)
            for _loop_index in range(loop_repeat):
                prev_hidden = hidden
                for layer_idx in range(s, e):
                    hidden, past_key_values = run_layer(layer_idx, hidden)
                if alpha != 1.0:
                    hidden = prev_hidden + alpha * (hidden - prev_hidden)
            for layer_idx in range(e, len(self.layers)):
                hidden, past_key_values = run_layer(layer_idx, hidden)
            packed_query_sequence = normalize(hidden)
        elif not state_active:
            # Keep the original sequential path untouched. This is the hard
            # parity baseline for every loop experiment.
            for layer_idx in range(len(self.layers)):
                packed_query_sequence, past_key_values = run_layer(
                    layer_idx, packed_query_sequence
                )
            packed_query_sequence = normalize(packed_query_sequence)
        else:
            if mode != "gen":
                raise ValueError("cross-step loop state is only valid in gen mode")
            if update_past_key_values:
                raise ValueError(
                    "cross-step loop state cannot mutate the prompt KV cache; "
                    "set update_past_key_values=False"
                )
            if enable_taylorseer:
                raise ValueError(
                    "TaylorSeer and cross-step loop state cannot run together"
                )
            state_mode = str(loop_state_mode).strip().lower()
            if state_mode not in {"semantic_token", "kv_prefix"}:
                raise ValueError(
                    "loop_state_mode must be 'semantic_token' or 'kv_prefix', "
                    f"got {loop_state_mode!r}"
                )
            start = int(loop_start_layer) if loop_start_layer is not None else -1
            end = int(loop_end_layer) if loop_end_layer is not None else -1
            if not 0 <= start < end <= len(self.layers):
                raise ValueError(
                    "loop layer range must satisfy "
                    f"0 <= start < end <= {len(self.layers)}, got [{start}, {end})"
                )

            state_positions = _resolve_loop_state_positions(
                packed_query_sequence=packed_query_sequence,
                packed_loop_semantic_token_indexes=(packed_loop_semantic_token_indexes),
            )

            if loop_external_state is not None:
                if loop_external_state_scale is None:
                    raise ValueError(
                        "loop_external_state_scale is required when an external state is provided"
                    )
                if float(loop_external_state_scale) < 0.0:
                    raise ValueError("loop_external_state_scale must be non-negative")
            if loop_state_in is not None and loop_state_scale is None:
                raise ValueError(
                    "loop_state_scale is required when a recurrent state is provided"
                )
            if loop_state_scale is not None and float(loop_state_scale) < 0.0:
                raise ValueError("loop_state_scale must be non-negative")

            adapter_states = []
            for module in self.modules():
                setter = getattr(module, "set_loop_enabled", None)
                if callable(setter):
                    adapter_states.append(
                        (module, bool(getattr(module, "loop_enabled", False)))
                    )
                    setter(False)
            try:
                hidden = packed_query_sequence
                for layer_idx in range(0, start):
                    hidden, past_key_values = run_layer(layer_idx, hidden)
                body_entry_hidden = hidden.clone()

                base_state = body_entry_hidden[state_positions]
                recurrent_state = None
                if loop_external_state is not None:
                    # Multi-round edit event: the reflection write replaces the
                    # self-recurrent state for this step.
                    reviewed_state = loop_external_state.to(
                        device=base_state.device,
                        dtype=base_state.dtype,
                    )
                    if reviewed_state.ndim == 1:
                        reviewed_state = reviewed_state.unsqueeze(0)
                    if reviewed_state.ndim != 2:
                        raise ValueError(
                            "loop_external_state must have shape [hidden], "
                            "[1, hidden], or [num_state_positions, hidden]"
                        )
                    if int(reviewed_state.shape[-1]) != int(base_state.shape[-1]):
                        raise ValueError(
                            "loop_external_state hidden width does not match BAGEL"
                        )
                    if int(reviewed_state.shape[0]) == 1:
                        reviewed_state = reviewed_state.expand_as(base_state)
                    elif tuple(reviewed_state.shape) != tuple(base_state.shape):
                        raise ValueError(
                            "loop_external_state token count must be 1 or match "
                            "the loop state position count"
                        )
                    recurrent_state = bounded_residual_merge(
                        base_state,
                        reviewed_state,
                        residual_scale=float(loop_external_state_scale),
                        alpha=float(loop_residual_alpha),
                    )
                elif loop_state_in is not None:
                    # Looped-Flows / RLT entry merge: stop-gradient on the
                    # incoming state, then an RMS-capped residual update so the
                    # body starts near its native entry distribution.
                    reviewed_state = loop_state_in.detach().to(
                        device=base_state.device,
                        dtype=base_state.dtype,
                    )
                    if reviewed_state.ndim != 2 or tuple(reviewed_state.shape) != tuple(
                        base_state.shape
                    ):
                        raise ValueError(
                            "loop_state_in must match the loop state positions: "
                            f"expected {tuple(base_state.shape)}, "
                            f"got {tuple(reviewed_state.shape)}"
                        )
                    recurrent_state = bounded_residual_merge(
                        base_state,
                        reviewed_state,
                        residual_scale=float(loop_state_scale),
                        alpha=float(loop_residual_alpha),
                    )
                # Direct-attention transport writes the recurrent state into
                # the K native semantic query slots. KV-prefix transport keeps
                # those queries native and substitutes only their per-layer
                # UND keys/values.
                hidden = body_entry_hidden
                kv_prefix_hidden = None
                if recurrent_state is not None:
                    if state_mode == "semantic_token":
                        hidden = body_entry_hidden.clone()
                        hidden[state_positions] = recurrent_state
                    else:
                        kv_prefix_hidden = recurrent_state

                for module, _ in adapter_states:
                    module.set_loop_enabled(True)
                for layer_idx in range(start, end):
                    hidden, past_key_values = run_layer(
                        layer_idx,
                        hidden,
                        checkpoint=True,
                        loop_adapter_enabled=True,
                        loop_kv_prefix_hidden=(
                            kv_prefix_hidden if state_mode == "kv_prefix" else None
                        ),
                        packed_loop_semantic_token_indexes=(
                            state_positions if state_mode == "kv_prefix" else None
                        ),
                    )
                    if state_mode == "kv_prefix":
                        kv_prefix_hidden = hidden[state_positions]
                for module, _ in adapter_states:
                    module.set_loop_enabled(False)

                # State update: the body-exit features at the state positions
                # become the next step's recurrent state (Looped Flows z).
                if return_loop_state:
                    loop_state_out = hidden[state_positions].clone()

                for layer_idx in range(end, len(self.layers)):
                    hidden, past_key_values = run_layer(layer_idx, hidden)
                packed_query_sequence = normalize(hidden)
            finally:
                for module, was_enabled in adapter_states:
                    module.set_loop_enabled(was_enabled)

        if enable_taylorseer:
            self.current["step"] += 1

        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
            loop_state_out=loop_state_out,
        )

    def forward_kvcache(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        use_gradient_checkpointing: bool = False,
    ) -> BaseNavitOutputWithPast:
        """KV-cache forward with optional per-layer checkpointing.

        Semantic review prefills a long image/text prefix into a mutable
        ``NaiveCache``. Calling ``forward_inference`` directly keeps every
        decoder layer's prefix activations alive and OOMs on 5k-token image
        contexts. This mirrors CoRT's BAGEL path: checkpoint only the long
        prefill segment, and reset the current layer's cache entry during
        recomputation so the mutable cache sees the same empty state.
        """
        enable_taylorseer = getattr(self, "enable_taylorseer", False)
        if enable_taylorseer:
            cal_type(self.cache_dic, self.current)
            self.current["stream"] = "layers_stream"

        cos, sin = self.rotary_emb(
            packed_query_sequence, packed_query_position_ids.unsqueeze(0)
        )
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_query_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs.update(mode=mode)
            if mode == "gen":
                assert packed_vae_token_indexes is not None
                assert packed_text_indexes is not None
                extra_inputs.update(
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_text_indexes=packed_text_indexes,
                )

        for layer_idx, decoder_layer in enumerate(self.layers):
            if enable_taylorseer:
                decoder_layer.current = self.current
                decoder_layer.cache_dic = self.cache_dic
                decoder_layer.enable_taylorseer = True
                self.current["layer"] = layer_idx

            actual_layer = getattr(
                decoder_layer, "_checkpoint_wrapped_module", decoder_layer
            )
            if use_gradient_checkpointing:
                cache_layer_idx = actual_layer.self_attn.layer_idx

                def _ckpt_layer_fn(
                    hidden,
                    _layer=actual_layer,
                    _idx=cache_layer_idx,
                ):
                    past_key_values.key_cache[_idx] = None
                    past_key_values.value_cache[_idx] = None
                    h, _ = _layer.forward_inference(
                        packed_query_sequence=hidden,
                        query_lens=query_lens,
                        packed_query_position_embeddings=packed_query_position_embeddings,
                        packed_query_indexes=packed_query_indexes,
                        past_key_values=past_key_values,
                        key_values_lens=key_values_lens,
                        packed_key_value_indexes=packed_key_value_indexes,
                        update_past_key_values=update_past_key_values,
                        is_causal=is_causal,
                        **extra_inputs,
                    )
                    return h

                packed_query_sequence = torch.utils.checkpoint.checkpoint(
                    _ckpt_layer_fn,
                    packed_query_sequence,
                    use_reentrant=False,
                )
            else:
                packed_query_sequence, past_key_values = actual_layer.forward_inference(
                    packed_query_sequence=packed_query_sequence,
                    query_lens=query_lens,
                    packed_query_position_embeddings=packed_query_position_embeddings,
                    packed_query_indexes=packed_query_indexes,
                    past_key_values=past_key_values,
                    key_values_lens=key_values_lens,
                    packed_key_value_indexes=packed_key_value_indexes,
                    update_past_key_values=update_past_key_values,
                    is_causal=is_causal,
                    **extra_inputs,
                )

        if self.use_moe:
            if mode == "und":
                packed_query_sequence = self.norm(packed_query_sequence)
            elif mode == "gen":
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                packed_query_sequence_[packed_text_indexes] = self.norm(
                    packed_query_sequence[packed_text_indexes]
                )
                packed_query_sequence_[packed_vae_token_indexes] = self.norm_moe_gen(
                    packed_query_sequence[packed_vae_token_indexes]
                )
                packed_query_sequence = packed_query_sequence_
        else:
            packed_query_sequence = self.norm(packed_query_sequence)

        if enable_taylorseer:
            self.current["step"] += 1

        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def init_moe(self):
        for name, param in self.named_parameters():
            if "moe_gen" in name:
                original_name = name.replace("_moe_gen", "")
                param.data.copy_(self.state_dict()[original_name].data)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ):

        # Keep wrapper and inner-model contracts aligned even if callers set
        # their training flags independently.
        outputs = self.model.forward_train(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        return outputs

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        loop_start_layer: Optional[int] = None,
        loop_end_layer: Optional[int] = None,
        loop_state_in: Optional[torch.Tensor] = None,
        loop_state_scale: Optional[float] = None,
        loop_external_state: Optional[torch.Tensor] = None,
        loop_external_state_scale: Optional[float] = None,
        loop_residual_alpha: float = 1.0,
        packed_boundary_token_indexes: Optional[torch.Tensor] = None,
        packed_loop_semantic_token_indexes: Optional[torch.Tensor] = None,
        loop_state_mode: str = "semantic_token",
        return_loop_state: bool = False,
        within_step_loop_start: Optional[int] = None,
        within_step_loop_end: Optional[int] = None,
        within_step_loop_repeat: int = 1,
        within_step_loop_damping: float = 1.0,
    ) -> BaseNavitOutputWithPast:

        # This method is also used during LoRA training. Call the matching
        # cache-native implementation explicitly instead of redispatching on
        # the inner module's ``training`` flag.
        outputs = self.model.forward_inference(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
            loop_start_layer=loop_start_layer,
            loop_end_layer=loop_end_layer,
            loop_state_in=loop_state_in,
            loop_state_scale=loop_state_scale,
            loop_external_state=loop_external_state,
            loop_external_state_scale=loop_external_state_scale,
            loop_residual_alpha=loop_residual_alpha,
            packed_boundary_token_indexes=packed_boundary_token_indexes,
            packed_loop_semantic_token_indexes=(packed_loop_semantic_token_indexes),
            loop_state_mode=loop_state_mode,
            return_loop_state=return_loop_state,
            within_step_loop_start=within_step_loop_start,
            within_step_loop_end=within_step_loop_end,
            within_step_loop_repeat=within_step_loop_repeat,
            within_step_loop_damping=within_step_loop_damping,
        )

        return outputs

    def forward_kvcache(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        use_gradient_checkpointing: bool = False,
    ) -> BaseNavitOutputWithPast:
        return self.model.forward_kvcache(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
