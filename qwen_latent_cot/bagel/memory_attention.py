# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Strict legacy Read mask, using native MoT weights and full residual updates.

Only this attention function differs from native BAGEL. It blocks every
non-Memory query from reading Memory, including SOI/EOI relay paths.
The masked SDPA call preserves the frozen parent implementation's numerics.
No adapters, recurrence, normalization changes or cache writes are added here.
"""
from typing import List, Optional, Tuple
import torch
from torch.nn.functional import scaled_dot_product_attention
from .modeling.bagel.qwen2_navit import NaiveCache
from .modeling.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from .attention import flash_attn_varlen_func

def round0_blocked_slices(
    query_lens: torch.Tensor,
    key_value_lens: torch.Tensor,
    packed_vae_token_indexes: Optional[torch.Tensor],
    packed_memory_token_indexes: Optional[torch.Tensor],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Per-sample (non-memory query, memory key) index pairs to block.

    Memory keys sit in the query half of the merged KV slice, after the past
    prefix of length ``K - Q``. Blocking every current non-memory query closes
    indirect ``memory -> UND boundary -> GEN`` relay paths across layers.
    """

    if packed_memory_token_indexes is None:
        return []
    if int(packed_memory_token_indexes.numel()) == 0:
        return []
    query_lengths = [int(length) for length in query_lens.tolist()]
    key_lengths = [int(length) for length in key_value_lens.tolist()]
    mem = packed_memory_token_indexes.to(dtype=torch.long)
    slices: List[Tuple[torch.Tensor, torch.Tensor]] = []
    query_offset = 0
    for query_length, key_length in zip(query_lengths, key_lengths):
        past = key_length - query_length
        mem_local = mem - query_offset
        mem_in = mem_local[(mem_local >= 0) & (mem_local < query_length)]
        non_mem = torch.ones(query_length, device=mem.device, dtype=torch.bool)
        non_mem[mem_in] = False
        non_mem_in = torch.arange(query_length, device=mem.device, dtype=torch.long)[
            non_mem
        ]
        slices.append((non_mem_in, past + mem_in))
        query_offset += query_length
    return slices


def _sdpa_varlen_inference(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lens: torch.Tensor,
    key_value_lens: torch.Tensor,
    causal: bool,
    blocked_slices: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
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
    sample_index = 0
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
        if blocked_slices is not None and sample_index < len(blocked_slices):
            non_mem_local, mem_keys = blocked_slices[sample_index]
            if int(non_mem_local.numel()) > 0 and int(mem_keys.numel()) > 0:
                if attention_mask is None:
                    attention_mask = torch.ones(
                        1,
                        1,
                        query_length,
                        key_length,
                        device=query.device,
                        dtype=torch.bool,
                    )
                else:
                    attention_mask = attention_mask.clone()
                non_mem_idx = non_mem_local.to(device=query.device)
                mem_idx = mem_keys.to(device=query.device)
                attention_mask[0, 0, non_mem_idx[:, None], mem_idx[None, :]] = False
        sample_index += 1
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


def blocked_memory_attention(
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
    packed_memory_token_indexes=None,
    block_gen_reads_memory=False,
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

        packed_query_states[packed_text_indexes] = self.q_proj(packed_text_query_sequence)
        packed_query_states[packed_vae_token_indexes] = self.q_proj_moe_gen(
            packed_vae_query_sequence
        )

        packed_key_states[packed_text_indexes] = self.k_proj(packed_text_query_sequence)
        packed_key_states[packed_vae_token_indexes] = self.k_proj_moe_gen(
            packed_vae_query_sequence
        )

        packed_value_states[packed_text_indexes] = self.v_proj(
            packed_text_query_sequence
        )
        packed_value_states[packed_vae_token_indexes] = self.v_proj_moe_gen(
            packed_vae_query_sequence
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

        # Preserve native float32 Q/K normalization before RoPE.
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
    cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

    blocked_slices = None
    if bool(block_gen_reads_memory) and mode == "gen":
        blocked_slices = round0_blocked_slices(
            query_lens,
            key_values_lens,
            packed_vae_token_indexes,
            packed_memory_token_indexes,
        )
        if not any(
            int(gen.numel()) > 0 and int(mem.numel()) > 0 for gen, mem in blocked_slices
        ):
            blocked_slices = None

    if flash_attn_varlen_func is None or blocked_slices is not None:
        packed_attn_output = _sdpa_varlen_inference(
            query=packed_query_states,
            key=merged_key_states,
            value=merged_value_states,
            query_lens=query_lens,
            key_value_lens=key_values_lens,
            causal=bool(is_causal),
            blocked_slices=blocked_slices,
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


