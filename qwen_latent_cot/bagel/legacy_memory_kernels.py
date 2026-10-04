# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Frozen Current MemLoop kernels from parent 6f936b7068272b999c0a9b7ec0a1434dbcd9161e.

Only function dispatch names change. Native module tensors remain shared;
no duplicate backbone, new gates, output merge, or re-entry adapter is used.
This compatibility control does not expose the removed training architectures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch.nn.functional import scaled_dot_product_attention
from transformers.utils import ModelOutput

from .modeling.bagel.qwen2_navit import NaiveCache, flash_attn_varlen_func
from .modeling.cache_utils.taylorseer import (
    cal_type,
    derivative_approximation,
    taylor_cache_init,
    taylor_formula,
)
from .modeling.qwen2.modeling_qwen2 import apply_rotary_pos_emb

SOURCE_REVISION = "6f936b7068272b999c0a9b7ec0a1434dbcd9161e"


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


def und_memory_row_mask(
    packed_text_indexes: Optional[torch.Tensor],
    packed_memory_token_indexes: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if packed_text_indexes is None:
        return None
    n = int(packed_text_indexes.shape[0])
    if (
        packed_memory_token_indexes is None
        or int(packed_memory_token_indexes.numel()) == 0
    ):
        return packed_text_indexes.new_zeros((n,), dtype=torch.bool)
    return torch.isin(
        packed_text_indexes,
        packed_memory_token_indexes.to(
            device=packed_text_indexes.device, dtype=packed_text_indexes.dtype
        ),
    )


def project_und_queries(
    q_proj,
    hidden: torch.Tensor,
    packed_text_indexes: Optional[torch.Tensor],
    packed_memory_token_indexes: Optional[torch.Tensor],
) -> torch.Tensor:
    forward_rows = getattr(q_proj, "forward_rows", None)
    if not callable(forward_rows):
        return q_proj(hidden)
    return forward_rows(
        hidden,
        row_mask=und_memory_row_mask(packed_text_indexes, packed_memory_token_indexes),
    )


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


@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    packed_query_sequence: torch.FloatTensor = None
    past_key_values: Optional[NaiveCache] = None
    memory_body_out: Optional[torch.FloatTensor] = None
    memory_round_hiddens: Optional[Tuple[torch.Tensor, ...]] = None
    gen_round_hiddens: Optional[Tuple[torch.Tensor, ...]] = None
    gen_suffix_round_hiddens: Optional[Tuple[torch.Tensor, ...]] = None
    prompt_value_residuals: Optional[Tuple[torch.Tensor, ...]] = None


def _legacy_attention(
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

        packed_query_states[packed_text_indexes] = project_und_queries(
            self.q_proj,
            packed_text_query_sequence,
            packed_text_indexes,
            packed_memory_token_indexes,
        )
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


def _legacy_decoder(
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
) -> BaseNavitOutputWithPast:

    enable_taylorseer = getattr(self, "enable_taylorseer", False)

    if enable_taylorseer and self.current["type"] == "full":
        self.current["module"] = "total"
        taylor_cache_init(cache_dic=self.cache_dic, current=self.current)

    if not enable_taylorseer or (enable_taylorseer and self.current["type"] == "full"):
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

        # Self Attention
        packed_query_sequence, past_key_values = _legacy_attention(
            self.self_attn,
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
            packed_memory_token_indexes=packed_memory_token_indexes,
            block_gen_reads_memory=block_gen_reads_memory,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
            packed_query_sequence = self.mlp(packed_query_sequence)
        elif mode == "gen":
            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
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


def legacy_forward_inference(
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
    packed_boundary_token_indexes: Optional[torch.Tensor] = None,
    within_step_loop_start: Optional[int] = None,
    within_step_loop_end: Optional[int] = None,
    within_step_loop_repeat: int = 1,
    within_step_loop_damping: float = 1.0,
    packed_memory_token_indexes: Optional[torch.Tensor] = None,
    memory_loop_repeat: int = 1,
    memory_loop_start: Optional[int] = None,
    memory_loop_end: Optional[int] = None,
    memory_body_in: Optional[torch.Tensor] = None,
    block_gen_reads_memory: bool = True,
    collect_round_diagnostics: bool = False,
    memory_read_only: bool = False,
    memory_read_adapter_mode: str = "read",
    collect_prompt_value_residuals: bool = False,
    prompt_cache_indexes: Optional[torch.Tensor] = None,
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
        loop_adapter_mode="off",
        packed_memory_token_indexes=None,
        block_gen_reads_memory=False,
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
        if (
            update_past_key_values
            and past_key_values is not None
            and bool(past_key_values.capture_layer_inputs)
        ):
            past_key_values.record_layer_input(
                layer_idx,
                hidden,
                packed_query_indexes,
                packed_key_value_indexes,
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
        if packed_memory_token_indexes is not None:
            layer_kwargs["packed_memory_token_indexes"] = packed_memory_token_indexes
            layer_kwargs["block_gen_reads_memory"] = bool(block_gen_reads_memory)
        use_checkpoint = (
            bool(checkpoint)
            and bool(getattr(self, "gradient_checkpointing", False))
            and self.training
            and torch.is_grad_enabled()
        )

        def _enable_loop_adapters():
            local_states = []
            if loop_adapter_mode != "off":
                modules_fn = getattr(actual_layer, "modules", None)
                iterable = modules_fn() if callable(modules_fn) else ()
                for module in iterable:
                    setter = getattr(module, "set_loop_mode", None)
                    if callable(setter):
                        local_states.append(
                            (
                                module,
                                str(getattr(module, "loop_mode", "off")),
                            )
                        )
                        setter(loop_adapter_mode)
            return local_states

        if use_checkpoint:

            def checkpointed_layer(layer_hidden):
                local_states = _enable_loop_adapters()
                try:
                    layer_output, _ = _legacy_decoder(
                        actual_layer,
                        packed_query_sequence=layer_hidden,
                        **layer_kwargs,
                    )
                    return layer_output
                finally:
                    for module, previous_mode in local_states:
                        module.set_loop_mode(previous_mode)

            hidden = torch.utils.checkpoint.checkpoint(
                checkpointed_layer,
                hidden,
                use_reentrant=False,
            )
            cache = past_key_values
        else:
            local_states = _enable_loop_adapters()
            try:
                hidden, cache = _legacy_decoder(
                    actual_layer,
                    packed_query_sequence=hidden,
                    **layer_kwargs,
                )
            finally:
                for module, previous_mode in local_states:
                    module.set_loop_mode(previous_mode)
        return hidden, cache

    def normalize(hidden):
        if self.use_moe:
            if mode == "und":
                return self.norm(hidden)
            if mode == "gen":
                normalized = torch.zeros_like(hidden)
                normalized[packed_text_indexes] = self.norm(hidden[packed_text_indexes])
                normalized[packed_vae_token_indexes] = self.norm_moe_gen(
                    hidden[packed_vae_token_indexes]
                )
                return normalized
            raise ValueError(f"unsupported MoT mode: {mode}")
        return self.norm(hidden)

    memory_body_out = None
    memory_round_hiddens = None
    gen_round_hiddens = None
    gen_suffix_round_hiddens = None

    loop_repeat = int(within_step_loop_repeat)
    mem_repeat = int(memory_loop_repeat)
    mem_indexes = packed_memory_token_indexes
    memory_body_active = (
        mem_indexes is not None
        and int(mem_indexes.numel()) > 0
        and memory_loop_start is not None
        and memory_loop_end is not None
        and mem_repeat >= 1
    )
    if memory_body_active:
        if update_past_key_values:
            raise ValueError(
                "memory body loop cannot mutate the prompt KV cache; "
                "set update_past_key_values=False"
            )
        s = int(memory_loop_start)
        e = int(memory_loop_end)
        if memory_read_only and mem_repeat != 1:
            raise ValueError("memory_read_only requires memory_loop_repeat=1")
        if memory_read_adapter_mode not in ("off", "read"):
            raise ValueError("memory_read_adapter_mode must be 'off' or 'read'")
        if not 0 <= s < e <= len(self.layers):
            raise ValueError(
                f"memory loop range [{s}, {e}) is invalid for {len(self.layers)} layers"
            )
        indexes = mem_indexes.to(device=packed_query_sequence.device, dtype=torch.long)
        if collect_prompt_value_residuals:
            if not memory_read_only:
                raise ValueError(
                    "prompt V residual collection requires memory_read_only=True"
                )
            if prompt_cache_indexes is None:
                raise ValueError(
                    "prompt V residual collection requires prompt_cache_indexes"
                )
            if int(prompt_cache_indexes.numel()) != int(indexes.numel()):
                raise ValueError("active prompt rows and anchor prompt rows must align")
        block_round0 = bool(block_gen_reads_memory)
        hidden = packed_query_sequence
        for layer_idx in range(0, s):
            hidden, past_key_values = run_layer(
                layer_idx,
                hidden,
                packed_memory_token_indexes=indexes,
                block_gen_reads_memory=block_round0,
            )
        h_base = hidden.clone()
        if memory_body_in is not None:
            hidden = h_base.clone()
            hidden[indexes] = memory_body_in.to(
                dtype=hidden.dtype, device=hidden.device
            )
        else:
            hidden = h_base
        memory_r = hidden[indexes]
        round_memory = [] if collect_round_diagnostics else None
        round_gen = [] if collect_round_diagnostics else None
        round_suffix_gen = [] if collect_round_diagnostics else None
        prompt_value_residuals = [] if collect_prompt_value_residuals else None
        gen_idx = packed_vae_token_indexes
        has_gen = gen_idx is not None and int(gen_idx.numel()) > 0

        def suffix_gen(body_hidden):
            cloned = body_hidden.clone()
            for layer_idx in range(e, len(self.layers)):
                cloned, _ = run_layer(
                    layer_idx,
                    cloned,
                    packed_memory_token_indexes=indexes,
                    block_gen_reads_memory=False,
                )
            cloned = normalize(cloned)
            return cloned[gen_idx] if has_gen else None

        for _round in range(mem_repeat):
            if _round > 0:
                nxt = h_base.clone()
                nxt[indexes] = memory_r
                hidden = nxt
            block_this = block_round0 and _round == 0
            for layer_idx in range(s, e):
                if collect_prompt_value_residuals:
                    decoder_layer = self.layers[layer_idx]
                    actual_layer = getattr(
                        decoder_layer,
                        "_checkpoint_wrapped_module",
                        decoder_layer,
                    )
                    prompt_hidden = actual_layer.input_layernorm(hidden[indexes])
                    dynamic_value = actual_layer.self_attn.v_proj(prompt_hidden).view(
                        -1,
                        actual_layer.self_attn.num_key_value_heads,
                        actual_layer.self_attn.head_dim,
                    )
                    anchor_indexes = prompt_cache_indexes.to(
                        device=dynamic_value.device, dtype=torch.long
                    )
                    anchor_value = past_key_values.value_cache[layer_idx][
                        anchor_indexes
                    ]
                    prompt_value_residuals.append(
                        dynamic_value.to(anchor_value.dtype) - anchor_value
                    )
                hidden, past_key_values = run_layer(
                    layer_idx,
                    hidden,
                    checkpoint=True,
                    loop_adapter_mode=(
                        memory_read_adapter_mode if block_this else "write"
                    ),
                    packed_memory_token_indexes=indexes,
                    block_gen_reads_memory=block_this,
                )
            memory_r = hidden[indexes]
            if collect_round_diagnostics:
                round_memory.append(memory_r)
            if collect_round_diagnostics and has_gen:
                round_gen.append(hidden[gen_idx])
            if collect_round_diagnostics and _round + 1 < mem_repeat:
                suffix_hidden = suffix_gen(hidden)
                if suffix_hidden is not None:
                    round_suffix_gen.append(suffix_hidden)
        if memory_read_only:
            # Phase 1 pair-grounding must stop at the end of the strict
            # Read body.  In particular, do not normalize, execute the
            # suffix, or expose this memory to GEN/context rows.
            return BaseNavitOutputWithPast(
                packed_query_sequence=hidden,
                past_key_values=past_key_values,
                memory_body_out=memory_r,
                memory_round_hiddens=(memory_r,),
                prompt_value_residuals=(
                    tuple(prompt_value_residuals)
                    if collect_prompt_value_residuals
                    else None
                ),
            )
        for layer_idx in range(e, len(self.layers)):
            hidden, past_key_values = run_layer(
                layer_idx,
                hidden,
                packed_memory_token_indexes=indexes,
                block_gen_reads_memory=False,
            )
        packed_query_sequence = normalize(hidden)
        memory_body_out = memory_r
        memory_round_hiddens = (
            tuple(round_memory) if collect_round_diagnostics else None
        )
        gen_round_hiddens = (
            tuple(round_gen) if collect_round_diagnostics and round_gen else None
        )
        if collect_round_diagnostics and has_gen:
            round_suffix_gen.append(packed_query_sequence[gen_idx])
        gen_suffix_round_hiddens = (
            tuple(round_suffix_gen)
            if collect_round_diagnostics and round_suffix_gen
            else None
        )
    elif loop_repeat > 1:
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
    else:
        seq_mem = (
            packed_memory_token_indexes
            if bool(block_gen_reads_memory)
            and packed_memory_token_indexes is not None
            and int(packed_memory_token_indexes.numel()) > 0
            else None
        )
        for layer_idx in range(len(self.layers)):
            packed_query_sequence, past_key_values = run_layer(
                layer_idx,
                packed_query_sequence,
                packed_memory_token_indexes=seq_mem,
                block_gen_reads_memory=bool(block_gen_reads_memory)
                and seq_mem is not None,
            )
        packed_query_sequence = normalize(packed_query_sequence)

    if enable_taylorseer:
        self.current["step"] += 1

    return BaseNavitOutputWithPast(
        packed_query_sequence=packed_query_sequence,
        past_key_values=past_key_values,
        memory_body_out=memory_body_out,
        memory_round_hiddens=memory_round_hiddens,
        gen_round_hiddens=gen_round_hiddens,
        gen_suffix_round_hiddens=gen_suffix_round_hiddens,
    )
