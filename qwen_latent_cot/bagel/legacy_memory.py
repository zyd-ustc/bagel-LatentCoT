"""Current MemLoop compatibility control with frozen parent kernels.

R in the unified API corresponds to parent memory_loop_repeat=1+R:
one strict Read followed by R Write rounds. Unlike new memory_only, memory
traverses prefix and suffix, GEN resets at body entry, and no new gate/alpha
is applied. No compatibility state persists between sampler timesteps.
"""

from __future__ import annotations

import torch

from .anchored_loop import LoopResult, memory_slot_stats, ratio
from .legacy_memory_kernels import legacy_forward_inference
from .navit_loop import QueryLayout


def legacy_layout(layout, slots):
    """Exact parent [SOI, M*K, GEN, EOI] query order per sample."""
    positions, native, memory, queries, cached = [], [], [], [], []
    qoff = merged = oldoff = 0
    for length, past in zip(layout.lengths.tolist(), layout.cache_lengths.tolist()):
        length, past = int(length), int(past)
        native.append(qoff)  # original SOI
        memory.extend(range(qoff + 1, qoff + 1 + slots))
        native.extend(range(qoff + 1 + slots, qoff + length + slots))
        positions.append(layout.positions[oldoff : oldoff + 1])
        positions.append(
            layout.positions.new_full((slots,), int(layout.positions[oldoff]))
        )
        positions.append(layout.positions[oldoff + 1 : oldoff + length])
        cached.extend(range(merged, merged + past))
        queries.extend(range(merged + past, merged + past + length + slots))
        qoff += length + slots
        merged += past + length + slots
        oldoff += length
    ids = layout.gen_indexes.new_tensor
    native, memory = ids(native), ids(memory)
    expanded = QueryLayout(
        layout.lengths + slots,
        torch.cat(positions),
        ids(queries),
        ids(cached),
        layout.cache_lengths,
        native[layout.gen_indexes],
        torch.cat([native[layout.und_indexes], memory]),
    )
    return expanded, native, memory


def forward_legacy_memory_branch(
    model, hidden, layout, cache, modules, config, velocity_head
):
    if config.memory_control != "correct":
        raise ValueError(
            "legacy_memory_only preserves Current MemLoop; memory interventions belong to the new workspace arms"
        )
    if not model.use_moe or getattr(model, "enable_taylorseer", False):
        raise ValueError("legacy_memory_only requires native MoT without TaylorSeer")
    # Reference velocity has no workspace and never uses loop gates/adapters.
    base_velocity = None
    if config.log_loop_stats:
        native_output = model.forward_inference(
            packed_query_sequence=hidden.clone(),
            query_lens=layout.lengths,
            packed_query_position_ids=layout.positions,
            packed_query_indexes=layout.query_indexes,
            past_key_values=cache,
            key_values_lens=layout.cache_lengths,
            packed_key_value_indexes=layout.cache_indexes,
            update_past_key_values=False,
            is_causal=False,
            mode="gen",
            packed_vae_token_indexes=layout.gen_indexes,
            packed_text_indexes=layout.und_indexes,
        )
        base_velocity = velocity_head(native_output.packed_query_sequence)[
            layout.gen_indexes
        ]
    workspace, native, mem = legacy_layout(layout, config.memory_slots)
    # Reconstruct the frozen parent's m0 from native SOI/EOI, independent of
    # every trained loop parameter. Parent default initialization seed is 0.
    base = hidden[layout.und_indexes[:2]].float().mean(0).to(hidden.dtype)
    generator = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn((config.memory_slots, hidden.shape[-1]), generator=generator)
    noise = 1e-4 * noise.to(hidden)
    initial_memory = (
        (base.unsqueeze(0) + noise)
        .unsqueeze(0)
        .expand(len(layout.lengths), -1, -1)
        .clone()
    )
    sequence = hidden.new_zeros((int(workspace.lengths.sum()), hidden.shape[-1]))
    sequence[native] = hidden
    sequence[mem] = initial_memory.reshape(-1, hidden.shape[-1])
    output = legacy_forward_inference(
        model,
        packed_query_sequence=sequence,
        query_lens=workspace.lengths,
        packed_query_position_ids=workspace.positions,
        packed_query_indexes=workspace.query_indexes,
        past_key_values=cache,
        key_values_lens=workspace.cache_lengths,
        packed_key_value_indexes=workspace.cache_indexes,
        update_past_key_values=False,
        is_causal=False,
        mode="gen",
        packed_vae_token_indexes=workspace.gen_indexes,
        packed_text_indexes=workspace.und_indexes,
        packed_memory_token_indexes=mem,
        memory_loop_repeat=config.runtime_loop_depth + 1,
        memory_loop_start=config.loop_start_layer,
        memory_loop_end=config.loop_end_layer,
        block_gen_reads_memory=True,
        collect_round_diagnostics=config.log_loop_stats,
    )
    velocity = velocity_head(output.packed_query_sequence)[workspace.gen_indexes]
    velocities = [
        velocity_head(value) for value in output.gen_suffix_round_hiddens or ()
    ]
    stats = []
    if config.log_loop_stats:
        for r, (v, m) in enumerate(zip(velocities, output.memory_round_hiddens)):
            stats.append(
                {
                    "round": r,
                    "legacy_round_type": "read" if r == 0 else "write",
                    "legacy_memory_loop_repeat": config.runtime_loop_depth + 1,
                    "velocity_delta_ratio": ratio(v - base_velocity, base_velocity),
                    "memory_update_ratio": ratio(
                        m.reshape_as(initial_memory)
                        - (
                            initial_memory
                            if r == 0
                            else output.memory_round_hiddens[r - 1].reshape_as(
                                initial_memory
                            )
                        ),
                        initial_memory
                        if r == 0
                        else output.memory_round_hiddens[r - 1].reshape_as(
                            initial_memory
                        ),
                    ),
                    "memory_slot_stats": memory_slot_stats(
                        m.reshape(len(layout.lengths), config.memory_slots, -1)
                    ),
                    "memory_initial_slot_stats": memory_slot_stats(initial_memory),
                }
            )
    return LoopResult(velocity, base_velocity, velocities, stats)
