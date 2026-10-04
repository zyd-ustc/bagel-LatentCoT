"""BAGEL layer backend for the shared anchored-loop runner.

Workspace tokens are appended PER SAMPLE only inside extra body executions.
Base, prefix, and suffix retain their exact native sequence and RoPE layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.utils.checkpoint

from .anchored_loop import AnchorState, ratio, run_anchored_loop


@dataclass
class QueryLayout:
    lengths: torch.Tensor
    positions: torch.Tensor
    query_indexes: torch.Tensor
    cache_indexes: torch.Tensor
    cache_lengths: torch.Tensor
    gen_indexes: torch.Tensor
    und_indexes: torch.Tensor

    def kwargs(self, model, hidden, cache):
        cos, sin = model.rotary_emb(hidden, self.positions.unsqueeze(0))
        return dict(
            query_lens=self.lengths,
            packed_query_position_embeddings=(cos.squeeze(0), sin.squeeze(0)),
            packed_query_indexes=self.query_indexes,
            past_key_values=cache,
            key_values_lens=self.cache_lengths,
            packed_key_value_indexes=self.cache_indexes,
            update_past_key_values=False,
            is_causal=False,
            mode="gen",
            packed_vae_token_indexes=self.gen_indexes,
            packed_text_indexes=self.und_indexes,
        )

    def with_memory(self, slots):
        """Return workspace layout, native-query remap, and UND memory indexes."""
        positions, native, memory, queries, cache = [], [], [], [], []
        qoff = merged = oldoff = 0
        for length, past in zip(self.lengths.tolist(), self.cache_lengths.tolist()):
            length, past = int(length), int(past)
            positions.append(self.positions[oldoff : oldoff + length])
            positions.append(
                self.positions.new_full((slots,), int(self.positions[oldoff]))
            )
            native.extend(range(qoff, qoff + length))
            memory.extend(range(qoff + length, qoff + length + slots))
            cache.extend(range(merged, merged + past))
            queries.extend(range(merged + past, merged + past + length + slots))
            qoff += length + slots
            merged += past + length + slots
            oldoff += length

        def ids(values):
            return self.gen_indexes.new_tensor(values)

        native, memory = ids(native), ids(memory)
        expanded = QueryLayout(
            self.lengths + slots,
            torch.cat(positions),
            ids(queries),
            ids(cache),
            self.cache_lengths,
            native[self.gen_indexes],
            torch.cat([native[self.und_indexes], memory]),
        )
        return expanded, native, memory


def forward_anchored_branch(
    model, hidden, layout, cache, modules, config, velocity_head
):
    if not model.use_moe:
        raise ValueError("anchored loop requires BAGEL MoT experts")
    if config.loop_end_layer > len(model.layers):
        raise ValueError("loop body exceeds decoder depth")
    if getattr(model, "enable_taylorseer", False):
        raise ValueError("TaylorSeer cannot approximate recurrent layer executions")
    batch = len(layout.lengths)
    counts = []
    offset = 0
    for length in layout.lengths.tolist():
        counts.append(
            int(
                (
                    (layout.gen_indexes >= offset)
                    & (layout.gen_indexes < offset + length)
                ).sum()
            )
        )
        offset += length
    if not counts or len(set(counts)) != 1:
        raise ValueError("one loop batch must use equal image-token counts")
    gen_shape = (batch, counts[0], hidden.shape[-1])
    original_kwargs = layout.kwargs(model, hidden, cache)

    def layer_call(index, sequence, kwargs, checkpoint=False):
        layer = model.layers[index]

        def call(value):
            # Explicit packed inference supports autograd and avoids selecting
            # the unrelated flex-attention training API when model.train().
            return layer.forward_inference(packed_query_sequence=value, **kwargs)[0]

        if (
            checkpoint
            and torch.is_grad_enabled()
            and getattr(model, "gradient_checkpointing", False)
        ):
            return torch.utils.checkpoint.checkpoint(
                call, sequence, use_reentrant=False
            )
        return call(sequence)

    # Original prefix and body: no scratchpad, gate, adapter, or changed mask.
    for index in range(config.loop_start_layer):
        hidden = layer_call(index, hidden, original_kwargs)
    entry_sequence = hidden.clone()
    entry_gen = hidden[layout.gen_indexes].reshape(gen_shape)
    boundary_anchors, layer_gen_anchors, layer_gen_references = [], [], []
    for index in range(config.loop_start_layer, config.loop_end_layer):
        boundary_anchors.append(hidden[layout.und_indexes].clone())
        layer_gen_anchors.append(hidden[layout.gen_indexes].clone())
        hidden = layer_call(index, hidden, original_kwargs)
        layer_gen_references.append(hidden[layout.gen_indexes].clone())
    base_sequence = hidden.clone()
    base_gen = hidden[layout.gen_indexes].reshape(gen_shape)
    workspace_layout, native_indexes, memory_indexes = layout.with_memory(
        config.memory_slots
    )
    workspace_kwargs = workspace_layout.kwargs(model, hidden, cache)
    workspace_boundaries = native_indexes[layout.und_indexes]

    def body(
        gen,
        memory,
        loop_modules,
        *,
        capture_memory_reads=None,
        memory_read_overrides=None,
        memory_reference_reads=None,
    ):
        sequence = entry_sequence.new_zeros(
            (int(workspace_layout.lengths.sum()), hidden.shape[-1])
        )
        sequence[native_indexes] = entry_sequence
        sequence[workspace_layout.gen_indexes] = gen.reshape(-1, hidden.shape[-1])
        if memory is not None:
            sequence[memory_indexes] = memory.reshape(-1, hidden.shape[-1])
        logs = []
        for local, index in enumerate(
            range(config.loop_start_layer, config.loop_end_layer)
        ):
            # Current query boundaries are frozen at the native layer depth.
            # Cached prompt tensors are read only (update_past_key_values=False).
            sequence = sequence.clone()
            sequence[workspace_boundaries] = boundary_anchors[local]
            if capture_memory_reads is not None:
                capture_memory_reads.append(
                    sequence[memory_indexes]
                    .reshape(batch, config.memory_slots, -1)
                    .clone()
                )
            if memory_read_overrides is not None:
                # Only GEN outputs are retained from this reader layer; UND
                # boundaries reset next layer and M writes are discarded.
                sequence[memory_indexes] = memory_read_overrides[local].reshape(
                    -1, hidden.shape[-1]
                )
            after = layer_call(index, sequence, workspace_kwargs, checkpoint=True)
            # Native transformation is the reference, not the layer input.
            # UND boundaries and memory retain their full native expert update.
            sequence = after.clone()
            reference_gen = layer_gen_references[local]
            if not config.loop_mode.startswith("direct_native_"):
                sequence[workspace_layout.gen_indexes] = loop_modules.gate(
                    local, reference_gen, after[workspace_layout.gen_indexes]
                )
            if memory is not None and config.memory_control in {"zero", "frozen"}:
                sequence = sequence.clone()
                sequence[memory_indexes] = memory.reshape(-1, hidden.shape[-1])
            if config.log_loop_stats:
                logs.append(
                    {
                        "layer": index,
                        "gen_write_ratio": ratio(
                            sequence[workspace_layout.gen_indexes] - reference_gen,
                            reference_gen,
                        ),
                        "raw_gen_correction_ratio": ratio(
                            after[workspace_layout.gen_indexes] - reference_gen,
                            reference_gen,
                        ),
                        "native_transform_ratio": ratio(
                            reference_gen - layer_gen_anchors[local],
                            layer_gen_anchors[local],
                        ),
                        "gate": None
                        if config.loop_mode.startswith("direct_native_")
                        else float(loop_modules.gate_logits[local].detach().sigmoid()),
                        "memory_read_delta_ratio": ratio(
                            memory_read_overrides[local]
                            - memory_reference_reads[local],
                            memory_reference_reads[local],
                        )
                        if memory_read_overrides is not None
                        else 0.0,
                    }
                )
        gen_out = sequence[workspace_layout.gen_indexes].reshape(gen_shape)
        memory_out = (
            sequence[memory_indexes].reshape(batch, config.memory_slots, -1)
            if memory is not None
            else None
        )
        return gen_out, memory_out, logs

    def readout(gen):
        sequence = base_sequence.clone()
        sequence[layout.gen_indexes] = gen.reshape(-1, hidden.shape[-1])
        for index in range(config.loop_end_layer, len(model.layers)):
            sequence = layer_call(index, sequence, original_kwargs, checkpoint=True)
        # Only GEN rows contribute to velocity. Norm equals the native final
        # GEN normalization and suffix never sees workspace tokens.
        return velocity_head(model.norm_moe_gen(sequence[layout.gen_indexes]))

    return run_anchored_loop(
        AnchorState(entry_gen, base_gen), modules, config, body, readout
    )
