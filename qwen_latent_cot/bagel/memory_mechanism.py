"""Explicit, inference-only Phase 0.5 memory interventions.

No prompt masking, learned adapters, persistent state or alternate loop depth.
The legacy training loop does not call this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class MemoryControl:
    present: bool
    can_update: bool
    write_source: str
    force_qkv_zero: bool = False


CONTROLS = {
    "native": MemoryControl(False, False, "none"),
    "static_null": MemoryControl(True, False, "null", True),
    "zero_dynamic": MemoryControl(True, True, "zero"),
    "normal": MemoryControl(True, True, "read"),
    "shuffled_dynamic": MemoryControl(True, True, "shuffled_read"),
    "frozen_correct": MemoryControl(True, False, "read"),
}
MODES = tuple(CONTROLS)


def control_for(mode: str) -> MemoryControl:
    if mode not in CONTROLS:
        raise ValueError(f"unknown memory_control_mode {mode!r}; expected {MODES}")
    return CONTROLS[mode]


def zero_memory_qkv(query, key, value, indexes):
    """Clamp AFTER biased projections, QK normalization and rotary embedding."""
    if indexes is None or indexes.numel() == 0:
        raise ValueError("strict null requires memory indexes")
    results = []
    for tensor in (query, key, value):
        result = tensor.clone()
        result[indexes] = 0
        results.append(result)
    return tuple(results)


def select_write_state(read, mode: str, batch_size: int):
    control = control_for(mode)
    if control.write_source in ("null", "zero"):
        return torch.zeros_like(read)
    if control.write_source == "shuffled_read":
        if batch_size < 2 or read.shape[0] % batch_size:
            raise ValueError("shuffled_dynamic requires complete paired memory states")
        return read.reshape(batch_size, -1, read.shape[-1]).roll(1, 0).reshape_as(read)
    if control.write_source == "read":
        return read.clone()
    raise ValueError("native has no Write state")


def run_decoder(hidden, *, mode, indexes, gen_indexes, query_lens,
                body_start, body_end, num_layers, run_layer, normalize,
                diagnostics=None, rounds=2):
    """Run native or prefix + strict Read + reset + Write + suffix exactly once.

    `run_layer` receives only inference controls, never enables adapters.
    Memory position count is checked per packed sample (not just globally).
    """
    control = control_for(mode)
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 2:
        raise ValueError("memory rounds must be an integer >= 2 (Read + Writes)")
    if mode != "normal" and rounds != 2:
        raise ValueError("R ablation is supported only for normal memory")
    s, e = int(body_start), int(body_end)
    if not 0 <= s < e <= num_layers:
        raise ValueError("invalid mechanism body range")
    indexes = (hidden.new_empty(0, dtype=torch.long) if indexes is None
               else indexes.to(device=hidden.device, dtype=torch.long))
    if indexes.numel() and (int(indexes[0]) < 0 or int(indexes[-1]) >= hidden.shape[0]
                            or not bool((indexes[1:] > indexes[:-1]).all())):
        raise ValueError("memory indexes must be sorted, unique and within packed rows")
    offset = 0
    counts = []
    for length in query_lens.tolist():
        counts.append(int(((indexes >= offset) & (indexes < offset + length)).sum()))
        offset += int(length)
    if offset != hidden.shape[0] or len(set(counts)) != 1:
        raise ValueError("memory indexes must tile each packed sample equally")
    if control.present != (indexes.numel() > 0):
        raise ValueError("native requires K=0; memory controls require K>0")
    if mode == "shuffled_dynamic" and len(counts) < 2:
        raise ValueError("shuffled_dynamic requires at least two distinct samples")
    trace = [] if diagnostics is not None else None
    if diagnostics is not None:
        diagnostics["attention"] = trace
    round_index = 0

    def layer(i, h, stage, block=False, frozen=None):
        if control.force_qkv_zero:
            h = h.clone()
            h[indexes] = 0
        trace_start = len(trace) if trace is not None else 0
        result, _ = run_layer(
            i, h,
            packed_memory_token_indexes=indexes if control.present else None,
            block_gen_reads_memory=block,
            force_memory_qkv_zero=control.force_qkv_zero,
            # Exact masses at the first/last body layer only. Hidden/velocity
            # probes still cover every timestep; no full attention maps persist.
            mechanism_attention_trace=trace if i in (s, e - 1) else None,
            mechanism_stage=stage,
        )
        if control.force_qkv_zero or frozen is not None:
            result = result.clone()
            result[indexes] = 0 if control.force_qkv_zero else frozen
        if diagnostics is not None and control.present:
            diagnostics.setdefault("memory_hidden_max", []).append({
                "stage": stage, "layer": i, "round": round_index,
                "max_abs": float(result[indexes].detach().float().abs().max()),
            })
        if trace is not None:
            for record in trace[trace_start:]:
                record["round"] = round_index
        return result

    if not control.present:
        for i in range(num_layers):
            hidden = layer(i, hidden, "native")
            if diagnostics is not None and i + 1 == e:
                diagnostics["gen_body_hidden"] = hidden[gen_indexes].detach().clone()
        hidden = normalize(hidden)
    else:
        for i in range(s):
            hidden = layer(i, hidden, "prefix", block=True)
        entry = hidden.clone()
        for i in range(s, e):
            hidden = layer(i, hidden, "read", block=True)
        read = hidden[indexes].clone()
        write = select_write_state(read, mode, len(counts))
        frozen = write if not control.can_update else None
        if diagnostics is not None:
            diagnostics["memory_read"] = read.detach().clone()
            diagnostics["memory_write_input"] = write.detach().clone()
        for round_index in range(1, rounds):
            hidden = entry.clone()  # Reset ALL non-memory rows on EVERY Write.
            hidden[indexes] = write
            if diagnostics is not None:
                diagnostics.setdefault("memory_write_inputs", []).append(write.detach().clone())
            for i in range(s, e):
                hidden = layer(i, hidden, "write", frozen=frozen)
            write = hidden[indexes].clone()
        if diagnostics is not None:
            diagnostics["gen_body_hidden"] = hidden[gen_indexes].detach().clone()
        for i in range(e, num_layers):
            hidden = layer(i, hidden, "suffix", frozen=frozen)
        hidden = normalize(hidden)
    if diagnostics is not None:
        diagnostics["gen_suffix_hidden"] = hidden[gen_indexes].detach().clone()
    return hidden


@torch.no_grad()
def append_attention_stats(trace, *, query, key, query_lens, key_lens,
                           memory_indexes, gen_indexes, blocked_slices,
                           layer, stage, chunk_size=32):
    """Exact mean attention masses, chunked over query rows; no full map saved.

    Q/K already include normalization/RoPE. GQA repeats KV heads in the same
    order as SDPA. Only generation (noncausal) calls invoke this routine.
    """
    if trace is None:
        return
    if query.shape[1] % key.shape[1]:
        raise ValueError("query heads must be a multiple of KV heads")
    q_offset = k_offset = 0
    empty = torch.empty(0, device=query.device, dtype=torch.long)
    memory_indexes = empty if memory_indexes is None else memory_indexes
    for sample, (qlen, klen) in enumerate(zip(query_lens.tolist(), key_lens.tolist())):
        qlen, klen = int(qlen), int(klen)
        prefix = klen - qlen
        mem = memory_indexes[(memory_indexes >= q_offset) & (memory_indexes < q_offset + qlen)] - q_offset
        gen = gen_indexes[(gen_indexes >= q_offset) & (gen_indexes < q_offset + qlen)] - q_offset
        keys = key[k_offset:k_offset + klen].float().transpose(0, 1)
        keys = keys.repeat_interleave(query.shape[1] // key.shape[1], dim=0)
        groups = {"prompt": torch.arange(prefix, device=query.device),
                  "memory": prefix + mem, "gen": prefix + gen}
        blocked_rows_mask = torch.zeros(qlen, device=query.device, dtype=torch.bool)
        blocked_keys_mask = torch.zeros(klen, device=query.device, dtype=torch.bool)
        if blocked_slices is not None:
            blocked_rows, blocked_keys = blocked_slices[sample]
            blocked_rows_mask[blocked_rows] = True
            blocked_keys_mask[blocked_keys] = True
        row = {"sample": sample, "layer": int(layer), "stage": stage}
        for label, queries in (("gen", gen), ("memory", mem)):
            totals = {name: 0.0 for name in groups}
            count = 0
            for part in queries.split(chunk_size):
                if part.numel() == 0:
                    continue
                q = query[q_offset + part].float().transpose(0, 1)
                # The surrounding model AMP context must not downcast this
                # diagnostic's explicitly FP32 score/softmax calculation.
                with torch.autocast(device_type=query.device.type, enabled=False):
                    logits = torch.matmul(q, keys.transpose(-1, -2)) / math.sqrt(query.shape[-1])
                    if blocked_slices is not None:
                        mask = blocked_rows_mask[part, None] & blocked_keys_mask[None, :]
                        logits.masked_fill_(mask[None], float("-inf"))
                    weights = logits.softmax(-1)
                count += int(part.numel()) * query.shape[1]
                for name, cols in groups.items():
                    totals[name] += float(weights[..., cols].sum())
            for name in groups:
                row[f"{label}_to_{name}"] = totals[name] / count if count else None
        trace.append(row)
        q_offset += qlen
        k_offset += klen
