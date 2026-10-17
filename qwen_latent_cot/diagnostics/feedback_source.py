"""Formal inference sources: original prompt at every UND writer layer.

Capture stays GEN-only. Composition preserves native rotated keys; memory's
own KV is added inside und_block. Training and inference use this same source.
"""
from dataclasses import replace
import torch
from .memory import KV, join_prefix
from .native import cache_kv
from .numerics import autocast_for


def prompt_source(flow, lengths, positions):
    """Read the original text cache, never a memory consumption context."""
    if not lengths or len(lengths) != len(positions) or any(n != p for n, p in zip(lengths, positions)):
        raise ValueError('formal feedback requires a clean text-only prompt cache')
    token_positions = tuple(p for n in lengths for p in range(n))
    return tuple(replace(kv, positions=token_positions, token_roles=('prompt',)*len(kv.keys))
                 for kv in cache_kv(flow['past_key_values'], lengths))


def compose_source(prompt, extra, layers=None):
    """Prompt at every layer; extra KV at the selected GEN prefix layers."""
    count = len(prompt)
    selected = set(range(count) if layers is None else layers)
    if not selected or any(i < 0 or i >= count for i in selected):
        raise ValueError('invalid source layer window')
    extra = {kv.layer: kv for kv in extra}
    if len(extra) < len(selected) or not selected <= extra.keys():
        raise ValueError('missing selected source layers')
    result = []
    for index, pkv in enumerate(prompt):
        if pkv.layer != index or not pkv.rotated:
            raise ValueError('prompt requires full-depth native rotated KV')
        if index not in selected:
            result.append(pkv)
            continue
        ekv = extra[index]
        if not ekv.rotated:
            raise ValueError('extra source must retain its native RoPE')
        keys, values, lengths = join_prefix(pkv, ekv.keys, ekv.values, ekv.lengths)
        def join_meta(a, b):
            if not a or not b:
                return ()
            merged = []; po = eo = 0
            for np, ne in zip(pkv.lengths, ekv.lengths):
                merged.extend(a[po:po+np]); merged.extend(b[eo:eo+ne])
                po += np; eo += ne
            return tuple(merged)
        result.append(KV(index, keys, values, lengths, True, ekv.source_kind, ekv.t,
                         ekv.round, join_meta(pkv.token_roles, ekv.token_roles),
                         join_meta(pkv.positions, ekv.positions), ekv.expert))
    return tuple(result)


def write_feedback(runtime, prompt, extra, *, layers=None, t=None,
                   writer=None, detach_source=True):
    """Single full-depth native UND writer entrance."""
    writer = runtime.writer if writer is None else writer
    if writer.topology != 'C' or writer.slots != 16:
        raise ValueError('formal feedback uses continuous UND and 16 slots')
    source = compose_source(prompt, extra, layers)
    positions = torch.arange(runtime.config['writer_position_start'],
                             runtime.config['writer_position_start']+writer.slots,
                             device=runtime.device)
    with autocast_for(runtime.device):
        return writer(runtime.seeds, source, positions, source_kind='text', t=t, detach_source=detach_source)
