"""Native BAGEL text-conditioning helpers for C2 readout preparation."""
from copy import deepcopy
from dataclasses import replace
import torch
from .native import cache_kv, device_inputs
from .numerics import autocast_for
from ..bagel.accelerator import seeded_context

def supplemental_flow(runtime, flow, lengths, positions, caption):
    """Append native text prefill to intact original prompt KV."""
    cache = deepcopy(flow['past_key_values'])
    inputs, new_lengths, new_positions = runtime.model.prepare_prompts(lengths, positions, [caption],
                                          runtime.bundle.tokenizer, runtime.bundle.token_ids)
    with autocast_for(runtime.device):
        cache = runtime.model.forward_cache_update_text(cache, **device_inputs(inputs, runtime.device))
    size = runtime.config['resolution']
    with seeded_context(runtime.device, 0):
        prepared = runtime.model.prepare_vae_latent(new_lengths, new_positions, [(size, size)], runtime.bundle.token_ids)
    prepared.pop('packed_init_noises')
    # Keep native unconditional CFG inputs; only conditional text is augmented.
    out = dict(flow); out.update(device_inputs(prepared, runtime.device)); out['past_key_values'] = cache
    return out


@torch.no_grad()
def correction_source(runtime, flow, lengths, positions, correction):
    """Native append on a clean prompt clone; extract only C, never duplicate P."""
    if len(lengths) != 1:
        raise ValueError('explicit feedback inference is batch one')
    augmented = supplemental_flow(runtime, flow, lengths, positions, correction)
    total = tuple(int(n) for n in augmented['key_values_lens'].tolist())
    count = total[0]-lengths[0]
    source = tuple(replace(kv, keys=kv.keys[lengths[0]:], values=kv.values[lengths[0]:],
                           lengths=(count,), positions=tuple(range(positions[0], positions[0]+count)),
                           token_roles=('correction',)*count)
                   for kv in cache_kv(augmented['past_key_values'], total))
    return source, augmented, count
