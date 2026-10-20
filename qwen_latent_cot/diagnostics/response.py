"""Frozen shared-query GEN response distillation for the C1 text writer."""
from copy import deepcopy
from contextlib import contextmanager
import torch
from .memory import project, rotate, packed_attention, consume_bank
from .native import device_inputs
from .numerics import autocast_for


@contextmanager
def capture_queries(runtime, flow, layers, reserve, count):
    """Observe native conditional GEN at a fixed reserved condition position gap."""
    decoder = runtime.decoder
    original = decoder.forward_inference
    handles = []; result = {}
    cache = flow['past_key_values']

    def shifted(*args, **kwargs):
        if kwargs.get('mode') == 'gen' and kwargs.get('past_key_values') is cache:
            kwargs = dict(kwargs)
            kwargs['packed_query_position_ids'] = kwargs['packed_query_position_ids']+reserve
        return original(*args, **kwargs)

    def observer(index):
        def hook(attention, args, kwargs):
            if kwargs.get('mode') != 'gen' or kwargs.get('past_key_values') is not cache:
                return
            if kwargs['query_lens'].numel() != 1:
                raise ValueError('response probes require one packed image')
            q, k, v = project(attention, kwargs['packed_query_sequence'], 'gen',
                              kwargs['packed_text_indexes'], kwargs['packed_vae_token_indexes'])
            rope = kwargs['packed_query_position_embeddings']
            q, k = rotate(q, rope).to(v.dtype), rotate(k, rope).to(v.dtype)
            image_indexes = kwargs['packed_vae_token_indexes']
            selected = image_indexes[torch.linspace(0, len(image_indexes)-1, min(count, len(image_indexes)),
                                                   device=q.device).long()]
            result[index] = dict(q=q[selected].detach().cpu(),
                                 k=torch.cat((cache.key_cache[index], k)).detach().cpu(),
                                 v=torch.cat((cache.value_cache[index], v)).detach().cpu(),
                                 query_indexes=selected.cpu())
        return hook
    try:
        decoder.forward_inference = shifted
        for index in layers:
            handles.append(decoder.layers[index].self_attn.register_forward_pre_hook(observer(index), with_kwargs=True))
        yield result
        if set(result) != set(layers):
            raise RuntimeError('conditional GEN query capture omitted a selected layer')
    finally:
        decoder.forward_inference = original
        for handle in handles:
            handle.remove()


@torch.no_grad()
def make_probe(runtime, row, t, seed):
    cfg = runtime.config
    x, _, _, _, meta = runtime.state(row, t, seed)
    flow, lengths, positions = runtime.flow_inputs(cfg['neutral_prompt'])
    with capture_queries(runtime, flow, cfg['response_layers'], cfg['response_position_reserve'],
                         cfg['response_query_count']) as tensors:
        runtime.velocity(x, t, flow)
    return dict(sample_id=row['sample_id'], t=t, noise_seed=seed, metadata=meta,
                prompt_lengths=lengths, prompt_positions=positions, layers=tensors)


@torch.no_grad()
def teacher_condition(runtime, caption):
    """Native caption prefill appended after the unchanged original prompt."""
    cache, lengths, positions = runtime.prompt_context(runtime.config['neutral_prompt'])
    inputs, _, _ = runtime.model.prepare_prompts(lengths, positions, [caption],
                                                 runtime.bundle.tokenizer, runtime.bundle.token_ids)
    ids = inputs['packed_text_ids'].tolist()
    if len(ids) > runtime.config['response_position_reserve']:
        raise ValueError('caption prefill exceeds fixed response position reserve')
    with autocast_for(runtime.device):
        updated = runtime.model.forward_cache_update_text(deepcopy(cache), **device_inputs(inputs, runtime.device))
    return {i: (updated.key_cache[i][lengths[0]:].detach(), updated.value_cache[i][lengths[0]:].detach())
            for i in runtime.config['response_layers']}, ids


def attention_response(q, common_k, common_v, condition_k, condition_v):
    """Common prompt/self and additional condition compete in ONE softmax."""
    keys = torch.cat((common_k, condition_k)); values = torch.cat((common_v, condition_v))
    return packed_attention(q, keys, values, (len(q),), (len(keys),))


def normalized_response_loss(student, teacher, floor=1e-8, baseline=None):
    teacher = teacher.detach().float()
    increment = teacher if baseline is None else teacher-baseline.detach().float()
    return (student.float()-teacher).square().mean()/increment.square().mean().clamp_min(floor)


def response_loss(runtime, bank, teacher, probe, *, filter_weak_teacher=True):
    memory = consume_bank(runtime.decoder, bank, probe['prompt_positions'])
    losses = []; metrics = []
    for index in runtime.config['response_layers']:
        source = probe['layers'][index]
        q, k, v = (source[name].to(runtime.device).detach() for name in ('q', 'k', 'v'))
        tk, tv = teacher[index]
        with torch.no_grad():
            target = attention_response(q, k, v, tk.to(q.dtype), tv.to(v.dtype))
            baseline = packed_attention(q, k, v, (len(q),), (len(k),))
        mk, mv = memory[index].keys.to(q.dtype), memory[index].values.to(v.dtype)
        student = attention_response(q, k, v, mk, mv)
        loss = normalized_response_loss(student, target, runtime.config['response_energy_floor'], baseline)
        teacher_energy = target.float().square().mean()
        condition_energy = (target-baseline).float().square().mean()
        weak = filter_weak_teacher and (float(teacher_energy) < runtime.config['response_energy_floor']
                    or float(condition_energy) < runtime.config.get('response_condition_energy_min', runtime.config['response_energy_floor']))
        if weak:
            loss = loss * 0.
        losses.append(loss)
        with torch.no_grad():
            delta_t = (target-baseline).float(); delta_s = (student-baseline).float()
            metrics.append(dict(layer=index, normalized_mse=float(loss.detach()),
                                mse=float((student.float()-target.float()).square().mean()),
                                teacher_norm=float(target.float().norm()), student_norm=float(student.float().norm()),
                                teacher_condition_delta_norm=float(delta_t.norm()),
                                student_condition_delta_norm=float(delta_s.norm()),
                                condition_delta_cosine=float(torch.nn.functional.cosine_similarity(delta_s.flatten(), delta_t.flatten(), dim=0))))
            if filter_weak_teacher:
                metrics[-1].update(teacher_energy=float(teacher_energy), condition_energy=float(condition_energy),
                                   weak_teacher_filtered=bool(weak))
    return torch.stack(losses).mean(), metrics
