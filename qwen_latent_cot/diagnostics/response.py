"""Frozen shared-query GEN response distillation for the C1 text writer."""
from copy import deepcopy
from contextlib import contextmanager
import random
import time
import torch
from .memory import project, rotate, packed_attention, consume_bank
from .native import device_inputs
from .numerics import autocast_for
from .record import FrozenGuard, load_checkpoint, save_checkpoint, write_json, append_json
from .data import file_hash
from .caption import caption_result
from . import distributed as parallel
from ..bagel.accelerator import synchronize, reset_peak_memory_stats, max_memory_allocated


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


def normalized_response_loss(student, teacher, floor=1e-8):
    teacher = teacher.detach().float()
    return (student.float()-teacher).square().mean()/teacher.square().mean().clamp_min(floor)


def response_loss(runtime, bank, teacher, probe):
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
        loss = normalized_response_loss(student, target, runtime.config['response_energy_floor'])
        losses.append(loss)
        with torch.no_grad():
            delta_t = (target-baseline).float(); delta_s = (student-baseline).float()
            metrics.append(dict(layer=index, normalized_mse=float(loss.detach()),
                                mse=float((student.float()-target.float()).square().mean()),
                                teacher_norm=float(target.float().norm()), student_norm=float(student.float().norm()),
                                teacher_condition_delta_norm=float(delta_t.norm()),
                                student_condition_delta_norm=float(delta_s.norm()),
                                condition_delta_cosine=float(torch.nn.functional.cosine_similarity(delta_s.flatten(), delta_t.flatten(), dim=0))))
    return torch.stack(losses).mean(), metrics


def validate_config(runtime):
    cfg = runtime.config
    layers = cfg['response_layers']
    if runtime.writer.topology != 'C' or not layers or len(set(layers)) != len(layers):
        raise ValueError('response distillation requires C and distinct readout layers')
    if any(not 0 <= i < len(runtime.decoder.layers) for i in layers) or max(layers) != len(runtime.decoder.layers)-1:
        raise ValueError('include the deepest UND layer so every C adapter receives gradients')
    if cfg['response_position_reserve'] < runtime.writer.slots or cfg['response_query_count'] <= 0:
        raise ValueError('invalid reserved positions or query count')
    if not cfg['response_t'] or any(not 0 < t < 1 for t in cfg['response_t']):
        raise ValueError('response probes require actual flow times between zero and one')
    if cfg['response_caption_weight'] < 0 or cfg['response_energy_floor'] <= 0:
        raise ValueError('invalid response loss coefficients')


@torch.no_grad()
def validate(runtime, rows, probes, output):
    runtime.writer.eval()
    records = []
    for index, row in enumerate(rows):
        source = runtime.text(row['caption'])[0]
        bank = runtime.memory(source)
        teacher, ids = teacher_condition(runtime, row['caption'])
        caption = caption_result(runtime, row, runtime.caption_prefix(bank), bank.slots, 'C1')
        selected = probes[index % len(probes)]
        errors = []
        for probe in selected:
            loss, metrics = response_loss(runtime, bank, teacher, probe)
            errors.append(float(loss))
            append_json(output/'responses.jsonl', dict(sample_id=row['sample_id'], query_sample_id=probe['sample_id'],
                        t=probe['t'], noise_seed=probe['noise_seed'], response_loss=float(loss), layers=metrics,
                        teacher_token_ids=ids))
        item = dict(sample_id=row['sample_id'], response_loss=sum(errors)/len(errors), **caption)
        append_json(output/'captions.jsonl', item); records.append(item)
    summary = dict(response_loss=sum(r['response_loss'] for r in records)/len(records),
                   nll=sum(r['nll'] for r in records)/len(records),
                   literal_fact_accuracy=sum(r['fact_accuracy'] for r in records)/len(records),
                   samples=len(records), quality_status='shared-query probe only; true GEN quality unverified')
    summary['selection_loss'] = summary['response_loss']+runtime.config['response_caption_weight']*summary['nll']
    write_json(output/'summary.json', summary)
    return summary


def train(runtime, rows, query_rows, output, parent):
    """All ranks jointly update ONLY C adapters; GEN is used solely for fixed probes."""
    from .native import teacher_forcing
    cfg = runtime.config; validate_config(runtime)
    if not parent:
        raise ValueError('response training requires the retained C1 checkpoint')
    load_checkpoint(parent, runtime, 'C1')
    pool = [r for r in rows if r['split'] == 'train' and r['label_type'] == 'clean']
    val = [r for r in rows if r['split'] == 'val' and r['label_type'] == 'clean']
    query_train = [r for r in query_rows if r['split'] == 'train']
    query_val = [r for r in query_rows if r['split'] == 'val']
    if not pool or not val or not query_train or not query_val or any(not r.get('image') or r.get('state') for r in query_rows):
        raise ValueError('verified train/val captions and independent clean train/val probe images are required')
    world, worker = parallel.world_size(), parallel.rank()
    if cfg['effective_batch'] % world:
        raise ValueError('global effective batch must be divisible by world size')
    parallel.broadcast_parameters(runtime.writer)
    guard = FrozenGuard(runtime.model, runtime.bundle.vae)
    if worker == 0:
        print('Preparing frozen GEN query probes on all ranks...', flush=True)
    query_row = query_train[worker % len(query_train)]
    probes = [make_probe(runtime, query_row, t, cfg['seed']+1000+worker) for t in cfg['response_t']]
    torch.save(probes, output/f'query_rank_{worker:02d}.pt')
    query_meta = dict(rank=worker, image=query_row['image'], image_hash=file_hash(query_row['image']),
                      sample_id=query_row['sample_id'], noise_seed=cfg['seed']+1000+worker, t=cfg['response_t'])
    metadata = parallel.gather_samples([query_meta])
    val_probes = []
    if worker == 0:
        print('Preparing held-out query probes and initial validation...', flush=True)
        for index, row in enumerate(query_val):
            val_probes.append([make_probe(runtime, row, t, cfg['seed']+2000+index) for t in cfg['response_t']])
        torch.save(val_probes, output/'query_validation.pt')
        write_json(output/'probe_manifest.json', dict(train=metadata,
                   validation=[dict(sample_id=r['sample_id'], image_hash=file_hash(r['image']), noise_seed=cfg['seed']+2000+i)
                               for i, r in enumerate(query_val)],
                   layers=cfg['response_layers'], query_count=cfg['response_query_count'],
                   readout='pre_output_projection_per_head', shared_position_reserve=cfg['response_position_reserve'],
                   teacher='native_caption_appended_after_neutral_prompt', writer_source='independent_caption_prefill',
                   common_kv='same_detached_prompt_and_full_GEN_self_KV', probe_source='frozen_baseline_independent_of_target_caption'))
        initial = output/'validation_000000'; initial.mkdir()
        baseline = validate(runtime, val, val_probes, initial)
        save_checkpoint(output/'init.pt', runtime, 0, 'C1-read-init', parent)
        save_checkpoint(output/'best.pt', runtime, 0, 'C1-read', parent)
        write_json(output/'selection.json', dict(step=0, rule=cfg['selection'], **baseline))
        best = baseline['selection_loss']
    parallel.barrier()
    if worker == 0:
        print(f'Starting C response distillation: {world} workers, {cfg["steps"]} steps', flush=True)
    optimizer = torch.optim.AdamW(runtime.writer.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    rng = random.Random(cfg['seed']); order = list(range(len(pool))); rng.shuffle(order); cursor = 0
    probe_rng = random.Random(cfg['seed']+31)
    reset_peak_memory_stats(runtime.device); start = time.perf_counter()
    for step in range(1, cfg['steps']+1):
        runtime.writer.train(); optimizer.zero_grad(set_to_none=True)
        sums = [0., 0., 0., 0.]; samples = []; audit = None
        for micro in range(cfg['effective_batch']):
            if cursor == len(order):
                rng.shuffle(order); cursor = 0
            row = pool[order[cursor]]; cursor += 1
            probe_index = probe_rng.randrange(len(probes))
            if micro % world != worker:
                continue
            probe = probes[probe_index]
            teacher, ids = teacher_condition(runtime, row['caption'])
            source = runtime.text(row['caption'])[0]
            with autocast_for(runtime.device):
                bank = runtime.memory(source)
                read_loss, layer_metrics = response_loss(runtime, bank, teacher, probe)
                cap_loss, _ = teacher_forcing(runtime.bundle, runtime.instruction_ids, runtime.answer_ids(row['caption']),
                                             runtime.caption_prefix(bank), bank.slots)
                loss = read_loss+cfg['response_caption_weight']*cap_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite response/caption loss')
            if step == 1 and micro == 0:
                tensors = [tensor for i in cfg['response_layers'] for tensor in (bank.layers[i].keys, bank.layers[i].values)]
                grads = torch.autograd.grad(read_loss, tensors, retain_graph=True)
                if any(not torch.isfinite(g).all() or not g.abs().max() > 0 for g in grads):
                    raise RuntimeError('response gradient must reach Memory K and V at every selected layer')
                audit = [dict(layer=i, k_max=float(grads[2*j].abs().max()), v_max=float(grads[2*j+1].abs().max()))
                         for j, i in enumerate(cfg['response_layers'])]
            (loss/cfg['effective_batch']).backward()
            sums = [sums[0]+float(loss.detach()), sums[1]+float(read_loss.detach()), sums[2]+float(cap_loss.detach()), sums[3]+1]
            samples.append(dict(sample_id=row['sample_id'], query_sample_id=probe['sample_id'], t=probe['t'],
                                noise_seed=probe['noise_seed'], rank=worker, global_micro=micro))
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in runtime.writer.parameters()):
            raise RuntimeError('missing/non-finite C adapter gradients')
        parallel.sum_gradients(runtime.writer)
        norm = torch.nn.utils.clip_grad_norm_(runtime.writer.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); guard.check(); synchronize(runtime.device)
        totals = parallel.sum_statistics(sums, runtime.device)
        global_samples = sorted(parallel.gather_samples(samples), key=lambda r: r['global_micro'])
        if worker == 0:
            append_json(output/'train.jsonl', dict(step=step, loss=totals[0]/totals[3], response_loss=totals[1]/totals[3],
                        caption_nll=totals[2]/totals[3], gradient_norm=float(norm), memory_gradient_audit=audit,
                        world_size=world, global_effective_batch=cfg['effective_batch'], sampled_states=global_samples,
                        elapsed_seconds=time.perf_counter()-start, rank0_peak_memory_bytes=max_memory_allocated(runtime.device)))
        if worker == 0 and (step == 1 or step % 10 == 0):
            print(f'Step {step}: response={totals[1]/totals[3]:.6f}, caption_nll={totals[2]/totals[3]:.6f}', flush=True)
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            parallel.barrier()
            if worker == 0:
                directory = output/f'validation_{step:06d}'; directory.mkdir()
                print(f'Validating step {step}...', flush=True)
                result = validate(runtime, val, val_probes, directory)
                if result['selection_loss'] < best:
                    best = result['selection_loss']; save_checkpoint(output/'best.pt', runtime, step, 'C1-read', parent)
                    write_json(output/'selection.json', dict(step=step, rule=cfg['selection'], **result))
                save_checkpoint(output/'last.pt', runtime, step, 'C1-read', parent)
            parallel.barrier()
    guard.check()
    if worker == 0:
        write_json(output/'training_summary.json', dict(stage='C1-read', steps=cfg['steps'], world_size=world,
                   global_effective_batch=cfg['effective_batch'], parent=parent, parent_hash=file_hash(parent),
                   best_validation_selection_loss=best, frozen_guard='passed', caption_weight=cfg['response_caption_weight'],
                   query_policy='rank_local_fixed_training_queries; held_out_validation_images_and_noise',
                   quality_status='GEN semantics and quality require subsequent real consumer evaluation'))
