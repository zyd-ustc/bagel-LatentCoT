"""C0 calibration, C1 text bridge, C2 visual transfer and matched controls."""
import random
import time
import torch
from .memory import consume_bank
from .native import teacher_forcing, free_caption
from .data import score_facts, cluster_interval
from .record import append_json, write_json, save_checkpoint, load_checkpoint, FrozenGuard
from . import distributed as parallel
from .numerics import autocast_for
from ..bagel.accelerator import synchronize, reset_peak_memory_stats, max_memory_allocated


def caption_result(runtime, row, prefix, start, stage):
    instruction_ids = runtime.c0_instruction_ids if stage == 'C0' else runtime.instruction_ids
    synchronize(runtime.device)
    start_time = time.perf_counter()
    with autocast_for(runtime.device):
        loss, _ = teacher_forcing(runtime.bundle, instruction_ids, runtime.answer_ids(row['caption']), prefix, start)
        text, ids = free_caption(runtime.bundle, instruction_ids, prefix, start,
                                 runtime.config['max_caption_tokens'])
    if not torch.isfinite(loss):
        raise FloatingPointError('non-finite caption NLL')
    synchronize(runtime.device)
    return dict(decoder_seconds=time.perf_counter()-start_time, nll=float(loss.detach()), text=text, generated_ids=ids, **score_facts(text, row['facts']))


def prepare_source(runtime, row, stage, t=None, seed=0):
    if stage in ('C0', 'C1'):
        source, inputs, positions = runtime.text(row['caption'])
        if max(positions) >= runtime.config['writer_position_start']:
            raise ValueError('text source overlaps fixed writer positions; predeclare a larger start for BOTH groups')
        return source, dict(state_source='text_oracle', source_token_ids=inputs['packed_text_ids'].tolist(),
                            source_positions=inputs['packed_text_position_ids'].tolist(), label_type=row['label_type'])
    x, flow, _, _, metadata = runtime.state(row, t, seed)
    return runtime.visual_source(x, t, flow), metadata


@torch.no_grad()
def evaluate(runtime, rows, stage, output, controls=True):
    result = []; by_id = {r['sample_id']: r for r in rows}
    guard = FrozenGuard(runtime.model, runtime.bundle.vae)
    initialization_only = not any(adapter.up.weight.detach().count_nonzero() for adapter in runtime.writer.adapters)
    for row in rows:
        states = [(None, 0)] if stage in ('C0', 'C1') else [(t, s) for t in runtime.config['eval_t'] for s in runtime.config['noise_seeds']]
        if row.get('state') and stage == 'C2':
            saved = torch.load(row['state'], map_location='cpu', weights_only=True)
            states = [(float(saved['t']), int(saved['noise_seed']))]
        for t, seed in states:
            start_time = time.perf_counter()
            source, meta = prepare_source(runtime, row, stage, t, seed)
            synchronize(runtime.device)
            meta['source_seconds'] = time.perf_counter()-start_time
            writer_times = {}
            if stage == 'C0':
                arms = {'text-oracle': (source, source[0].lengths[0])}
            else:
                bank = runtime.memory(source)
                writer_times['Correct'] = runtime.last_writer_seconds
                tensor_name = f'{row["sample_id"]}_t{t}_seed{seed}_bank.pt'
                torch.save([dict(layer=kv.layer, keys=kv.keys.cpu(), values=kv.values.cpu(),
                                 lengths=kv.lengths, rotated=False) for kv in bank.layers], output/tensor_name)
                arms = {'Correct': (runtime.caption_prefix(bank), bank.slots)}
                if controls:
                    off_bank = runtime.memory(source, True)
                    writer_times['Source-off'] = runtime.last_writer_seconds
                    arms['Source-off'] = (runtime.caption_prefix(off_bank), bank.slots)
                    arms['Memory-masked'] = (None, bank.slots)
                    if row.get('swap_id'):
                        other = by_id.get(row['swap_id'])
                        if other is None:
                            raise ValueError('evaluation slice omitted a swap partner')
                        other_source, other_meta = prepare_source(runtime, other, stage, t, seed)
                        if stage == 'C2' and (other_meta['prompt'], other_meta['t']) != (meta['prompt'], meta['t']):
                            raise ValueError('source swap must preserve t and conditioning')
                        other_bank = runtime.memory(other_source)
                        writer_times['Source-swap'] = runtime.last_writer_seconds
                        writer_times['Memory-swap'] = 0.
                        arms['Source-swap'] = (runtime.caption_prefix(other_bank), bank.slots)
                        arms['Memory-swap'] = arms['Source-swap']
                        if t == 1 and meta['state_source'] == 'controlled':
                            if meta['x_t_hash'] != other_meta['x_t_hash']:
                                raise RuntimeError('pure noise depends on clean image; stop')
                    if t == 1:
                        arms['Pure-noise'] = arms['Correct']
            decoded = {}
            for arm, (prefix, start) in arms.items():
                evidence_arm = 'Source-swap' if arm == 'Memory-swap' else arm
                if evidence_arm not in decoded:
                    decoded[evidence_arm] = caption_result(runtime, row, prefix, start, stage)
                item = dict(sample_id=row['sample_id'], scene_id=row['scene_id'], family_id=row.get('family_id') or row.get('pair_id') or row['scene_id'],
                            pair_id=row.get('pair_id'), split=row['split'], stage=stage, arm=arm,
                            topology=runtime.writer.topology, swap_id=row.get('swap_id') if 'swap' in arm else None,
                            target_token_ids=runtime.answer_ids(row['caption']), caption=row['caption'], initialization_only=initialization_only,
                            readout='post_block', writer_position_ids=list(range(runtime.config['writer_position_start'], runtime.config['writer_position_start']+runtime.writer.slots)),
                            caption_query_start=start, memory_masked=arm=='Memory-masked',
                            source_expert='GEN_native' if stage=='C2' else 'UND_native_text_oracle', **meta,
                            writer_seconds=writer_times.get(arm, 0.),
                            instruction_token_ids=runtime.c0_instruction_ids if stage == 'C0' else runtime.instruction_ids,
                            intervention_evidence_group='donor_bank_swap' if 'swap' in arm else arm,
                            equivalent_to='Source-swap' if arm == 'Memory-swap' else None,
                            **decoded[evidence_arm])
                if 'swap' in arm and other is not None:
                    item['donor_fact_accuracy'] = score_facts(item['text'], other['facts'])['fact_accuracy']
                    shared = [f for f in row['facts'] if f in other['facts']]
                    item['unchanged_fact_accuracy'] = score_facts(item['text'], shared)['fact_accuracy']
                append_json(output/'captions.jsonl', item); result.append(item)
            guard.check()
            synchronize(runtime.device)
            append_json(output/'timing.jsonl', dict(sample_id=row['sample_id'], t=t, noise_seed=seed,
                        seconds=time.perf_counter()-start_time, peak_memory_bytes=max_memory_allocated(runtime.device)))
    check_pure_noise(result)
    grouped = {}
    for row in result:
        key = (row['arm'], row.get('t'), row['label_type'], row['state_source'])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (arm, t, label, source), items in grouped.items():
        summary.append(dict(arm=arm, t=t, label_type=label, state_source=source,
                            nll=sum(r['nll'] for r in items)/len(items),
                            fact_accuracy=cluster_interval(items, 'fact_accuracy'),
                            unknown_assertion_rate=cluster_interval(items, 'unknown_assertion_rate'), samples=len(items)))
    write_json(output/'caption_summary.json', summary)
    return result


def check_pure_noise(rows):
    noise = {}
    for row in rows:
        if row.get('t') == 1 and row['arm'] == 'Correct' and row['state_source'] == 'controlled':
            # Fixed resolution/dtype/prompt are enforced by this runtime. Group by seed,
            # never by the hash whose equality this audit is supposed to establish.
            current = (row['x_t_hash'], row['generated_ids'])
            previous = noise.setdefault(row['noise_seed'], current)
            if previous != current:
                raise RuntimeError('pure-noise state or output differs across x0 labels; stop')


def train(runtime, rows, stage, output, parent=None, scratch=False):
    cfg = runtime.config
    world, worker = parallel.world_size(), parallel.rank()
    if cfg['effective_batch'] % world:
        raise ValueError('global effective_batch must be divisible by world size')
    if stage == 'C2':
        if parent:
            load_checkpoint(parent, runtime, 'C1')
        elif not scratch:
            raise ValueError('C2 requires its matching C1 checkpoint or explicit --scratch')
    elif parent or scratch:
        raise ValueError('C1 starts from shared initialization, not a checkpoint')
    pool = [r for r in rows if r['split'] == 'train' and r['label_type'] == 'clean']
    val = [r for r in rows if r['split'] == 'val' and r['label_type'] == 'clean']
    if not pool or not val or not any(r['facts'] for r in val):
        raise ValueError('training requires verified train/val pools and validation facts')
    if stage == 'C2' and any(r.get('state') for r in pool+val):
        raise ValueError('main C2 train/validation uses controlled clean states only')
    parallel.broadcast_parameters(runtime.writer)
    guard = FrozenGuard(runtime.model, runtime.bundle.vae)
    optimizer = torch.optim.AdamW(runtime.writer.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    rng = random.Random(cfg['seed']); order = list(range(len(pool))); rng.shuffle(order); cursor = 0
    time_rng = random.Random(cfg['seed']+1); noise_rng = random.Random(cfg['seed']+2)
    noise_index = 0
    reset_peak_memory_stats(runtime.device)
    if worker == 0:
        save_checkpoint(output/'init.pt', runtime, 0, stage+'-init', parent)
    parallel.barrier()
    best = (-float('inf'), -float('inf'))
    start_time = time.perf_counter()
    for step in range(1, cfg['steps']+1):
        optimizer.zero_grad(set_to_none=True)
        losses = []; sampled_states = []; gradient_audit = None
        for micro in range(cfg['effective_batch']):
            if cursor == len(order):
                rng.shuffle(order); cursor = 0
            row = pool[order[cursor]]; cursor += 1
            t = time_rng.choice(cfg['train_t']) if stage == 'C2' else None
            seed = noise_rng.randrange(2**63-1); noise_index += 1
            # Every rank advances the same global schedule, but executes its own slice.
            if micro % world != worker:
                continue
            sampled_states.append(dict(sample_id=row['sample_id'], t=t, noise_seed=seed, rank=worker, global_micro=micro))
            source, _ = prepare_source(runtime, row, stage, t, seed)
            with autocast_for(runtime.device):
                bank = runtime.memory(source)
                loss, _ = teacher_forcing(runtime.bundle, runtime.instruction_ids, runtime.answer_ids(row['caption']),
                                          runtime.caption_prefix(bank), bank.slots)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite training NLL')
            if step == 1 and micro == 0:
                tensors = [value for kv in bank.layers for value in (kv.keys, kv.values)]
                grads = torch.autograd.grad(loss, tensors, retain_graph=True)
                if any(not torch.isfinite(g).all() or not g.abs().max() > 0 for g in grads):
                    raise RuntimeError('caption gradient must reach both K and V at every Memory layer')
                gradient_audit = [dict(layer=i, k_max=float(grads[2*i].abs().max()),
                                       v_max=float(grads[2*i+1].abs().max())) for i in range(len(bank.layers))]
            (loss/cfg['effective_batch']).backward(); losses.append(float(loss.detach()))
            if step == 1 and micro == 0:
                gradients = {n: p.grad for n, p in runtime.writer.named_parameters()}
                if any(g is None or not torch.isfinite(g).all() for g in gradients.values()):
                    raise RuntimeError('missing or non-finite adapter gradient')
                if not any(g.abs().max() > 0 for n, g in gradients.items() if '.up.' in n):
                    raise RuntimeError('caption gradients do not reach Memory adapters')
                guard.check()
        parallel.sum_gradients(runtime.writer)
        norm = torch.nn.utils.clip_grad_norm_(runtime.writer.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); guard.check()
        synchronize(runtime.device)
        loss_sum, sample_count = parallel.sum_statistics([sum(losses), len(losses)], runtime.device)
        sampled_states = sorted(parallel.gather_samples(sampled_states), key=lambda item: item['global_micro'])
        if worker == 0:
            append_json(output/'train.jsonl', dict(step=step, sampled_states=sampled_states, memory_gradient_audit=gradient_audit,
                        world_size=world, global_effective_batch=cfg['effective_batch'], nll=loss_sum/sample_count,
                        gradient_norm=float(norm), elapsed_seconds=time.perf_counter()-start_time,
                        rank0_peak_memory_bytes=max_memory_allocated(runtime.device)))
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            parallel.barrier()
            if worker == 0:
                validation_path = output/f'validation_{step:06d}'; validation_path.mkdir()
                scored = evaluate(runtime, val, stage, validation_path, controls=False)
                primary = [r for r in scored if r['arm'] == 'Correct' and r.get('t') != 1]
                accuracy = cluster_interval(primary, 'fact_accuracy')
                if accuracy is None:
                    raise ValueError('validation has no known facts for checkpoint selection')
                # Same rule/budget for L/C. Literal facts and NLL are saved separately.
                score = (accuracy['mean'], -sum(r['nll'] for r in primary)/len(primary))
                if score > best:
                    best = score; save_checkpoint(output/'best.pt', runtime, step, stage, parent)
                    write_json(output/'selection.json', dict(step=step, rule=cfg['selection'], fact_accuracy=score[0], nll=-score[1]))
                save_checkpoint(output/'last.pt', runtime, step, stage, parent)
                torch.save(dict(optimizer=optimizer.state_dict(), step=step, rng=rng.getstate(), order=order,
                                cursor=cursor, noise_index=noise_index, time_rng=time_rng.getstate(), noise_rng=noise_rng.getstate()), output/'optimizer.pt')
            parallel.barrier()
    guard.check()
    if worker == 0:
        write_json(output/'training_summary.json', dict(stage=stage, scratch=scratch, parent=parent,
                    world_size=world, global_effective_batch=cfg['effective_batch'],
                    steps=cfg['steps'], best_validation_fact_accuracy=best[0], frozen_guard='passed',
                    quality_status='not established by NLL or engineering checks'))
