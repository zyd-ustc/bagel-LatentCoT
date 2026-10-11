"""C0 calibration, C1 text bridge, C2 visual transfer and matched controls."""
import random
import time
import torch
from .memory import consume_bank
from .native import teacher_forcing, free_caption
from .data import score_facts, cluster_interval
from .record import append_json, write_json, save_checkpoint, load_checkpoint, FrozenGuard
from .numerics import autocast_for
from ..bagel.accelerator import synchronize, reset_peak_memory_stats, max_memory_allocated


def caption_result(runtime, row, prefix, start):
    synchronize(runtime.device)
    start_time = time.perf_counter()
    with autocast_for(runtime.device):
        loss, _ = teacher_forcing(runtime.bundle, runtime.instruction_ids, runtime.answer_ids(row['caption']), prefix, start)
        text, ids = free_caption(runtime.bundle, runtime.instruction_ids, prefix, start,
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
                        arms['Memory-swap'] = (runtime.caption_prefix(other_bank), bank.slots)
                        if t == 1 and meta['state_source'] == 'controlled':
                            if meta['x_t_hash'] != other_meta['x_t_hash']:
                                raise RuntimeError('pure noise depends on clean image; stop')
                    if t == 1:
                        arms['Pure-noise'] = arms['Correct']
            for arm, (prefix, start) in arms.items():
                item = dict(sample_id=row['sample_id'], scene_id=row['scene_id'], family_id=row.get('family_id') or row.get('pair_id') or row['scene_id'],
                            pair_id=row.get('pair_id'), split=row['split'], stage=stage, arm=arm,
                            topology=runtime.writer.topology, swap_id=row.get('swap_id') if 'swap' in arm else None,
                            target_token_ids=runtime.answer_ids(row['caption']), caption=row['caption'], initialization_only=initialization_only,
                            readout='post_block', writer_position_ids=list(range(runtime.config['writer_position_start'], runtime.config['writer_position_start']+runtime.writer.slots)),
                            caption_query_start=start, memory_masked=arm=='Memory-masked',
                            source_expert='GEN_native' if stage=='C2' else 'UND_native_text_oracle', **meta,
                            writer_seconds=writer_times.get(arm, 0.),
                            **caption_result(runtime, row, prefix, start))
                if 'swap' in arm and other is not None:
                    item['donor_fact_accuracy'] = score_facts(item['text'], other['facts'])['fact_accuracy']
                    shared = [f for f in row['facts'] if f in other['facts']]
                    item['unchanged_fact_accuracy'] = score_facts(item['text'], shared)['fact_accuracy']
                append_json(output/'captions.jsonl', item); result.append(item)
            guard.check()
            synchronize(runtime.device)
            append_json(output/'timing.jsonl', dict(sample_id=row['sample_id'], t=t, noise_seed=seed,
                        seconds=time.perf_counter()-start_time, peak_memory_bytes=max_memory_allocated(runtime.device)))
    # Pure-noise controls compare EXACT free outputs across different x0 labels.
    noise = {}
    for row in result:
        if row.get('t') == 1 and row['arm'] == 'Correct' and row['state_source'] == 'controlled':
            key = (row['noise_seed'], row['x_t_hash'])
            previous = noise.setdefault(key, row['generated_ids'])
            if previous != row['generated_ids']:
                raise RuntimeError('pure-noise output differs across x0 labels; stop')
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


def train(runtime, rows, stage, output, parent=None, scratch=False):
    cfg = runtime.config
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
    guard = FrozenGuard(runtime.model, runtime.bundle.vae)
    optimizer = torch.optim.AdamW(runtime.writer.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    rng = random.Random(cfg['seed']); order = list(range(len(pool))); rng.shuffle(order); cursor = 0
    times = list(cfg['train_t']); rng.shuffle(times); noise_index = 0
    reset_peak_memory_stats(runtime.device)
    save_checkpoint(output/'init.pt', runtime, 0, stage+'-init', parent)
    best = (-float('inf'), -float('inf'))
    start_time = time.perf_counter()
    for step in range(1, cfg['steps']+1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(cfg['effective_batch']):
            if cursor == len(order):
                rng.shuffle(order); cursor = 0
            row = pool[order[cursor]]; cursor += 1
            t = times[noise_index % len(times)] if stage == 'C2' else None
            seed = cfg['noise_seeds'][noise_index % len(cfg['noise_seeds'])]; noise_index += 1
            source, _ = prepare_source(runtime, row, stage, t, seed)
            with autocast_for(runtime.device):
                bank = runtime.memory(source)
                loss, _ = teacher_forcing(runtime.bundle, runtime.instruction_ids, runtime.answer_ids(row['caption']),
                                          runtime.caption_prefix(bank), bank.slots)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite training NLL')
            (loss/cfg['effective_batch']).backward(); losses.append(float(loss.detach()))
            if step == 1 and micro == 0:
                gradients = {n: p.grad for n, p in runtime.writer.named_parameters()}
                if any(g is None or not torch.isfinite(g).all() for g in gradients.values()):
                    raise RuntimeError('missing or non-finite adapter gradient')
                if not any(g.abs().max() > 0 for n, g in gradients.items() if '.up.' in n):
                    raise RuntimeError('caption gradients do not reach Memory adapters')
                guard.check()
        norm = torch.nn.utils.clip_grad_norm_(runtime.writer.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); guard.check()
        synchronize(runtime.device)
        append_json(output/'train.jsonl', dict(step=step, nll=sum(losses)/len(losses), gradient_norm=float(norm),
                    elapsed_seconds=time.perf_counter()-start_time, peak_memory_bytes=max_memory_allocated(runtime.device)))
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
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
                            cursor=cursor, noise_index=noise_index), output/'optimizer.pt')
    guard.check()
    write_json(output/'training_summary.json', dict(stage=stage, scratch=scratch, parent=parent,
                steps=cfg['steps'], best_validation_fact_accuracy=best[0], frozen_guard='passed',
                quality_status='not established by NLL or engineering checks'))
