"""State-matched controls, per-round evidence, and independently completed sampling."""
from dataclasses import replace
from contextlib import contextmanager
import csv
import random
import torch
from .flow_training import conditional_gen, flow_mse
from .loop_training import memory_loop
from .visual_objectives import semantic_result, state_probe, text_keep, bank_response, generation_caption
from .record import FrozenGuard, append_json, write_json
from .data import tensor_hash, velocity_stats, cluster_interval, file_hash
from .schedule import native_times, nearest_node
from .native import gen_memory_call
from .memory import MemoryBank, KV
from .gen import supplemental_flow


def states_for(row, config):
    if row.get('state') or row.get('label_kind') in ('observation_proxy', 'target_observation_gap'):
        return [(float(row['t']), int(row['noise_seed']))]
    return [(t, seed) for t in config['eval_t'] for seed in config['noise_seeds']]


def state_diagnostics(bank, reference, output, baseline):
    layers = []
    for kv, old in zip(bank.layers, reference.layers):
        layers.append(dict(layer=kv.layer, keys=velocity_stats(kv.keys, old.keys),
                           values=velocity_stats(kv.values, old.values),
                           key_norm=float(kv.keys.float().norm()), value_norm=float(kv.values.float().norm())))
    return dict(memory=layers, conditional_velocity=velocity_stats(output.velocity, baseline.velocity))


@contextmanager
def observe_layers(runtime, cache):
    """Read hooks only; distinguish raw/normalized hidden and attention output."""
    handles = []; tensors = {}
    def input_hook(index):
        def hook(module, args, kwargs):
            if kwargs.get('past_key_values') is cache and kwargs.get('mode') == 'gen':
                tensors.setdefault(index, {})['hidden_before_norm'] = kwargs['packed_query_sequence'].detach().float().cpu()
        return hook
    def attention_hook(index):
        def pre(module, args, kwargs):
            if kwargs.get('past_key_values') is cache and kwargs.get('mode') == 'gen':
                tensors.setdefault(index, {})['hidden_after_norm'] = kwargs['packed_query_sequence'].detach().float().cpu()
        def post(module, args, kwargs, output):
            if kwargs.get('past_key_values') is cache and kwargs.get('mode') == 'gen':
                tensors.setdefault(index, {})['attention_output'] = output[0].detach().float().cpu()
        return pre, post
    try:
        for index, layer in enumerate(runtime.decoder.layers):
            handles.append(layer.register_forward_pre_hook(input_hook(index), with_kwargs=True))
            pre, post = attention_hook(index)
            handles.append(layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(layer.self_attn.register_forward_hook(post, with_kwargs=True))
        def head_hook(module, args, result):
            if 'final_head' not in tensors:
                tensors['final_head'] = dict(normalized_hidden=args[0].detach().float().cpu(),
                                             velocity_head_output=result.detach().float().cpu())
        handles.append(runtime.model.llm2vae.register_forward_hook(head_hook))
        yield tensors
    finally:
        for handle in handles: handle.remove()


@torch.no_grad()
def evaluate(student, rows, output, *, rounds=1, controls=True, save_images=False,
             continue_sampling=False, layer_diagnostics=False, donor_rows=None):
    runtime = student.runtime; cfg = runtime.config
    guard = FrozenGuard(runtime.model, runtime.bundle.vae, student.text_teacher, exclude_trainable=True)
    by_id = {r['sample_id']: r for r in (donor_rows if donor_rows is not None else rows)}; records = []; blind = []
    output.mkdir(parents=True, exist_ok=True)
    fixed_source = None; fixed_metadata = None
    for row in rows:
        for t, seed in states_for(row, cfg):
            requested_t = t
            if continue_sampling and row['label_kind'] == 'target_clean' and not row.get('state'):
                t = nearest_node(cfg, requested_t)[1]
            state = runtime.training_state(row, t, seed)
            base = conditional_gen(runtime, state)
            if fixed_source is None:
                fixed_source = base.source; fixed_metadata = dict(state.metadata, sample_id=row['sample_id'])
            if cfg.get('stage') == 'C2-observe':
                from .loop_training import LoopOutput
                bank = student.write(base.source, t)
                loop = LoopOutput(base, [bank], [base], 1)
            else:
                loop = memory_loop(student, state, rounds, diagnostics=True)
            arms = {'R1': (loop.banks[0], loop.exits[0])}
            provenance = {'R1': dict(state.metadata, source_round=0),
                          'R2-new': dict(state.metadata, source_round=1),
                          'R2-old': dict(state.metadata, source_round=0),
                          'R2-fixed': dict(state.metadata, source_round=0, reused_M1=True)}
            if rounds == 2:
                arms['R2-new'] = (loop.banks[-1], loop.exits[-1])
                if controls:
                    for policy in ('old', 'fixed'):
                        other = memory_loop(student, state, 2, feedback=policy, diagnostics=True)
                        arms['R2-'+policy] = (other.banks[-1], other.exits[-1])
                    torch.testing.assert_close(arms['R2-fixed'][1].velocity, arms['R1'][1].velocity,
                                               rtol=cfg.get('numeric_rtol', .02), atol=cfg.get('numeric_atol', .002))
            donor = by_id.get(row.get('swap_id'))
            if controls:
                direct = student.write(base.source, t, visual=False)
                off = student.write(base.source, t, source_off=True)
                fixed = student.write(fixed_source, t)
                neutral = student.writer.fixed_bank(runtime.seeds)
                zero = MemoryBank(tuple(KV(kv.layer, torch.zeros_like(kv.keys), torch.zeros_like(kv.values),
                                           kv.lengths, False) for kv in neutral.layers), 'fixed', neutral.slots)
                provenance.update({'C1-direct-GEN': dict(state.metadata, source_round=0, writer_architecture=student.architecture),
                                   'Source-off': dict(source='ignored', t=t),
                                   'Fixed-source': fixed_metadata,
                                   'G-fixed': dict(source='zero_KV'),
                                   'G-neutral': dict(source='neutral_seed'),
                                   'G-layout': dict(source='masked'),
                                   'R0': dict(source='no_Memory'),
                                   'Target-C1-oracle': dict(source='target_caption_C1', oracle=True),
                                   'Target-text-oracle': dict(source='target_caption_native_prefill', oracle=True)})
                for name, bank, masked in [('C1-direct-GEN', direct, False), ('Source-off', off, False),
                                           ('Fixed-source', fixed, False), ('G-fixed', zero, False), ('G-neutral', neutral, False),
                                           ('G-layout', neutral, True)]:
                    arms[name] = (bank, conditional_gen(runtime, state, bank, masked=masked, capture=False))
                arms['R0'] = (neutral, base)
                if row.get('target_caption') and student.text_teacher is not None:
                    teacher = student.teacher_bank(row['target_caption'])
                    arms['Target-C1-oracle'] = (teacher, conditional_gen(runtime, state, teacher, capture=False))
                    oracle_flow = supplemental_flow(runtime, state.flow, state.lengths, state.positions, row['target_caption'])
                    lens = tuple(oracle_flow['key_values_lens'].tolist())
                    pos = [int(oracle_flow['packed_position_ids'][0])]
                    oracle_state = replace(state, flow=oracle_flow, lengths=lens, positions=pos)
                    arms['Target-text-oracle'] = (teacher, conditional_gen(runtime, oracle_state, capture=False))
                if donor:
                    donor_state = runtime.training_state(donor, t, seed)
                    if (donor_state.metadata['prompt'], donor_state.t) != (state.metadata['prompt'], state.t):
                        raise ValueError('swap must preserve actual prompt and t')
                    if cfg.get('stage') == 'C2-observe':
                        donor_base = conditional_gen(runtime, donor_state)
                        donor_loop = LoopOutput(donor_base, [student.write(donor_base.source, t)], [donor_base], 1)
                    else:
                        donor_loop = memory_loop(student, donor_state, 1, diagnostics=True)
                    provenance['Source-swap'] = dict(donor_state.metadata, sample_id=donor['sample_id'], source_round=0)
                    provenance['R2-swap'] = dict(donor_state.metadata, sample_id=donor['sample_id'], source_round=1)
                    bank = donor_loop.banks[0]
                    arms['Source-swap'] = (bank, conditional_gen(runtime, state, bank, capture=False))
                    if rounds == 2:
                        swapped = memory_loop(student, state, 2, feedback='swap',
                                              donor_source=donor_loop.exits[0].source, diagnostics=True,
                                              donor_metadata=donor_state.metadata)
                        arms['R2-swap'] = (swapped.banks[-1], swapped.exits[-1])
            teacher_caption = generation_caption(row, cfg) if student.text_teacher is not None else None
            probe = state_probe(runtime, state) if teacher_caption else None
            response_teacher = student.teacher_bank(teacher_caption) if teacher_caption else None
            for arm, (bank, prediction) in arms.items():
                item = dict(sample_id=row['sample_id'], scene_id=row['scene_id'], family_id=row['family_id'],
                            split=row['split'], arm=arm, t=t, requested_t=requested_t, noise_seed=seed, round=prediction.round,
                            label_kind=row['label_kind'], confirmed_label=row['confirmed_label'],
                            state=state.metadata, source_provenance=provenance.get(arm), caption=semantic_result(runtime, row, bank),
                            conditional_flow_mse=float(flow_mse(prediction, state)) if state.metadata.get('flow_supervision') else None,
                            diagnostics=state_diagnostics(bank, loop.banks[0], prediction, base),
                            caption_probe='UND reads associated Memory bank; not a GEN image caption',
                            response_position_reserve=16, writer_position_start=cfg['writer_position_start'],
                            oracle='oracle' in arm, frozen_guard='passed')
                if probe is not None and arm not in ('R0', 'Target-text-oracle'):
                    read_loss, read_metrics = bank_response(runtime, bank, response_teacher, probe)
                    item['read_response'] = dict(loss=float(read_loss), layers=read_metrics)
                if layer_diagnostics:
                    inspected_state = oracle_state if arm == 'Target-text-oracle' else state
                    inspected_bank = None if arm in ('R0', 'Target-text-oracle') else bank
                    with observe_layers(runtime, inspected_state.flow['past_key_values']) as observed:
                        repeated = conditional_gen(runtime, inspected_state, inspected_bank,
                                                   capture=False, masked=arm=='G-layout')
                    torch.testing.assert_close(prediction.velocity, repeated.velocity,
                                               rtol=cfg.get('numeric_rtol', .02), atol=cfg.get('numeric_atol', .002))
                    name = f'layers_{row["sample_id"]}_t{t}_s{seed}_{arm}.pt'
                    torch.save(observed, output/name); item['layer_tensors'] = name
                if save_images:
                    if arm == 'R0':
                        conditional, cfg_velocity = runtime.velocity(state.x_t, t, state.flow)
                    elif arm == 'Target-text-oracle':
                        conditional, cfg_velocity = runtime.velocity(state.x_t, t, oracle_state.flow)
                    else:
                        conditional, cfg_velocity = runtime.gen_arm(state.x_t, t, state.flow, bank,
                                                                    state.lengths, state.positions, arm=='G-layout')
                    item['cfg_velocity'] = velocity_stats(cfg_velocity, runtime.velocity(state.x_t, t, state.flow)[1])
                    image_state = state.x_t-t*cfg_velocity
                    item['image_policy'] = 'local_x0_prediction'
                    if continue_sampling:
                        from .schedule import native_suffix
                        times = native_suffix(cfg, t)
                        image_state = runtime.continue_once(state.x_t, t, cfg_velocity,
                                                           oracle_state.flow if arm=='Target-text-oracle' else state.flow, times)
                        item['image_policy'] = 'single_native_node_injection_then_native_suffix'
                    key = f'blind_{len(blind):06d}.png'
                    runtime.image(image_state).save(output/key)
                    item['image'] = key
                    blind.append(dict(image=key, sample_id=row['sample_id'], family_id=row['family_id'],
                                      arm=arm, t=t, noise_seed=seed, policy=item['image_policy']))
                append_json(output/'metrics_by_t_round.jsonl', item); records.append(item)
            guard.check()
    confirmed = [dict(r, nll=r['caption']['nll'], fact_accuracy=r['caption']['fact_accuracy'],
                      unknown_assertion_rate=r['caption']['unknown_assertion_rate'])
                 for r in records if r['confirmed_label']]
    summaries = []
    for key in sorted({(r['arm'], r['t'], r['label_kind']) for r in records}):
        group = [r for r in confirmed if (r['arm'], r['t'], r['label_kind']) == key]
        summaries.append(dict(arm=key[0], t=key[1], label_kind=key[2], confirmed_samples=len(group),
                              fact_accuracy=cluster_interval(group, 'fact_accuracy'),
                              unknown_assertion_rate=cluster_interval(group, 'unknown_assertion_rate'),
                              t1_scope='no_image_evidence_negative_control' if key[1] == 1 else None))
    paired = []
    for treatment, control in [('R1','Source-off'), ('R1','Source-swap'), ('R2-new','R1'), ('R2-new','R2-old')]:
        controls_by_key = {(r['sample_id'],r['t'],r['noise_seed'],r['label_kind']):r for r in confirmed if r['arm']==control}
        pairs = []
        for r in confirmed:
            key = (r['sample_id'],r['t'],r['noise_seed'],r['label_kind'])
            other = controls_by_key.get(key)
            if r['arm']==treatment and r['t']!=1 and other and r['fact_accuracy'] is not None and other['fact_accuracy'] is not None:
                pairs.append(dict(r, difference=r['fact_accuracy']-other['fact_accuracy']))
        for kind in sorted({r['label_kind'] for r in pairs}):
            paired.append(dict(treatment=treatment,control=control,label_kind=kind,
                               family_paired_fact_difference=cluster_interval([r for r in pairs if r['label_kind']==kind], 'difference')))
    write_json(output/'summary.json', dict(paired_differences=paired, status='executed; acceptance not inferred from NLL', groups=summaries,
               total_records=len(records), unconfirmed_label_records=sum(not r['confirmed_label'] for r in records)))
    if blind:
        random.Random(cfg['seed']).shuffle(blind)
        write_json(output/'blind_mapping.json', blind)
        with (output/'blind_annotations.csv').open('w', newline='') as stream:
            fields = ['image', 'target_facts', 'original_fact_damage', 'quality', 'artifacts', 'reviewer']
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerows({'image': row['image']} for row in blind)
    return records


@torch.no_grad()
def evaluate_feedback(student, rows, output, *, updates=1, donor_rows=None):
    """Fixed states, full CFG suffixes and paired content controls; no NLL ranking."""
    from .feedback_source import prompt_source
    from .schedule import native_suffix
    runtime = student.runtime; cfg = runtime.config
    student.eval(); records = []
    by_id = {r['sample_id']:r for r in (donor_rows if donor_rows is not None else rows)}
    output.mkdir(parents=True, exist_ok=True)
    fields = ['image', 'sample_id', 'arm', 'target_fact_fix', 'correct_fact_preservation', 'quality', 'notes']
    with (output/'annotations.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            for t, seed in states_for(row, cfg):
                if not row.get('state'):
                    t = nearest_node(cfg, t)[1]
                state = runtime.training_state(row, t, seed)
                loop = memory_loop(student, state, updates, diagnostics=True)
                arms = {'Base': None, 'New': loop.banks[-1]}
                arms['Layout'] = student.writer.fixed_bank(runtime.seeds)
                if updates > 1:
                    for control in ('old', 'fixed'):
                        arms[control.capitalize()] = memory_loop(student, state, updates, feedback=control).banks[-1]
                p = state.prompt_kv if state.prompt_kv is not None else prompt_source(state.flow, state.lengths, state.positions)
                arms['Source-off'] = student.write(loop.baseline.source, t, prompt=p, source_off=True)
                donor = by_id.get(row.get('swap_id'))
                if donor:
                    ds = runtime.training_state(donor, donor.get('t',t), donor.get('noise_seed',seed))
                    if ds.metadata['prompt'] != state.metadata['prompt'] or ds.t != t:
                        raise ValueError('swap must preserve prompt and actual t')
                    source = conditional_gen(runtime, ds).source
                    arms['Source-swap'] = student.write(source, t, prompt=p)
                if student.text_teacher is not None and row.get('feedback') and not row['feedback']['satisfied']:
                    arms['Text-memory'] = student.teacher_bank(row['feedback']['correction'], state)
                for arm, bank in arms.items():
                    velocity = (runtime.velocity(state.x_t, t, state.flow) if bank is None else
                                runtime.gen_arm(state.x_t, t, state.flow, bank, state.lengths, state.positions,
                                                masked=arm=='Layout'))
                    final = runtime.continue_once(state.x_t, t, velocity[1], state.flow, native_suffix(cfg, t))
                    image = f"{row['sample_id']}_t{t:.8f}_s{seed}_{arm}.png"
                    runtime.image(final).save(output/image)
                    item = dict(sample_id=row['sample_id'], arm=arm, image=image, t=t, noise_seed=seed,
                                prompt=state.metadata['prompt'], state_hash=state.metadata['x_t_hash'],
                                passes=updates+1 if arm == 'New' else None,
                                image_policy='single_node_then_native_suffix', quality_status='pending review')
                    append_json(output/'final_images.jsonl', item); records.append(item)
                    writer.writerow({key:item.get(key, '') for key in fields})
    return records


@torch.no_grad()
def evaluate_text_keep(student, text_rows, visual_rows, output):
    runtime = student.runtime
    row = visual_rows[0]; t, seed = states_for(row, runtime.config)[0]
    probe = state_probe(runtime, runtime.training_state(row, t, seed))
    for row in text_rows:
        loss, metrics = text_keep(student, row, probe)
        bank = student.write(runtime.text(row['caption'])[0], None, visual=False)
        teacher = student.teacher_bank(row['caption'])
        append_json(output/'text_keep.jsonl', dict(sample_id=row['sample_id'], loss=float(loss),
                    metrics=metrics, student=semantic_result(runtime, dict(row, label_kind='target_clean'), bank),
                    frozen_teacher=semantic_result(runtime, dict(row, label_kind='target_clean'), teacher),
                    probe_state=probe['metadata']))


@torch.no_grad()
def complete_sampling(student, row, seed, rounds=2, arm='new', *, output=None):
    """Independent native trajectory; neutral seeds reset at EVERY denoising node.

    First 20 updates (not all 50 nodes) use feedback; all CFG branches remain
    native. Each node uses final-round velocity for exactly ONE Euler update.
    """
    runtime = student.runtime; cfg = runtime.config
    from .explicit_feedback import initial_noise
    from .flow_training import TrainingState
    flow, lengths, positions = runtime.flow_inputs(row['prompt'])
    x = initial_noise(runtime, flow, seed)
    initial = TrainingState(x, None, None, 1., flow, lengths, positions,
                            dict(prompt=row['prompt'], t=1., noise_seed=seed, x_t_hash=tensor_hash(x),
                                 state_source='prompt_trajectory'))
    state = initial.x_t; times = native_times(cfg); records = []
    from ..bagel.accelerator import synchronize
    import time
    synchronize(runtime.device); start = time.perf_counter()
    for index, (current, following) in enumerate(zip(times, times[1:])):
        node = replace(initial, x_t=state, x0=None, epsilon=None, t=current,
                       metadata=dict(initial.metadata, t=current, step_index=index,
                                     x_t_hash=tensor_hash(state), state_source='student_trajectory'))
        calls = 0
        if index < cfg.get('loop_active_updates', 20) and arm != 'base':
            loop = memory_loop(student, node, rounds, feedback=arm)
            bank = loop.banks[-1]; calls += loop.gen_calls
            with gen_memory_call(runtime.decoder, node.flow['past_key_values'], bank, node.lengths, node.positions):
                conditional, velocity = runtime.velocity(state, current, node.flow)
            # The extra conditional+CFG recomputation is real and is included in cost.
            calls += 1+int(cfg['cfg_text_scale'] > 1)
        else:
            conditional, velocity = runtime.velocity(state, current, node.flow)
            calls += 1+int(cfg['cfg_text_scale'] > 1)
        state = state-(current-following)*velocity
        records.append(dict(step_index=index, t=current, next_t=following, rounds=rounds if index < cfg.get('loop_active_updates',20) and arm != 'base' else 0,
                            gen_calls_including_cfg=calls, full_gen_layer_calls=calls*len(runtime.decoder.layers),
                            sampler_updates=1, state_hash=tensor_hash(state)))
    synchronize(runtime.device)
    if output is not None:
        name = f'{row["sample_id"]}_s{seed}_R{rounds}_{arm}'
        runtime.image(state).save(output/(name+'.png'))
        write_json(output/(name+'.json'), dict(sample_id=row['sample_id'], family_id=row['family_id'],
                   seed=seed, arm=arm, rounds=rounds, policy='complete_memory_window_then_native',
                   prompt=initial.metadata['prompt'], initial_noise_hash=tensor_hash(initial.x_t),
                   source_layers=cfg.get('source_layers'), passes=rounds+1 if arm!='base' else 1,
                   feedback_updates=rounds if arm!='base' else 0,
                   seconds=time.perf_counter()-start, nodes=records,
                   gen_calls=sum(r['gen_calls_including_cfg'] for r in records)))
    return state, records


@torch.no_grad()
def refresh_trajectories(student, train_rows, output, step):
    """Optional train-only snapshot refresh; snapshots are NOT fake paired flow labels."""
    count = student.runtime.config.get('trajectory_refresh_count', 0)
    if not count: return
    if any(row['split'] != 'train' for row in train_rows):
        raise ValueError('trajectory refresh must never include held-out families')
    from .visual_record import save_visual_checkpoint
    directory = output/'train_trajectory_refresh'; directory.mkdir()
    checkpoint = directory/'source.pt'
    save_visual_checkpoint(checkpoint, student, step, 'C3-loop', None)
    digest = file_hash(checkpoint)
    for row in train_rows[:count]:
        x, nodes = complete_sampling(student, row, step, output=directory)
        torch.save(dict(x_t=x.cpu(), t=0., state_source='trajectory', prompt=student.runtime.config['neutral_prompt'],
                        checkpoint_hash=digest, noise_seed=step, step_index=len(nodes),
                        sampler_steps=student.runtime.config['sampler_steps'],
                        timestep_shift=student.runtime.config['timestep_shift']), directory/(row['sample_id']+'.pt'))
    write_json(directory/'manifest.json', dict(checkpoint_hash=digest, split='train', step=step,
               sample_ids=[r['sample_id'] for r in train_rows[:count]],
               label_status='unlabelled auxiliary snapshots; no paired flow target'))


@torch.no_grad()
def collect_label_states(runtime, rows, output):
    """Create unreviewed candidates only; never invent captions or verified facts."""
    for row in rows:
        for t, seed in states_for(row, runtime.config):
            state = runtime.training_state(row, t, seed)
            name = f'{row["sample_id"]}_t{t}_s{seed}'
            method = runtime.config.get('observation_proxy_method', 'conditional_x0_hat')
            if method == 'conditional_x0_hat':
                velocity = conditional_gen(runtime, state, capture=False).velocity
                preview = state.x_t-state.t*velocity
            elif method == 'decode_noisy_latent':
                preview = state.x_t
            else:
                raise ValueError('unknown observation proxy method')
            runtime.image(preview).save(output/(name+'.png'))
            torch.save(dict(x_t=state.x_t.cpu(), t=t, noise_seed=seed,
                            metadata=state.metadata), output/(name+'.pt'))
            append_json(output/'observation_candidates.jsonl', dict(row,
                        sample_id=name.replace('.', '_'), label_kind='observation_proxy',
                        t=t, noise_seed=seed, round=0, verified=False, review_status='pending',
                        observation_method=method+'_v1',
                        observation_preview_hash=file_hash(output/(name+'.png')),
                        observation_caption='', observation_facts=[], uncertain_facts=[],
                        observation_image_hash=state.metadata.get('image_hash'),
                        observation_state_hash=state.metadata['x_t_hash'],
                        observation_preview=name+'.png', state_tensor=name+'.pt'))
