"""Time-conditioned UND queries. GEN keys, RoPE and all native weights stay native."""
import math
import torch
from torch import nn
from .memory import MemoryWriter
from .numerics import autocast_for


class VisualQueryResidual(nn.Module):
    def __init__(self, width, rank, time_width=8):
        super().__init__()
        if time_width % 2:
            raise ValueError('time features require an even width')
        self.register_buffer('frequencies', 2.**torch.arange(time_width//2), persistent=True)
        self.down = nn.Linear(width, rank, bias=False)
        self.time = nn.Linear(time_width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden, t):
        # No zero gate: up learns on step one, down/time can learn from step two.
        phase = self.frequencies * (math.pi*float(t))
        features = torch.cat((phase.sin(), phase.cos())).to(self.down.weight.dtype)
        value = self.down(hidden.to(self.down.weight.dtype))+self.time(features)
        return self.up(value).to(hidden.dtype)


class VisualReadAdapter(nn.Module):
    architecture = 'und_query_time_low_rank_v1'

    def __init__(self, decoder, rank=32):
        super().__init__()
        self.layers = nn.ModuleList([VisualQueryResidual(decoder.embed_tokens.weight.shape[1], rank)
                                    for _ in decoder.layers])
        self.to(decoder.embed_tokens.weight.device)

    def forward(self, layer, hidden, t):
        return self.layers[layer](hidden, t)


class VisualStudent(nn.Module):
    """Register only newly trained modules; independently freeze the text teacher."""
    def __init__(self, runtime):
        super().__init__()
        if runtime.writer.topology != 'C':
            raise ValueError('visual/loop training requires continuous C writer')
        object.__setattr__(self, 'runtime', runtime)
        self.writer = runtime.writer
        self.visual = VisualReadAdapter(runtime.decoder, runtime.config.get('visual_rank', self.writer.rank))
        self.writer.requires_grad_(runtime.config.get('train_writer', False))
        teacher = MemoryWriter(runtime.decoder, 'C', self.writer.slots, self.writer.rank)
        teacher.load_state_dict(self.writer.state_dict(), strict=True)
        teacher.eval().requires_grad_(False)
        # Not part of optimizer, student state or broadcast parameter collection.
        object.__setattr__(self, 'text_teacher', teacher)

    def write(self, source, t, *, source_off=False, visual=True, detach_source=True, source_layers=None):
        runtime = self.runtime
        positions = torch.arange(runtime.config['writer_position_start'],
                                 runtime.config['writer_position_start']+self.writer.slots,
                                 device=runtime.device)
        with autocast_for(runtime.device):
            return self.writer(runtime.seeds, source, positions, source_off,
                               source_kind='visual' if visual else 'text', t=t,
                               visual_adapter=self.visual if visual else None,
                               detach_source=detach_source, source_layers=source_layers)

    @torch.no_grad()
    def teacher_bank(self, caption):
        runtime = self.runtime
        source = runtime.text(caption)[0]
        positions = torch.arange(runtime.config['writer_position_start'],
                                 runtime.config['writer_position_start']+self.writer.slots,
                                 device=runtime.device)
        with autocast_for(runtime.device):
            return self.text_teacher(runtime.seeds, source, positions)

    def parameter_groups(self):
        return {name: [p for p in module.parameters() if p.requires_grad]
                for name, module in (('visual', self.visual), ('writer', self.writer))}


def validate_config(runtime, stage):
    cfg = runtime.config
    if stage not in ('C2-observe', 'C2-joint', 'C3-loop'):
        raise ValueError('select C2-observe/C2-joint/C3-loop')
    if cfg.get('visual_architecture') != VisualReadAdapter.architecture:
        raise ValueError('unknown visual architecture')
    if cfg.get('label_interface_version') != 'visual_labels_v1':
        raise ValueError('unknown visual label instruction interface')
    if set(cfg.get('label_instructions', {})) != {'target_clean', 'observation_proxy', 'target_observation_gap'}:
        raise ValueError('declare all fixed label instructions')
    if cfg.get('response_teacher') != 'frozen_text_memory16' or cfg['response_position_reserve'] != cfg['slots']:
        raise ValueError('v1 requires independent C1 text teacher in deployment 16-slot layout')
    if cfg['slots'] != 16 or runtime.writer.topology != 'C':
        raise ValueError('v1 uses continuous writer with 16 slots')
    if cfg.get('visual_source_policy') not in ('full_query', 'image_only'):
        raise ValueError('unknown visual source policy')
    if not cfg['train_t'] or any(not 0 < t < 1 for t in cfg['train_t']) or 1. not in cfg['eval_t']:
        raise ValueError('exclude pure noise from training and include it in separate evaluation')
    if not 0 <= cfg['replay_fraction'] < 1 or int(cfg['effective_batch']*cfg['replay_fraction']) != cfg['effective_batch']*cfg['replay_fraction']:
        raise ValueError('replay fraction must select an integer count in the GLOBAL batch')
    if set(cfg['loss_weights']) != {'sem', 'read', 'gen', 'keep'} or any(w < 0 for w in cfg['loss_weights'].values()):
        raise ValueError('declare separate nonnegative sem/read/gen/keep coefficients')
    if stage == 'C2-observe' and (cfg['loss_weights']['read'] or cfg['loss_weights']['gen']):
        raise ValueError('C2-observe is semantic-only; select C2-joint for generated responses')
    if cfg['effective_batch'] <= 0 or cfg['steps'] <= 0 or cfg['eval_every'] <= 0:
        raise ValueError('training budgets must be positive')
    layers = cfg['response_layers']
    if not layers or len(set(layers)) != len(layers) or any(not 0 <= i < len(runtime.decoder.layers) for i in layers):
        raise ValueError('invalid response readout layers')
    if cfg['response_energy_floor'] <= 0:
        raise ValueError('response energy floor must be positive')


def train(student, rows, text_rows, stage, output, parent, *, resume=None, skip_validation=False):
    """Joint 16-rank updates with deterministic global scheduling and text replay."""
    import random
    import time
    from . import distributed as parallel
    from .record import FrozenGuard, append_json, write_json
    from .visual_record import save_visual_checkpoint, load_visual_checkpoint
    from .visual_objectives import semantic_loss, state_probe, bank_response, generation_caption, text_keep
    from .flow_training import conditional_gen, flow_mse, ExecutionCounter
    from .loop_training import memory_loop
    from .visual_evaluation import evaluate, evaluate_text_keep, refresh_trajectories
    from ..bagel.accelerator import synchronize, reset_peak_memory_stats, max_memory_allocated
    runtime = student.runtime; cfg = runtime.config
    validate_config(runtime, stage)
    pool = [row for row in rows if row['split'] == 'train']
    val = [row for row in rows if row['split'] == 'val']
    validation_limit = cfg.get('validation_max_families', 0)
    if validation_limit:
        import hashlib
        families = sorted({row['family_id'] for row in val}, key=lambda family:
                          hashlib.sha256(f"{cfg['seed']}:{family}".encode()).hexdigest())
        selected_families = set(families[:validation_limit])
        val = [row for row in val if row['family_id'] in selected_families]
    replay = [row for row in text_rows if row['split'] == 'train']
    replay_val = [row for row in text_rows if row['split'] == 'val']
    batch = cfg['effective_batch']; world = parallel.world_size(); worker = parallel.rank()
    replay_count = int(batch*cfg['replay_fraction'])
    if not pool or (not val and not skip_validation) or batch % world or (replay_count and (not replay or not replay_val)):
        raise ValueError('require train/val, text replay train/val and globally divisible batch')
    if cfg['loss_weights']['gen'] and any(row.get('state') for row in pool):
        raise ValueError('trajectory semantic training has no paired flow target; gen weight must be zero')
    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    rng = random.Random(cfg['seed']); order = list(range(len(pool))); rng.shuffle(order); cursor = 0
    start_step = 0; best = None
    if resume:
        saved = load_visual_checkpoint(resume, student, optimizer=optimizer, resume=True, expected_stage=stage)
        state = saved['schedule']; rng.setstate(state['rng']); order = state['order']; cursor = state['cursor']
        if sorted(order) != list(range(len(pool))):
            raise ValueError('resume dataset sampling pool changed')
        start_step, best = saved['step'], saved['best']
    parallel.broadcast_parameters(student)
    guard = FrozenGuard(runtime.model, runtime.bundle.vae, student.text_teacher,
                        *([student.writer] if not cfg['train_writer'] else []))
    teacher_guard = FrozenGuard(student.text_teacher)
    reset_peak_memory_stats(runtime.device); start = time.perf_counter()
    if worker == 0:
        write_json(output/'validation_selection.json', dict(
                   maximum_families=validation_limit, seed=cfg['seed'],
                   policy='stable SHA256 order by seed/family; source val split only',
                   family_ids=sorted({row['family_id'] for row in val}),
                   sample_ids=[row['sample_id'] for row in val]))
        save_visual_checkpoint(output/'init.pt', student, start_step, stage, parent)
        write_json(output/'freeze_gradient_audit.json', dict(status='started',
                   model_files=runtime.native_files, guard='native/teacher tensor identity and version',
                   trainable={name: p.numel() for name, p in student.named_parameters() if p.requires_grad},
                   teacher_hash=student.text_teacher_hash, writer_trainable=cfg['train_writer'],
                   text_replay_provides_gradient=cfg['train_writer'], cross_round_gradients=cfg['cross_round_gradients']))
    parallel.barrier()
    counter = ExecutionCounter(runtime)
    for step in range(start_step+1, cfg['steps']+1):
        student.train(); optimizer.zero_grad(set_to_none=True)
        sums = [0.]*15; samples = []
        # Spread replay ranks across steps; each rank advances the same RNG/order.
        replay_indexes = {(step+i) % batch for i in range(replay_count)}
        for micro in range(batch):
            if cursor == len(order):
                rng.shuffle(order); cursor = 0
            row = pool[order[cursor]]; cursor += 1
            t = row.get('t', rng.choice(cfg['train_t']))
            seed = row.get('noise_seed', rng.randrange(2**63-1))
            text_row = replay[rng.randrange(len(replay))] if replay else None
            if micro % world != worker:
                continue
            before_cost = list(counter.counts)
            state = runtime.training_state(row, t, seed)
            is_replay = micro in replay_indexes
            components = {name: torch.zeros((), device=runtime.device) for name in ('sem', 'read', 'gen', 'keep')}
            gen_calls = 0; read_metrics = []; keep_metrics = None
            if is_replay:
                probe = state_probe(runtime, state); gen_calls += 1
                keep, keep_metrics = text_keep(student, text_row, probe)
                components['keep'] = keep
                label_kind = 'text_replay'; rounds = 0
            else:
                rounds = 2 if stage == 'C3-loop' else 1
                if stage == 'C2-observe' or (stage == 'C2-joint' and not cfg['loss_weights']['gen']):
                    with torch.no_grad(): source = conditional_gen(runtime, state).source
                    banks = [student.write(source, t)]; exits = []; gen_calls += 1
                else:
                    loop = memory_loop(student, state, rounds,
                                       checkpoint_activations=cfg.get('activation_checkpointing', True))
                    banks, exits = loop.banks, loop.exits; gen_calls += loop.gen_calls
                components['sem'] = torch.stack([semantic_loss(runtime, row, bank) for bank in banks]).mean()
                caption = generation_caption(row, cfg)
                if cfg['loss_weights']['read'] and caption is not None:
                    probe = state_probe(runtime, state); gen_calls += 1
                    teacher_bank = student.teacher_bank(caption)
                    reads = [bank_response(runtime, bank, teacher_bank, probe) for bank in banks]
                    components['read'] = torch.stack([item[0] for item in reads]).mean()
                    read_metrics = [dict(round=i+1, layers=item[1]) for i, item in enumerate(reads)]
                if cfg['loss_weights']['gen']:
                    components['gen'] = torch.stack([flow_mse(exit, state) for exit in exits]).mean()
                label_kind = row['label_kind']
            loss = sum(cfg['loss_weights'][name]*value for name, value in components.items())
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite visual objective')
            objective_gradients = {}
            if step in cfg.get('objective_gradient_audit_steps', [1]):
                for name, value in components.items():
                    if value.requires_grad and cfg['loss_weights'][name]:
                        gradients = torch.autograd.grad(value, trainable, retain_graph=True, allow_unused=True)
                        if any(g is not None and not torch.isfinite(g).all() for g in gradients):
                            raise FloatingPointError('non-finite objective gradient: '+name)
                        objective_gradients[name] = float(sum(g.float().square().sum() for g in gradients if g is not None).sqrt())
                    else:
                        objective_gradients[name] = None
            if loss.requires_grad:
                (loss/batch).backward()
            elif not is_replay:
                raise RuntimeError('visual loss has no path to trainable parameters')
            for index, name in enumerate(('sem', 'read', 'gen', 'keep')):
                sums[2*index] += float(components[name].detach())
                sums[2*index+1] += int((name == 'keep') == is_replay)
            cost = [a-b for a,b in zip(counter.counts, before_cost)]
            for j,value in enumerate(cost): sums[12+j] += value
            sums[8] += float(loss.detach()); sums[9] += 1; sums[10] += gen_calls; sums[11] += int(not is_replay)*rounds
            samples.append(dict(sample_id=(text_row if is_replay else row)['sample_id'],
                                state_sample_id=row['sample_id'], global_micro=micro, rank=worker,
                                t=t, noise_seed=seed, label_kind=label_kind, rounds=rounds,
                                gen_conditional_calls=gen_calls, state=state.metadata,
                                read_metrics=read_metrics, keep_metrics=keep_metrics,
                                actual_execution_entries=dict(gen=cost[0], gen_layers=cost[1], und_layers=cost[2]),
                                objective_unweighted_gradient_norms=objective_gradients,
                                full_GEN_input_gradient=bool(cfg['loss_weights']['gen']) and not is_replay))
        missing = [name for name, p in student.named_parameters() if p.requires_grad and p.grad is None]
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainable):
            raise FloatingPointError('non-finite trainable-module gradient')
        parallel.sum_gradients(student, allow_unused=True)
        group_norms = {name: float(torch.stack([p.grad.float().square().sum() for p in group]).sum().sqrt())
                       for name, group in student.parameter_groups().items() if group}
        if group_norms.get('visual', 0.) == 0:
            raise RuntimeError('global visual adapter gradient is zero')
        norm = torch.nn.utils.clip_grad_norm_(trainable, cfg['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); guard.check(); teacher_guard.check(); synchronize(runtime.device)
        totals = parallel.sum_statistics(sums, runtime.device)
        global_samples = sorted(parallel.gather_samples(samples), key=lambda item: item['global_micro'])
        resources = parallel.gather_samples([dict(rank=worker, peak_memory_bytes=max_memory_allocated(runtime.device))])
        if worker == 0:
            append_json(output/'training.jsonl', dict(step=step, stage=stage,
                loss=totals[8]/totals[9], losses={name: totals[2*i]/max(totals[2*i+1], 1)
                for i, name in enumerate(('sem', 'read', 'gen', 'keep'))},
                gradient_norm=float(norm), group_gradient_norms=group_norms,
                rank0_unused_parameter_names=missing, world_size=world, global_effective_batch=batch,
                gen_conditional_calls=int(totals[10]), actual_gen_entries_including_recomputation=int(totals[12]),
                full_gen_layer_calls=int(totals[13]), und_layer_calls=int(totals[14]),
                cost_basis='actual layer entries incl checkpoint/audit recomputation; not FLOPs',
                visual_writer_layer_calls=int(totals[11])*len(runtime.decoder.layers),
                elapsed_seconds=time.perf_counter()-start, resources=resources,
                sampled_states=global_samples, frozen_guard='passed'))
            if step == start_step+1 or step % 10 == 0:
                print(f'Step {step}: sem={totals[0]/max(totals[1],1):.6f}, read={totals[2]/max(totals[3],1):.6f}, gen={totals[4]/max(totals[5],1):.6f}', flush=True)
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            parallel.barrier()
            schedule = dict(rng=rng.getstate(), order=order, cursor=cursor)
            if not skip_validation:
                directory = output/f'validation_{step:06d}'
                local_directory = directory/f'rank_{worker:02d}'
                local_directory.mkdir(parents=True, exist_ok=True)
                families = sorted({row['family_id'] for row in val})
                assigned = set(families[worker::world])
                local_val = [row for row in val if row['family_id'] in assigned]
                records = evaluate(student, local_val, local_directory,
                                   rounds=2 if stage == 'C3-loop' else 1, controls=False)
                primary = [row for row in records if row['arm'] == 'R1' and row['t'] != 1]
                nll_sum, count = parallel.sum_statistics([sum(row['caption']['nll'] for row in primary), len(primary)], runtime.device)
                if count == 0: raise ValueError('validation requires non-pure-noise states')
                score = nll_sum/count
                brief = parallel.gather_samples([dict(sample_id=row['sample_id'], family_id=row['family_id'],
                          scene_id=row['scene_id'], arm=row['arm'], t=row['t'], noise_seed=row['noise_seed'],
                          label_kind=row['label_kind'], confirmed_label=row['confirmed_label'],
                          nll=row['caption']['nll'], fact_accuracy=row['caption']['fact_accuracy'],
                          unknown_assertion_rate=row['caption']['unknown_assertion_rate'], rank=worker)
                          for row in records])
                if replay_val:
                    evaluate_text_keep(student, replay_val[worker::world], val, local_directory)
                parallel.barrier()
                if worker == 0:
                    write_json(directory/'metrics_index.json', brief)
                    write_json(directory/'selection.json', dict(selection_nll=score, states=int(count),
                               scope='R1 semantic validation excludes t1; no acceptance inference'))
                    if best is None or score < best:
                        best = score
                        save_visual_checkpoint(output/'best.pt', student, step, stage, parent, optimizer, schedule, best)
                    refresh_trajectories(student, pool, directory, step)
            if worker == 0:
                save_visual_checkpoint(output/'last.pt', student, step, stage, parent, optimizer, schedule, best)
                write_json(output/'freeze_gradient_audit.json', dict(status='passed', step=step,
                           model_files=runtime.native_files, text_teacher_hash=student.text_teacher_hash,
                           guard='native/teacher tensor identity and inplace version; native gradients absent',
                           group_gradient_norms=group_norms, text_replay_provides_gradient=cfg['train_writer']))
            parallel.barrier()
    counter.close()
    if worker == 0:
        write_json(output/'training_summary.json', dict(stage=stage, completed_steps=cfg['steps'],
                   resumed_from_step=start_step, best_validation_nll=best, skip_validation=skip_validation,
                   world_size=world, global_effective_batch=batch, text_teacher_hash=student.text_teacher_hash,
                   quality_status='training completed; semantic and final quality claims require paired evaluation'))
