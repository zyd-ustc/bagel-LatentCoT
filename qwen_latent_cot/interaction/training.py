"""Paired flow SFT and common optimizer/checkpoint bookkeeping."""
import json
import time
from pathlib import Path
import torch
from . import checkpoint
from .data import PairSampler
from ..diagnostics import distributed as parallel
from ..diagnostics.flow_training import flow_mse
from ..diagnostics.record import FrozenGuard, write_json
from .loop import training_times, supervised_loss
from ..bagel.accelerator import synchronize, reset_peak_memory_stats, max_memory_allocated


class Trainer:
    def __init__(self, student, rows, val, output, stage, provenance, *, saved=None, wandb=None):
        self.student, self.rows, self.val = student, rows, val
        self.output, self.stage = Path(output), stage
        self.provenance, self.wandb = provenance, wandb
        self.config, self.device = student.config, student.runtime.device
        self.output.mkdir(parents=True, exist_ok=True)
        from ..diagnostics.training_logging import configure_logging
        self.logger = configure_logging(self.output)
        self.sampler = rows.sampler() if hasattr(rows, 'sampler') else PairSampler(rows, self.config['seed'])
        parallel.broadcast_parameters(student)
        self.optimizer = torch.optim.AdamW(student.optimizer_groups(), weight_decay=self.config['weight_decay'])
        self.start = checkpoint.resume(saved, student, self.optimizer, self.sampler, provenance) if saved else 0
        self.guard = FrozenGuard(student.runtime.model, student.runtime.bundle.vae, exclude_trainable=True)
        reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()
        if parallel.rank() == 0:
            write_json(self.output/'run.json', dict(stage=stage, config=self.config, provenance=provenance,
                       parent_stage=student.parent['stage'], parent_step=student.parent['step'],
                       parent_noise_curriculum=student.parent['config'].get('noise_curriculum'),
                       sampling={k: student.runtime.config[k] for k in ('sampler_steps', 'timestep_shift', 'resolution', 'cfg_text_scale', 'cfg_renorm_type', 'cfg_renorm_min')},
                       memory_times=training_times(student.runtime.config),
                       train_pairs=len(rows), val_pairs=len(val), world=parallel.world_size(),
                       trainable={k: sum(p.numel() for p in v) for k, v in student.parameter_groups().items()}))

    def update(self, step):
        parallel.sum_gradients(self.student)
        group_norms = {name: float(torch.stack([p.grad.float().square().sum() for p in values]).sum().sqrt())
                       for name, values in self.student.parameter_groups().items() if values}
        norm = torch.nn.utils.clip_grad_norm_([p for p in self.student.parameters() if p.requires_grad],
                                             self.config['gradient_clip'], error_if_nonfinite=True)
        for group in self.optimizer.param_groups:
            group['lr'] = self.config[group['name']+'_lr']*min(1., step/max(1, self.config.get('warmup_steps', 0)))
        self.optimizer.step()
        self.guard.check()
        return dict(gradient_norm=float(norm), **{f'gradient_{k}': v for k, v in group_norms.items()})

    def log(self, step, values):
        values['peak_memory_gib'] = parallel.max_statistics([max_memory_allocated(self.device)/1024**3], self.device)[0]
        record = dict(step=step, elapsed_seconds=time.perf_counter()-self.started, **values)
        if parallel.rank() == 0:
            with (self.output/'train.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            self.logger.info('step=%d %s', step, json.dumps(values))
            if self.wandb:
                self.wandb.log({'trainer/step': step, **{f'{self.stage}/{k}': v for k, v in values.items()}})

    def save(self, step):
        # Store every rank RNG: exact resume also covers future stochastic kernels.
        states = parallel.gather_samples([checkpoint.rng_state(self.device)])
        if parallel.rank() == 0:
            checkpoint.save(self.output/f'step_{step:06d}.pt', self.student, self.stage, step,
                            optimizer=self.optimizer, sampler=self.sampler,
                            provenance=self.provenance, rng_by_rank=states)
        parallel.barrier()

    @torch.no_grad()
    def validate(self, step):
        from .sampling import sample
        runtime = self.student.runtime
        records = []
        directory = self.output/f'validation_{step:06d}'
        directory.mkdir(exist_ok=True)
        for index in range(parallel.rank(), min(len(self.val), self.config['val_images']), parallel.world_size()):
            row = self.val[index]
            split = None
            if self.stage == 'stage_a':
                from .stage_a import training_state
                state, split = training_state(self.student, row, training_times(runtime.config)[-1], self.config['seed']+index, self.config['seed']+index)
            else:
                state = runtime.training_state(row, training_times(runtime.config)[-1], self.config['seed']+index)
            result = self.student(state)
            if split is not None:
                from .stage_a import correction_loss
                terms = correction_loss(result, state, self.config)
                metrics = dict(round_cos=[float(x) for x in terms.round_cos])
            else:
                terms = supervised_loss(result, state, self.config['round_loss_weights'],
                                        hard_patch_alpha=self.config['hard_patch_alpha'], keep_weight=self.config['keep_weight'])
                metrics = dict(round_weighted_flow=[float(x) for x in terms.round_weighted_flow],
                               round_keep=[float(x) for x in terms.round_keep])
            record = dict(sample_id=row['sample_id'], caption=row['caption'], image=row['image'],
                          loss=float(terms.total), round_flow=[float(x) for x in terms.round_flow],
                          base_flow=float(flow_mse(result.baseline, state)), **metrics)
            if split is not None:
                from .stage_a import semantic_loss
                record.update(partial_prompt=split.partial, removed_clauses=split.removed,
                              semantic_ce=float(semantic_loss(self.student, result.banks[-1], split)))
                from .stage_a import memory_controls
                family = lambda value: value.get('family_id', value['sample_id'])
                donor = next((v for v in self.val if family(v) != family(row)), None)
                if donor is not None:
                    donor_state, _ = training_state(self.student, donor, state.t, self.config['seed']+index+1, self.config['seed']+index+1)
                    donor_result = self.student(donor_state)
                    off, shuffled = memory_controls(self.student, state, result, donor_result.banks[-1])
                    record.update(off_flow=float(off), shuffled_flow=float(shuffled),
                                  correct_flow=float(terms.round_flow[-1]), donor_sample_id=donor['sample_id'])
                    del donor_state, donor_result
                else:
                    record['memory_controls'] = 'unavailable: no distinct validation family'
            for arm, native in (('native', True), ('loop', False)):
                image, nodes = sample(self.student, split.partial if split else row['caption'], self.config['seed']+index, native=native,
                                      reader_prompt=split.full if split and not native else None)
                image.save(directory/f'{index:03d}_{arm}.png')
                write_json(directory/f'{index:03d}_{arm}.json', nodes)
            records.append(record)
        records = parallel.gather_samples(records)
        if parallel.rank() == 0:
            write_json(directory/'records.json', records)
        parallel.barrier()


def train_sft(trainer):
    cfg, student = trainer.config, trainer.student
    batch = cfg['global_batch']
    if batch % parallel.world_size():
        raise ValueError('SFT global_batch must be divisible by world size')
    nodes = training_times(student.runtime.config)
    for step in range(trainer.start+1, cfg['steps']+1):
        started = time.perf_counter()
        trainer.optimizer.zero_grad(set_to_none=True)
        rows = trainer.sampler.take(batch)
        draws = [(trainer.sampler.rng.choice(nodes), trainer.sampler.rng.randrange(2**63-1)) for _ in rows]
        rounds = cfg['feedback_updates']
        values = [0.]*(3+3*rounds)
        for index in range(parallel.rank(), batch, parallel.world_size()):
            t, seed = draws[index]
            state = student.runtime.training_state(rows[index], t, seed)
            output = student(state)
            base = output.baseline
            terms = supervised_loss(output, state, cfg['round_loss_weights'],
                                    hard_patch_alpha=cfg['hard_patch_alpha'], keep_weight=cfg['keep_weight'])
            loss = terms.total
            (loss/batch).backward()
            values[0] += float(loss.detach())
            values[1] += float(flow_mse(base, state))
            values[2] += float((output.velocity.detach().float()-base.velocity.float()).square().mean().sqrt())
            for group_index, losses in enumerate((terms.round_flow, terms.round_weighted_flow, terms.round_keep)):
                for round_index, round_loss in enumerate(losses):
                    values[3+group_index*rounds+round_index] += float(round_loss.detach())
            del output, base, state, loss, terms, losses, round_loss
        gradients = trainer.update(step)
        values = parallel.sum_statistics(values, trainer.device)
        synchronize(trainer.device)
        trainer.log(step, dict(loss=values[0]/batch, round_flow=[v/batch for v in values[3:3+rounds]],
                              round_weighted_flow=[v/batch for v in values[3+rounds:3+2*rounds]],
                              round_keep=[v/batch for v in values[3+2*rounds:]], base_flow=values[1]/batch, delta_v_rms=values[2]/batch,
                              t_mean=sum(t for t, _ in draws)/batch, t_min=min(t for t, _ in draws),
                              t_max=max(t for t, _ in draws),
                              step_seconds=time.perf_counter()-started, **gradients))
        if step in cfg['save_steps'] or step == cfg['steps']:
            trainer.save(step)
            if cfg['val_images']:
                trainer.validate(step)


def train_stage_a(trainer):
    from .stage_a import training_state, memory_controls, losses
    cfg, student = trainer.config, trainer.student
    batch = cfg['global_batch']
    if batch % parallel.world_size(): raise ValueError('Stage A global_batch must be divisible by world size')
    nodes = training_times(student.runtime.config)
    for step in range(trainer.start+1, cfg['steps']+1):
        started = time.perf_counter()
        trainer.optimizer.zero_grad(set_to_none=True)
        rows = trainer.sampler.take(batch)
        draws = [(trainer.sampler.rng.choice(nodes), trainer.sampler.rng.randrange(2**63-1),
                  trainer.sampler.rng.randrange(2**63-1)) for _ in rows]
        # Every rank draws the same rows/donors before rank-strided computation.
        donors = list(rows)
        if cfg['memory_contrast_weight']:
            donors += trainer.sampler.take(max(2, batch))
            family = lambda row: row.get('family_id', row['sample_id'])
            donor_rows = [next((d for d in donors if family(d) != family(row)), None) for row in rows]
            if any(d is None for d in donor_rows):
                raise ValueError('Shuffled Memory requires at least two distinct sampled families')
        names = ('loss','base_flow','correct_flow','semantic_ce','memory_contrast','off_flow','shuffled_flow',
                 'delta_v_rms','target_residual_rms','removed_word_fraction','masked_fraction','cos_loss')
        values = [0.]*len(names)
        round_values = [0.]*cfg['feedback_updates']
        for index in range(parallel.rank(), batch, parallel.world_size()):
            t, seed, split_seed = draws[index]
            state, split = training_state(student, rows[index], t, seed, split_seed)
            donor_bank = None
            if cfg['memory_contrast_weight']:
                # No donor gradients. Swap only its bank into the fixed recipient.
                with torch.no_grad():
                    donor_state, _ = training_state(student, donor_rows[index], t, seed+1, split_seed+1)
                    donor_output = student(donor_state)
                    donor_bank = donor_output.banks[-1]
                    del donor_state, donor_output
            output = student(state)
            off, shuffled = (memory_controls(student, state, output, donor_bank) if donor_bank is not None
                             else (None, None))
            loss, flow, sem, contrast = losses(student, state, output, split, shuffled_error=shuffled)
            (loss/batch).backward()
            with torch.no_grad():
                base = output.baseline.velocity.float()
                measurements = (loss, flow_mse(output.baseline,state), flow.round_flow[-1], sem, contrast,
                                off if off is not None else 0., shuffled if shuffled is not None else 0.,
                                (output.velocity.float()-base).square().mean().sqrt(),
                                (state.correction_target-base).square().mean().sqrt(),
                                len(split.removed.split())/max(1,len(split.full.split())), bool(split.removed),
                                flow.round_cos[-1])
                for j,value in enumerate(measurements): values[j] += float(value)
                for j,value in enumerate(flow.round_flow): round_values[j] += float(value)
            if parallel.rank() == 0 and index == 0 and step == trainer.start+1:
                write_json(trainer.output/f'prompt_split_step_{step:06d}.json',
                           dict(sample_id=rows[index]['sample_id'], full=split.full, partial=split.partial,
                                removed=split.removed, reason=split.reason, dropped_units=split.dropped))
            del output, state, loss, flow, sem, contrast, measurements, donor_bank
        gradients = trainer.update(step)
        values = parallel.sum_statistics(values, trainer.device)
        synchronize(trainer.device)
        round_values = parallel.sum_statistics(round_values, trainer.device)
        record = {k:v/batch for k,v in zip(names,values)}
        record['round_flow'] = [v/batch for v in round_values]
        record['applied_delta_to_target_rms'] = record['delta_v_rms']/max(record['target_residual_rms'],1e-8)
        record.update(off_minus_correct=record['off_flow']-record['correct_flow'] if cfg['memory_contrast_weight'] else None,
                      shuffled_minus_correct=record['shuffled_flow']-record['correct_flow'] if cfg['memory_contrast_weight'] else None,
                      velocity_scale=student.bridge.vel_scale.detach().float().cpu().tolist(),
                      memory_gates=(student.bridge.gate_scale*student.bridge.memory_gates.detach()).tanh().float().cpu().tolist(),
                      reentry_gate=float((student.bridge.gate_scale*student.bridge.strength.detach()).tanh()),
                      t_mean=sum(d[0] for d in draws)/batch, step_seconds=time.perf_counter()-started, **gradients)
        trainer.log(step,record)
        if step in cfg['save_steps'] or step == cfg['steps']:
            trainer.save(step)
            if cfg['val_images']: trainer.validate(step)
