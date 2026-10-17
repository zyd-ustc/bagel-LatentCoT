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
from ..diagnostics.schedule import native_times
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
        self.sampler = PairSampler(rows, self.config['seed'])
        parallel.broadcast_parameters(student.adapters)
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
                       memory_times=native_times(student.runtime.config)[:self.config['early_steps']],
                       train_pairs=len(rows), val_pairs=len(val), world=parallel.world_size(),
                       trainable={k: sum(p.numel() for p in v) for k, v in student.parameter_groups().items()}))

    def update(self, step):
        parallel.sum_gradients(self.student.adapters)
        group_norms = {name: float(torch.stack([p.grad.float().square().sum() for p in values]).sum().sqrt())
                       for name, values in self.student.parameter_groups().items() if values}
        norm = torch.nn.utils.clip_grad_norm_([p for p in self.student.adapters.parameters() if p.requires_grad],
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
        from ..diagnostics.shallow_sampling import sample
        runtime = self.student.runtime
        records = []
        directory = self.output/f'validation_{step:06d}'
        directory.mkdir(exist_ok=True)
        for index in range(parallel.rank(), min(len(self.val), self.config['val_images']), parallel.world_size()):
            row = self.val[index]
            state = runtime.training_state(row, native_times(runtime.config)[10], self.config['seed']+index)
            current, base, _ = self.student(state)
            record = dict(sample_id=row['sample_id'], caption=row['caption'], image=row['image'],
                          flow=float(flow_mse(current, state)), base_flow=float(flow_mse(base, state)))
            for arm, enabled, passes, mode in (
                ('parent', False, 2, 'correct'), ('correct', True, 2, 'correct'),
                ('masked', True, 2, 'masked'), ('swap', True, 2, 'swap')):
                self.student.adapters_enabled = enabled
                try:
                    image, _ = sample(self.student, row['caption'], self.config['seed']+index, passes,
                                      self.config['early_steps'], memory_mode=mode,
                                      donor_seed=self.config['seed']+index+1 if mode == 'swap' else None)
                    image.save(directory/f'{index:03d}_{arm}.png')
                finally:
                    self.student.adapters_enabled = True
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
    nodes = native_times(student.runtime.config)[:cfg['early_steps']]
    for step in range(trainer.start+1, cfg['steps']+1):
        started = time.perf_counter()
        trainer.optimizer.zero_grad(set_to_none=True)
        rows = trainer.sampler.take(batch)
        draws = [(trainer.sampler.rng.choice(nodes), trainer.sampler.rng.randrange(2**63-1)) for _ in rows]
        values = [0., 0., 0.]
        for index in range(parallel.rank(), batch, parallel.world_size()):
            t, seed = draws[index]
            state = student.runtime.training_state(rows[index], t, seed)
            output, base, _ = student(state)
            loss = flow_mse(output, state)
            (loss/batch).backward()
            values[0] += float(loss.detach())
            values[1] += float(flow_mse(base, state))
            values[2] += float((output.velocity.detach().float()-base.velocity.float()).square().mean().sqrt())
            del output, base, state, loss
        gradients = trainer.update(step)
        values = parallel.sum_statistics(values, trainer.device)
        synchronize(trainer.device)
        trainer.log(step, dict(flow=values[0]/batch, base_flow=values[1]/batch, delta_v_rms=values[2]/batch,
                              t_mean=sum(t for t, _ in draws)/batch, t_min=min(t for t, _ in draws),
                              t_max=max(t for t, _ in draws),
                              step_seconds=time.perf_counter()-started, **gradients))
        if step in cfg['save_steps'] or step == cfg['steps']:
            trainer.save(step)
            if cfg['val_images']:
                trainer.validate(step)
