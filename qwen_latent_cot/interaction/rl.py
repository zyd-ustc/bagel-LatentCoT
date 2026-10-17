"""One stochastic transition per native rollout; one group-policy update."""
import hashlib
import json
from pathlib import Path
import time
import torch
from .sde import transition, advantages, policy_loss
from .rewards import JudgeQueue, load_questions
from ..diagnostics import distributed as parallel
from ..diagnostics.flow_training import TrainingState
from ..diagnostics.feedback_source import prompt_source
from ..diagnostics.schedule import native_times
from ..bagel.accelerator import synchronize


def trajectory_state(runtime, x, t, caption, context=None):
    flow, lengths, positions = context or runtime.flow_inputs(caption)
    return TrainingState(x, None, None, t, flow, lengths, positions,
                         dict(prompt=caption, state_source='rl_rollout', flow_supervision=False),
                         prompt_source(flow, lengths, positions))


@torch.no_grad()
def rollout(student, caption, initial_seed, transition_seed, selected):
    runtime, cfg = student.runtime, student.config
    times = native_times(runtime.config)
    if selected not in range(1, cfg['early_steps']):
        raise ValueError('Select a stochastic step in [1, early_steps)')
    context = runtime.flow_inputs(caption)
    size = runtime.config['resolution']//runtime.model.latent_downsample
    width = runtime.model.latent_channel*runtime.model.latent_patch_size**2
    x = torch.randn(size*size, width, generator=torch.Generator().manual_seed(initial_seed)).to(runtime.device)
    recorded = None
    for index, (t, s) in enumerate(zip(times, times[1:])):
        if index < cfg['early_steps']:
            state = trajectory_state(runtime, x, t, caption, context)
            output, _, _ = student(state, guidance=True)
            velocity = output.velocity
        else:
            velocity = runtime.velocity(x, t, context[0])[1]
        if index == selected:
            noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(transition_seed)).to(x.device)
            result = transition(x, velocity, t, s, cfg['noise_level'], noise=noise)
            recorded = dict(x_t=x.detach().cpu(), x_s=result.sample.detach().cpu(),
                            t=t, s=s, old_score=result.score.detach().cpu(), step_index=index,
                            initial_seed=initial_seed, transition_seed=transition_seed)
            x = result.sample
        else:
            x = x+(s-t)*velocity
    return runtime.image(x), recorded


def replay(student, row, *, mode='policy'):
    state = trajectory_state(student.runtime, row['x_t'].to(student.runtime.device), row['t'], row['caption'])
    output, _, _ = student(state, guidance=True, mode=mode)
    return transition(state.x_t, output.velocity, row['t'], row['s'], student.config['noise_level'],
                      next_sample=row['x_s'].to(state.x_t.device))


@torch.no_grad()
def update_metrics(student, records):
    """Measure the actual update: pre-update ratios are one by construction."""
    values = [0., 0., 0.]
    for row in records:
        current = replay(student, row)
        ratio = float((current.score-row['old_score'].to(current.score.device)).exp())
        values[0] += ratio
        values[1] += abs(ratio-1) > student.config['clip']
        values[2] += float((current.mean-row['reference_mean'].to(current.mean.device)).square().mean()/(2*current.variance))
    return values


def train_rl(trainer, questions_path, queue_path):
    cfg, student = trainer.config, trainer.student
    if cfg['candidates'] != 8:
        raise ValueError('First Flow-GRPO protocol requires eight candidates')
    questions = load_questions(questions_path, trainer.rows)
    # A reward cache can cover a reviewed subset of the training pool.
    by_caption = {}
    for row in trainer.rows:
        if row['caption'] in questions:
            by_caption.setdefault(row['caption'], row)
    trainer.sampler.rows = list(by_caption.values())
    if parallel.rank() == 0:
        (trainer.output/'reward_data.json').write_text(json.dumps(dict(
            questions_path=str(Path(questions_path).resolve()), approved_captions=len(by_caption),
            questions_sha256=trainer.provenance['reward_questions_sha256']), indent=2))
    if trainer.start and any(i >= len(trainer.sampler.rows) for i in trainer.sampler.order):
        raise ValueError('Resumed RL sampler does not match the approved reward subset')
    queue = JudgeQueue(queue_path, cfg['judge_timeout_seconds'])
    for step in range(trainer.start+1, cfg['steps']+1):
        started = time.perf_counter()
        prompts = trainer.sampler.take(cfg['prompts_per_batch'])
        batch = len(prompts)*8
        jobs = []
        for group, prompt in enumerate(prompts):
            for candidate in range(8):
                jobs.append(dict(group=group, candidate=candidate, caption=prompt['caption'],
                                 initial_seed=trainer.sampler.rng.randrange(2**63-1),
                                 transition_seed=trainer.sampler.rng.randrange(2**63-1),
                                 selected=trainer.sampler.rng.randrange(1, cfg['early_steps'])))
        local, requests = [], []
        directory = trainer.output/f'rollout_{step:06d}'
        directory.mkdir(exist_ok=True)
        for job in jobs[parallel.rank()::parallel.world_size()]:
            key = f'b{step:06d}_g{job["group"]:03d}_c{job["candidate"]}'
            image, record = rollout(student, job['caption'], job['initial_seed'], job['transition_seed'], job['selected'])
            record.update(job, old_policy_step=step-1,
                          condition_id=hashlib.sha256((job['caption']+trainer.provenance['parent_sha256']).encode()).hexdigest())
            torch.save(record, directory/f'{key}.pt')
            requests.append(queue.submit(key, image, questions[job['caption']]))
            local.append(record)
        scored = queue.collect(requests)
        for record, path in zip(local, requests):
            record.update(scored[path])
        summary = parallel.gather_samples([{k: r[k] for k in ('group', 'candidate', 'reward')} for r in local])
        normalized, flat = advantages(summary)
        trainer.optimizer.zero_grad(set_to_none=True)
        totals = [0., 0., 0., 0., 0.]
        for record in local:
            with torch.no_grad():
                reference = replay(student, record, mode='reference')
                record['reference_mean'] = reference.mean.cpu()
            current = replay(student, record)
            advantage = normalized[(record['group'], record['candidate'])]
            loss, ratio, kl = policy_loss(current, record['old_score'].to(trainer.device), advantage,
                                          reference, cfg['clip'], cfg['beta'])
            if abs(float(ratio.detach())-1.) > .01:
                raise RuntimeError('Unchanged rollout/replay ratio differs from 1 by more than 0.01')
            (loss/batch).backward()
            totals[0] += float(loss.detach())
            totals[1] += float(ratio.detach())
            totals[2] += float(kl.detach())
            totals[3] += float(abs(float(ratio.detach())-1) > cfg['clip'])
            totals[4] += record['reward']
        gradients = trainer.update(step)
        post = parallel.sum_statistics(update_metrics(student, local), trainer.device)
        totals = parallel.sum_statistics(totals, trainer.device)
        synchronize(trainer.device)
        trainer.log(step, dict(loss=totals[0]/batch, ratio=totals[1]/batch, kl=totals[2]/batch,
                              clip_fraction=totals[3]/batch, reward=totals[4]/batch,
                              post_update_ratio=post[0]/batch, post_update_clip_fraction=post[1]/batch,
                              post_update_kl=post[2]/batch,
                              zero_variance_groups=flat, images=batch, step_seconds=time.perf_counter()-started, **gradients))
        if parallel.rank() == 0:
            (directory/'rewards.json').write_text(json.dumps(summary, indent=2))
        if step in cfg['save_steps'] or step == cfg['steps']:
            trainer.save(step)
            if cfg['val_images']:
                trainer.validate(step)
