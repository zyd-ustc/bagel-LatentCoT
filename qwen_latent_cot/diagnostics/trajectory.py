"""Capture actual native sampler nodes; observation labels remain unverified."""
import torch
from .data import controlled_noise, tensor_hash
from .schedule import native_times, nearest_node
from .record import append_json, write_json, FrozenGuard


@torch.no_grad()
def collect(runtime, rows, output):
    guard = FrozenGuard(runtime.model, runtime.bundle.vae)
    cfg = runtime.config
    times = native_times(cfg)
    requested = [t for t in cfg['eval_t'] if t < 1]
    selected = {nearest_node(cfg, t)[0] for t in requested}
    records = []
    for row in rows:
        for seed in cfg['noise_seeds']:
            flow, _, _ = runtime.flow_inputs(row['prompt'])
            num = len(flow['packed_vae_token_indexes'])
            width = runtime.model.latent_channel*runtime.model.latent_patch_size**2
            x = controlled_noise(torch.zeros(num, width, device=runtime.device), 1, seed)
            for step, (t, following) in enumerate(zip(times, times[1:])):
                actual = float(t)
                conditional, final = runtime.velocity(x, actual, flow)
                if step in selected:
                    name = f'{row["sample_id"]}_seed{seed}_step{step:03d}'
                    saved = dict(x_t=x.cpu(), t=actual, step_index=step, noise_seed=seed,
                                 prompt=row['prompt'], state_source='trajectory', x_t_hash=tensor_hash(x),
                                 sampler_steps=cfg['sampler_steps'], timestep_shift=cfg['timestep_shift'])
                    torch.save(saved, output/(name+'.pt'))
                    runtime.image(x-actual*conditional).save(output/(name+'_conditional_x0_hat.png'))
                    runtime.image(x-actual*final).save(output/(name+'_cfg_x0_hat.png'))
                    record = dict(sample_id=name, scene_id=row['scene_id'], family_id=row.get('family_id') or row.get('pair_id') or row['scene_id'],
                                  split=row['split'], prompt=row['prompt'], state=name+'.pt',
                                  actual_t=actual, step_index=step, noise_seed=seed, x_t_hash=tensor_hash(x),
                                  label_type='observed', caption='', facts=[], verified=False,
                                  annotation_image=name+'_conditional_x0_hat.png')
                    append_json(output/'observation_candidates.jsonl', record); records.append(record)
                x = x-(actual-float(following))*final
            runtime.image(x).save(output/f'{row["sample_id"]}_seed{seed}_final.png')
            guard.check()
    write_json(output/'trajectory_summary.json', dict(nodes=len(records), requested_t=requested,
                actual_t=sorted({r['actual_t'] for r in records}), label_status='requires verified y_obs/unknown annotation',
                frozen_guard='passed'))
