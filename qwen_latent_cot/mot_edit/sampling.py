"""Commit clean edits to the actual latent; re-read and re-denoise each new state."""
import json
from pathlib import Path
import torch
from ..correction.sampling import times
from ..correction import distributed as parallel


@torch.no_grad()
def sample(model,prompt,seed,rounds=2):
    rt=model.runtime;flow=rt.flow(prompt);n=len(flow['packed_vae_token_indexes'])
    channels=rt.model.latent_channel*rt.model.latent_patch_size**2
    x=torch.randn((n,channels),generator=torch.Generator().manual_seed(seed)).to(rt.device)
    trace=[];grid=times(model.config)
    for index,(t,next_t) in enumerate(zip(grid,grid[1:])):
        if rounds and index in model.config['feedback_indices']:
            for r in range(rounds):
                before=x
                x,w,vc,vg,delta=model.round(x,t,flow)
                g=vg.float()-vc.float();raw=w.float()-vc.float()
                trace.append(dict(index=index,t=t,round=r+1,delta_rms=float(delta.square().mean().sqrt()),
                    raw_delta_rms=float(raw.square().mean().sqrt()),
                    delta_to_base_rms=float(delta.square().mean().sqrt()/vg.float().square().mean().sqrt().clamp_min(1e-12)),
                    committed_shift_rms=float((x-before).square().mean().sqrt()),
                    applied_guidance_coefficient=float((delta*g).sum()/g.square().sum().clamp_min(1e-12))))
                if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite committed edit')
        # Native denoising starts from the committed state. Do not apply delta twice.
        vg,_,_=rt.proposal(x,t,flow,capture=False,with_conditional=True)
        x=x-(t-next_t)*vg
    return rt.image(x),trace


@torch.no_grad()
def semantic_validation(model,output,step):
    cfg=model.config;path=Path(cfg['semantic_prompts'])
    if not path.is_absolute():path=Path(__file__).resolve().parents[2]/path
    prompts=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    root=Path(output);directory=root/f'semantic_{step:06d}';directory.mkdir(exist_ok=True)
    baseline=root/'semantic_native';baseline.mkdir(exist_ok=True);records=[]
    for i in range(parallel.rank(),len(prompts),parallel.world_size()):
        row=prompts[i];record=dict(row,index=i,seed=cfg['seed'],arms={})
        for arm,rounds in [('native',0),('one_round',1),('two_round',2)]:
            destination=(baseline if arm=='native' else directory)/f'{i:03d}_{arm}.png'
            if not destination.exists():
                image,trace=sample(model,row['prompt'],cfg['seed'],rounds)
                image.save(destination);destination.with_suffix('.json').write_text(json.dumps(trace,indent=2)+'\n')
            record['arms'][arm]=str(destination.relative_to(root))
        (directory/f'case_{i:03d}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');records.append(record)
    records=parallel.gather_samples(records)
    if parallel.rank()==0:(directory/'records.json').write_text(json.dumps(sorted(records,key=lambda r:r['index']),ensure_ascii=False,indent=2)+'\n')
    parallel.barrier()


@torch.no_grad()
def observation_validation(model,output,step):
    """Eight generated states: actual proposal image and native-head caption, no target input to reader."""
    cfg=model.config;rt=model.runtime;path=Path(cfg['semantic_prompts'])
    if not path.is_absolute():path=Path(__file__).resolve().parents[2]/path
    prompts=[json.loads(line)['prompt'] for line in path.read_text().splitlines()[:4]]
    directory=Path(output)/f'observation_{step:06d}';directory.mkdir(exist_ok=True);records=[];grid=times(cfg)
    for i in range(parallel.rank(),8,parallel.world_size()):
        prompt=prompts[i//2];seed=cfg['seed']+i%2;flow=rt.flow(prompt)
        n=len(flow['packed_vae_token_indexes']);channels=rt.model.latent_channel*rt.model.latent_patch_size**2
        x=torch.randn((n,channels),generator=torch.Generator().manual_seed(seed)).to(rt.device)
        stop=cfg['feedback_indices'][i%len(cfg['feedback_indices'])]
        for index,(t,next_t) in enumerate(zip(grid,grid[1:])):
            vg,_,_=rt.proposal(x,t,flow,capture=False,with_conditional=True)
            if index==stop:
                proposal=x.float()-t*vg.float();break
            x=x-(t-next_t)*vg
        caption=model.caption_text(proposal)
        rt.image(proposal).save(directory/f'{i:03d}_proposal.png')
        record=dict(index=i,prompt=prompt,seed=seed,t=t,caption=caption,image=f'{i:03d}_proposal.png')
        (directory/f'case_{i:03d}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');records.append(record)
    records=parallel.gather_samples(records)
    if parallel.rank()==0:(directory/'records.json').write_text(json.dumps(sorted(records,key=lambda r:r['index']),ensure_ascii=False,indent=2)+'\n')
    parallel.barrier()
