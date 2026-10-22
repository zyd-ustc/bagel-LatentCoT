"""One authorized8000-step budget: observation1000, actual two-round edits7000."""
import json
from pathlib import Path
import time
import zlib
import torch
from torch.nn import functional as F
from .model import project_safe,apply_edit
from . import checkpoint
from .data import Sampler
from .sampling import semantic_validation,observation_validation
from ..correction import distributed as parallel
from ..correction.training import FrozenGuard
from ..correction.sampling import times
from ..bagel.accelerator import synchronize,max_memory_allocated,reset_peak_memory_stats


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def optimizer_for(model):
    return torch.optim.AdamW(model.trainable().values(),lr=model.config['lr'],betas=(.9,.999),weight_decay=model.config['weight_decay'])


def train(model,corpus,output,provenance,saved=None):
    cfg=model.config;rt=model.runtime;root=Path(output);world=parallel.world_size();batch=cfg['global_batch']
    if batch%world:raise ValueError('Global batch must divide world size')
    if parallel.rank()==0:root.mkdir(parents=True,exist_ok=bool(saved))
    parallel.barrier();sampler=Sampler(corpus.ids('train'),cfg['seed']);start=0
    if saved:
        if saved['config']!=cfg or saved['provenance']!=provenance:raise ValueError('Resume configuration or provenance differs')
        if len(saved['rng_by_rank'])!=world:raise ValueError('Resume world size differs')
        checkpoint.apply(saved,model);sampler.load_state_dict(saved['sampler']);start=saved['step']
    parallel.broadcast_parameters(model.observation);parallel.broadcast_parameters(model.feedback);parallel.broadcast_parameters(model.editor)
    # Broadcast inactive adapters too: they become active at the stage transition.
    if world>1:
        for p in model.all_adapters().values():torch.distributed.broadcast(p,0)
    optimizer=optimizer_for(model)
    if saved:
        optimizer.load_state_dict(saved['optimizer']);checkpoint.restore_rng(saved['rng_by_rank'][parallel.rank()],rt.device)
    guard=FrozenGuard(rt.model,rt.bundle.vae)
    if parallel.rank()==0:
        write(root/'run.json',dict(config=cfg,provenance=provenance,world=world,start_step=start,
            total_budget=cfg['steps'],observation_steps=cfg['observation_steps'],edit_steps=cfg['steps']-cfg['observation_steps'],
            adapter_parameters={role:sum(p.numel() for p in getattr(model,role).adapters().values()) for role in ('observation','feedback','editor')},
            new_output_heads=0,heads='frozen native lm_head and llm2vae',train_pairs=len(sampler.ids),dataset=corpus.manifest))
    grid=times(cfg);nodes=[grid[i] for i in cfg['feedback_indices']]
    reset_peak_memory_stats(rt.device);began=time.perf_counter()
    for step in range(start+1,cfg['steps']+1):
        wanted='observation' if step<=cfg['observation_steps'] else 'edit'
        if model.stage!=wanted:
            model.set_stage(wanted);optimizer=optimizer_for(model)
            if parallel.rank()==0:print(json.dumps(dict(event='stage_transition',step=step,stage=wanted)),flush=True)
        started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        ids=sampler.take(batch);draws=[(sampler.rng.choice(nodes),sampler.rng.randrange(2**63-1)) for _ in ids]
        # Equal per-pair weighting; both edit-round losses contribute half, step only once.
        stats=[0.]*7
        for i in range(parallel.rank(),batch,world):
            row=corpus.row(ids[i]);t,seed=draws[i]
            posterior=(zlib.crc32(row['uid'].encode())+cfg['seed'])%(2**32)
            x0=rt.encode(row['image'],posterior)
            if wanted=='observation':
                loss=model.caption_loss(x0,row['prompt']);(loss/batch).backward();stats[0]+=float(loss.detach())
                del loss,x0;continue
            epsilon=torch.randn(x0.shape,generator=torch.Generator().manual_seed(seed)).to(rt.device)
            x=(1-t)*x0.float()+t*epsilon;flow=rt.flow(row['prompt'])
            vg,_,vc=rt.proposal(x,t,flow,capture=False,with_conditional=True)
            truth=epsilon.float()-x0.float();g=vg.float()-vc.float();target_delta=project_safe(truth-vc.float(),g)
            target=vc.float()+target_delta;proposal=(x.float()-t*vg.float()).detach()
            stats[1]+=float(target_delta.square().mean())
            for r in range(cfg['rounds']):
                w=model.round(proposal,t,flow)
                loss=F.mse_loss(w.float(),target);(loss/(batch*cfg['rounds'])).backward()
                proposal,delta=apply_edit(x,t,vc,vg,w.detach())
                coefficient=float((delta*g).sum()/g.square().sum().clamp_min(1e-12))
                stats[0]+=float(loss.detach())/cfg['rounds'];stats[2+r]+=float(loss.detach())
                stats[4]+=float(coefficient < -1e-6)/cfg['rounds']
                stats[5]+=float((vg.float()+delta-truth).square().mean())/cfg['rounds']
                stats[6]+=float(delta.square().mean().sqrt())/cfg['rounds']
                del w,loss,delta
            del x0,epsilon,x,flow,vc,vg,truth,g,target,target_delta,proposal
        # Last-layer prefix Q/O are unused by layer-input KV export; zero grads are explicit.
        parallel.sum_gradients(model,allow_unused=True)
        gradients={}
        for role in ('observation','feedback','editor'):
            ps=[p for p in getattr(model,role).adapters().values() if p.requires_grad and p.grad is not None]
            gradients['gradient_'+role]=float(torch.stack([p.grad.float().square().sum() for p in ps]).sum().sqrt()) if ps else 0.
        norm=torch.nn.utils.clip_grad_norm_(model.trainable().values(),cfg['gradient_clip'],error_if_nonfinite=True)
        within=step if wanted=='observation' else step-cfg['observation_steps']
        lr=cfg['lr']*min(1.,within/max(1,cfg['warmup_steps']))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step();guard.check();values=parallel.sum_statistics(stats,rt.device)
        peak=parallel.max_statistics([max_memory_allocated(rt.device)/1024**3],rt.device)[0];synchronize(rt.device)
        record=dict(step=step,stage=wanted,stage_step=within,loss=values[0]/batch,lr=lr,gradient_norm=float(norm),**gradients,
            step_seconds=time.perf_counter()-started,elapsed_seconds=time.perf_counter()-began,peak_memory_gib=peak,world=world)
        if wanted=='edit':record.update(zero_correction_mse=values[1]/batch,round1_loss=values[2]/batch,round2_loss=values[3]/batch,
            applied_negative_fraction=values[4]/batch,applied_physical_mse=values[5]/batch,delta_rms=values[6]/batch)
        if parallel.rank()==0:
            with (root/'train.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            print(json.dumps(record,allow_nan=False),flush=True)
        if step in cfg['save_steps'] or step==cfg['steps']:
            rngs=parallel.gather_samples([checkpoint.rng_state(rt.device)])
            if parallel.rank()==0:checkpoint.save(root/f'step_{step:06d}.pt',model,step,optimizer,sampler,provenance,rngs)
            parallel.barrier()
            if step==cfg['observation_steps'] and cfg.get('observe_validation',True):observation_validation(model,root,step)
            if step in cfg['semantic_eval_steps']:semantic_validation(model,root,step)
    corpus.close()
