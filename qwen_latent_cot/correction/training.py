"""Natural-pair residual MSE, synchronous accumulation, and paired image validation."""
import json
from pathlib import Path
import time
import zlib
import torch
from torch.nn import functional as F
from . import distributed as parallel
from . import checkpoint
from .data import Sampler
from .objective import guidance_safe,residual_target,applied_correction,guidance_coefficient
from .model import proposal_from,correction_loss
from .sampling import times,active_steps,sample
from ..bagel.accelerator import synchronize,reset_peak_memory_stats,max_memory_allocated


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class FrozenGuard:
    def __init__(self,*modules):
        self.parameters=[p for module in modules for p in module.parameters() if not p.requires_grad]
        self.versions=[p._version for p in self.parameters]
    def check(self):
        if any(p.requires_grad or p.grad is not None or p._version!=version for p,version in zip(self.parameters,self.versions)):
            raise RuntimeError('Frozen native/UND parameter was modified')


def paired_state(runtime,row,t,noise_seed,*,with_conditional=False):
    # Stable posterior per sample across epochs/ranks without content-hashing every image.
    posterior_seed=(zlib.crc32(row['sample_id'].encode())+runtime.config['seed'])%(2**32)
    x0=runtime.encode(row['image'],posterior_seed)
    epsilon=torch.randn(x0.shape,generator=torch.Generator().manual_seed(noise_seed)).to(x0.device)
    x_t=(1-t)*x0.float()+t*epsilon
    flow=runtime.flow(row['caption'])
    if runtime.config.get('correction_space')=='post_cfg':
        if with_conditional:base,context,conditional=runtime.proposal(x_t,t,flow,with_conditional=True)
        else:base,context=runtime.proposal(x_t,t,flow)
    else:
        base,context=runtime.predict(x_t,t,flow);conditional=base
    result=(x0,epsilon,x_t,flow,base,context)
    return (*result,conditional) if with_conditional else result



@torch.no_grad()
def validate(corrector,rows,output,step):
    cfg=corrector.config;runtime=corrector.runtime;directory=Path(output)/f'validation_{step:06d}';directory.mkdir(exist_ok=True)
    records=[];nodes=times(cfg)[:active_steps(cfg)];deployed=cfg.get('correction_space')=='post_cfg'
    selected=sorted(set((0,len(nodes)//3,2*len(nodes)//3,len(nodes)-1))) if deployed else [len(nodes)-1]
    for index in range(parallel.rank(),len(rows),parallel.world_size()):
        row=rows[index]
        for node in selected:
            t=nodes[node]
            x0,epsilon,x_t,flow,base,context,conditional=paired_state(runtime,row,t,cfg['seed']+index,with_conditional=True)
            raw_delta=corrector(proposal_from(x_t,t,base),t,flow,context)
            delta=applied_correction(raw_delta,base,cfg,conditional)
            target=residual_target(base,x0,epsilon,cfg,conditional)
            record=dict(sample_id=row['sample_id'],family_id=row['family_id'],caption=row['caption'],t=t,
                residual_mse=float(correction_loss(delta,base,x0,epsilon)),
                base_mse=float((base.float()-(epsilon-x0.float())).square().mean()),delta_rms=float(delta.square().mean().sqrt()))
            record.update(objective_mse=float(F.mse_loss(raw_delta.float(),target)),zero_correction_mse=float(target.square().mean()),
                conditional_base_mse=float((conditional.float()-(epsilon-x0.float())).square().mean()),
                conditional_corrected_mse=float((conditional.float()+delta.float()-(epsilon-x0.float())).square().mean()),
                raw_delta_rms=float(raw_delta.float().square().mean().sqrt()))
            if guidance_safe(cfg):
                g=base.float()-conditional.float()
                record.update(guidance_projection_raw=float(guidance_coefficient(raw_delta,g)),
                    guidance_projection_applied=float(guidance_coefficient(delta,g)))
            if deployed:
                global_delta=applied_correction(corrector.calibration(t).expand_as(base),base,cfg,conditional)
                record['global_mse']=float(correction_loss(global_delta,base,x0,epsilon))
            records.append(record)
            del x0,epsilon,x_t,flow,base,context,delta,raw_delta,target,conditional
        arms=[('native',dict(native=True)),('corrected',{})]
        if deployed:arms.insert(1,('global',dict(global_only=True)))
        for label,kwargs in arms:
            image,trace=sample(corrector,row['caption'],cfg['seed']+index,**kwargs)
            image.save(directory/f'{index:03d}_{label}.png');write(directory/f'{index:03d}_{label}.json',trace)
    records=parallel.gather_samples(records)
    if parallel.rank()==0:write(directory/'records.json',records)
    parallel.barrier()


@torch.no_grad()
def semantic_validation(corrector,output,step):
    """Fixed annotated prompts; three image arms, not an automatic success score."""
    cfg=corrector.config
    prompt_path=Path(cfg['semantic_prompts'])
    if not prompt_path.is_absolute():prompt_path=Path(__file__).resolve().parents[2]/prompt_path
    prompts=[json.loads(line) for line in prompt_path.read_text().splitlines() if line.strip()]
    directory=Path(output)/f'semantic_{step:06d}';directory.mkdir(exist_ok=True)
    baseline=Path(output)/'semantic_native';baseline.mkdir(exist_ok=True)
    records=[]
    for index in range(parallel.rank(),len(prompts),parallel.world_size()):
        row=prompts[index];record=dict(row,index=index,seed=cfg['seed'],arms={})
        for label,kwargs in [('native',dict(native=True)),('global',dict(global_only=True)),('corrected',{})]:
            target=(baseline if label=='native' else directory)/f'{index:03d}_{label}.png'
            if not target.exists():
                image,trace=sample(corrector,row['prompt'],cfg['seed'],**kwargs)
                image.save(target);write(target.with_suffix('.json'),trace)
            record['arms'][label]=str(target.relative_to(output))
        write(directory/f'case_{index:03d}.json',record);records.append(record)
    records=parallel.gather_samples(records)
    if parallel.rank()==0:write(directory/'records.json',sorted(records,key=lambda r:r['index']))
    parallel.barrier()


def train(corrector,corpus,output,provenance,saved=None):
    cfg=corrector.config;device=corrector.runtime.device;out=Path(output)
    if parallel.rank()==0 and out.exists() and not saved:raise FileExistsError('Fresh training requires a fresh output directory')
    if parallel.rank()==0:out.mkdir(parents=True,exist_ok=bool(saved))
    parallel.barrier()
    sampler=Sampler(corpus,cfg['seed']);batch=cfg['global_batch'];world=parallel.world_size()
    if batch%world:raise ValueError('Global batch must be divisible by world size')
    parallel.broadcast_parameters(corrector)
    optimizer=torch.optim.AdamW([p for p in corrector.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    start=0
    if saved:
        if {k:v for k,v in saved['provenance'].items() if k!='source_revision'}!={k:v for k,v in provenance.items() if k!='source_revision'}:
            raise ValueError('Resume native model/corpus provenance differs')
        mutable={'steps','save_steps','val_images','activation_checkpointing'}
        if {k:v for k,v in saved['config'].items() if k not in mutable}!={k:v for k,v in cfg.items() if k not in mutable}:
            raise ValueError('Resume optimization/method config differs')
        if len(saved['rng_by_rank'])!=world:raise ValueError('Exact resume requires the same world size')
        checkpoint.apply(saved,corrector);optimizer.load_state_dict(saved['optimizer']);sampler.load_state_dict(saved['sampler'])
        checkpoint.restore_rng(saved['rng_by_rank'][parallel.rank()],device);start=saved['step']
    guard=FrozenGuard(corrector.runtime.model,corrector.runtime.bundle.vae,corrector)
    val=corpus.validation(cfg['val_images']) if cfg['val_images'] else []
    if parallel.rank()==0:
        write(out/'run.json',dict(config=cfg,provenance=provenance,world=world,start_step=start,
            train_pairs=sum(len(v) for v in corpus.groups('train')),trainable_parameters=sum(p.numel() for p in corrector.parameters() if p.requires_grad),
            objective='MSE(raw_delta, H_g(epsilon-x0-v_conditional))' if guidance_safe(cfg) else 'MSE(delta, epsilon-x0-v0)',
            metric_meanings=dict(loss='raw prediction MSE to the configured training target',zero_correction_mse='zero prediction MSE to the same target',
                base_mse='unmodified base velocity MSE to epsilon-x0',applied_base_mse='base plus actually applied correction MSE to epsilon-x0'),
            velocity_space=cfg.get('correction_space','conditional'),
            global_baseline='train-only per-time constant-field least-squares mean' if hasattr(corrector,'calibration') else None,initialization='native_pretrained_UND',
            validation=[dict(sample_id=r['sample_id'],family_id=r['family_id']) for r in val]))
    nodes=times(cfg)[:active_steps(cfg)];reset_peak_memory_stats(device);began=time.perf_counter()
    for step in range(start+1,cfg['steps']+1):
        started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        rows=sampler.take(batch);draws=[(sampler.rng.choice(nodes),sampler.rng.randrange(2**63-1)) for _ in rows]
        values=[0.]*4
        extra_names=('zero_correction_mse','conditional_base_mse','conditional_corrected_mse','applied_base_mse',
                     'target_anti_removed_rms','raw_negative_fraction','applied_negative_fraction','raw_delta_rms')
        extras=[0.]*len(extra_names)
        pending=torch.zeros((len(nodes),corrector.calibration.sums.shape[1]+1),device=device) if hasattr(corrector,'calibration') else None
        for index in range(parallel.rank(),batch,world):
            t,noise_seed=draws[index]
            x0,epsilon,x_t,flow,base,context,conditional=paired_state(corrector.runtime,rows[index],t,noise_seed,with_conditional=True)
            delta=corrector(proposal_from(x_t,t,base),t,flow,context)
            target=residual_target(base,x0,epsilon,cfg,conditional)
            loss=F.mse_loss(delta.float(),target);(loss/batch).backward()
            if pending is not None:corrector.calibration.observe(pending,t,target)
            with torch.no_grad():
                values[0]+=float(loss);values[1]+=float((epsilon-x0.float()-base.float()).square().mean())
                applied=applied_correction(delta,base,cfg,conditional)
                values[2]+=float(applied.square().mean().sqrt());values[3]+=t
                truth=epsilon.float()-x0.float();g=base.float()-conditional.float()
                conditional_residual=truth-conditional.float()
                row_extras=(float(target.square().mean()),float(conditional_residual.square().mean()),
                    float((conditional.float()+applied-truth).square().mean()),float((base.float()+applied-truth).square().mean()),
                    float((conditional_residual-target).square().mean().sqrt()) if guidance_safe(cfg) else 0.,
                    float(guidance_coefficient(delta,g)<-1e-6),float(guidance_coefficient(applied,g)<-1e-6),
                    float(delta.float().square().mean().sqrt()))
                extras=[a+b for a,b in zip(extras,row_extras)]
            del x0,epsilon,x_t,flow,base,context,delta,loss,target,conditional,applied,truth,g,conditional_residual
        if pending is not None:corrector.calibration.update(pending)
        parallel.sum_gradients(corrector)
        groups=corrector.group_parameters()
        gradients={f'gradient_{name}':float(torch.stack([p.grad.float().square().sum() for p in parameters]).sum().sqrt()) for name,parameters in groups.items()}
        norm=torch.nn.utils.clip_grad_norm_([p for p in corrector.parameters() if p.requires_grad],cfg['gradient_clip'],error_if_nonfinite=True)
        lr=cfg['lr']*min(1.,step/max(1,cfg['warmup_steps']))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step();guard.check()
        combined=parallel.sum_statistics(values+extras,device);values=combined[:4];extras=combined[4:]
        peak=parallel.max_statistics([max_memory_allocated(device)/1024**3],device)[0];synchronize(device)
        record=dict(step=step,loss=values[0]/batch,base_mse=values[1]/batch,delta_rms=values[2]/batch,t_mean=values[3]/batch,
                    gradient_norm=float(norm),**gradients,lr=lr,step_seconds=time.perf_counter()-started,
                    elapsed_seconds=time.perf_counter()-began,peak_memory_gib=peak,world=world)
        record.update({name:value/batch for name,value in zip(extra_names,extras)})
        if parallel.rank()==0:
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            print(json.dumps(record,allow_nan=False),flush=True)
        if step in cfg['save_steps'] or step==cfg['steps']:
            rngs=parallel.gather_samples([checkpoint.rng_state(device)])
            if parallel.rank()==0:checkpoint.save(out/f'step_{step:06d}.pt',corrector,step,optimizer,sampler,provenance,rngs)
            parallel.barrier()
            if val and (step==cfg['steps'] or step%100==0):validate(corrector,val,out,step)
            if step in cfg.get('semantic_eval_steps',[]):semantic_validation(corrector,out,step)
    corpus.close()
