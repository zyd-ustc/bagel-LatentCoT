"""Natural-pair residual MSE, synchronous accumulation, and paired image validation."""
import json
from pathlib import Path
import time
import zlib
import torch
from . import distributed as parallel
from . import checkpoint
from .data import Sampler
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


def paired_state(runtime,row,t,noise_seed):
    # Stable posterior per sample across epochs/ranks without content-hashing every image.
    posterior_seed=(zlib.crc32(row['sample_id'].encode())+runtime.config['seed'])%(2**32)
    x0=runtime.encode(row['image'],posterior_seed)
    epsilon=torch.randn(x0.shape,generator=torch.Generator().manual_seed(noise_seed)).to(x0.device)
    x_t=(1-t)*x0.float()+t*epsilon
    flow=runtime.flow(row['caption'])
    base,context=runtime.predict(x_t,t,flow)
    return x0,epsilon,x_t,flow,base,context


@torch.no_grad()
def validate(corrector,rows,output,step):
    cfg=corrector.config;runtime=corrector.runtime;directory=Path(output)/f'validation_{step:06d}';directory.mkdir(exist_ok=True)
    records=[];nodes=times(cfg)[:active_steps(cfg)]
    for index in range(parallel.rank(),len(rows),parallel.world_size()):
        row=rows[index];t=nodes[-1]
        x0,epsilon,x_t,flow,base,context=paired_state(runtime,row,t,cfg['seed']+index)
        delta=corrector(proposal_from(x_t,t,base),t,flow,context)
        records.append(dict(sample_id=row['sample_id'],family_id=row['family_id'],caption=row['caption'],
            residual_mse=float(correction_loss(delta,base,x0,epsilon)),
            base_mse=float((base.float()-(epsilon-x0.float())).square().mean()),
            delta_rms=float(delta.square().mean().sqrt())))
        del x0,epsilon,x_t,flow,base,context,delta
        for native,label in ((True,'native'),(False,'corrected')):
            image,trace=sample(corrector,row['caption'],cfg['seed']+index,native=native)
            image.save(directory/f'{index:03d}_{label}.png');write(directory/f'{index:03d}_{label}.json',trace)
    records=parallel.gather_samples(records)
    if parallel.rank()==0:write(directory/'records.json',records)
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
            objective='MSE(delta, epsilon-x0-v0)',initialization='native_pretrained_UND',
            validation=[dict(sample_id=r['sample_id'],family_id=r['family_id']) for r in val]))
    nodes=times(cfg)[:active_steps(cfg)];reset_peak_memory_stats(device);began=time.perf_counter()
    for step in range(start+1,cfg['steps']+1):
        started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        rows=sampler.take(batch);draws=[(sampler.rng.choice(nodes),sampler.rng.randrange(2**63-1)) for _ in rows]
        values=[0.]*4
        for index in range(parallel.rank(),batch,world):
            t,noise_seed=draws[index]
            x0,epsilon,x_t,flow,base,context=paired_state(corrector.runtime,rows[index],t,noise_seed)
            delta=corrector(proposal_from(x_t,t,base),t,flow,context)
            loss=correction_loss(delta,base,x0,epsilon);(loss/batch).backward()
            with torch.no_grad():
                values[0]+=float(loss);values[1]+=float((epsilon-x0.float()-base.float()).square().mean())
                values[2]+=float(delta.square().mean().sqrt());values[3]+=t
            del x0,epsilon,x_t,flow,base,context,delta,loss
        parallel.sum_gradients(corrector)
        groups=corrector.group_parameters()
        gradients={f'gradient_{name}':float(torch.stack([p.grad.float().square().sum() for p in parameters]).sum().sqrt()) for name,parameters in groups.items()}
        norm=torch.nn.utils.clip_grad_norm_([p for p in corrector.parameters() if p.requires_grad],cfg['gradient_clip'],error_if_nonfinite=True)
        lr=cfg['lr']*min(1.,step/max(1,cfg['warmup_steps']))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step();guard.check()
        values=parallel.sum_statistics(values,device)
        peak=parallel.max_statistics([max_memory_allocated(device)/1024**3],device)[0];synchronize(device)
        record=dict(step=step,loss=values[0]/batch,base_mse=values[1]/batch,delta_rms=values[2]/batch,t_mean=values[3]/batch,
                    gradient_norm=float(norm),**gradients,lr=lr,step_seconds=time.perf_counter()-started,
                    elapsed_seconds=time.perf_counter()-began,peak_memory_gib=peak,world=world)
        if parallel.rank()==0:
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            print(json.dumps(record,allow_nan=False),flush=True)
        if step in cfg['save_steps'] or step==cfg['steps']:
            rngs=parallel.gather_samples([checkpoint.rng_state(device)])
            if parallel.rank()==0:checkpoint.save(out/f'step_{step:06d}.pt',corrector,step,optimizer,sampler,provenance,rngs)
            parallel.barrier()
            if val and (step==cfg['steps'] or step%100==0):validate(corrector,val,out,step)
    corpus.close()
