"""Caption inheritance, native GEN SFT, and single-loss guidance-safe training."""
import json
from pathlib import Path
import random
import time
import torch
from torch import nn
from torch.nn import functional as F
from ..correction import distributed as dist
from ..t2i.data import Pairs,Sampler,pixels
from .checkpoint import EMA,parameters
from .readout import Corrector
from .objective import residual_target,project,coefficient
from .runtime import gen_parameter,autocast
from . import checkpoint


class Task(nn.Module):
    def __init__(self,runtime,config,stage,reader=None):
        super().__init__();object.__setattr__(self,'runtime',runtime)
        self.config,self.stage=config,stage
        if stage=='t2i':
            self.native=runtime.model
            for name,p in self.native.named_parameters():p.requires_grad_(gen_parameter(name))
        else:
            self.corrector=Corrector(runtime,config);self.corrector.initialize_global(config['active_steps'])
            if stage=='caption':self.corrector.head.requires_grad_(False)
            elif reader:checkpoint.load_portable(reader,self.corrector.reader,runtime,'caption')
            else:raise ValueError('Correction requires a trained U1.5 caption reader export')

    def calibration_state(self):
        if self.stage=='t2i':return None
        return dict(sums=self.corrector.global_sums,counts=self.corrector.global_counts)

    def load_calibration(self,state):
        if state is not None:
            self.corrector.global_sums.copy_(state['sums'].to(self.runtime.device))
            self.corrector.global_counts.copy_(state['counts'].to(self.runtime.device))

    def forward(self,clean,prompt,node,noise_seed,drop=False):
        rt=self.runtime;device=rt.device
        noise=torch.randn(clean.shape,generator=torch.Generator(device=device).manual_seed(noise_seed),device=device)
        t=rt.times()[node];sigma=rt.noise_scale();state=t*clean+(1-t)*sigma*noise
        prefix=rt.prefix(None if drop else prompt)
        if self.stage=='t2i':
            prediction=rt.predict(state.to(next(rt.model.parameters()).dtype),t,prefix).velocity
            target=clean-sigma*noise
            return F.mse_loss(prediction.float(),target.float()),dict(t=float(t))
        with torch.no_grad():
            predicted=rt.predict(state.to(next(rt.model.parameters()).dtype),t,prefix,capture=True)
        if self.stage=='caption':
            return self.corrector.reader.caption_loss(predicted.observation,t,prompt),dict(t=float(t))
        with torch.no_grad():
            reference=rt.predict(state.to(next(rt.model.parameters()).dtype),t,rt.prefix(None)).velocity if self.config['cfg_scale']>1 else predicted.velocity
            base=rt.guided(predicted.velocity,reference,self.config['cfg_scale'])
            target=residual_target(clean,noise,sigma,predicted.velocity,base)
        raw=self.corrector(predicted.observation,t,prefix.hidden)
        guidance=base.float()-predicted.velocity.float()
        delta=project(raw,guidance)
        stats=dict(t=float(t),zero_correction_mse=float(target.square().mean()),
            conditional_base_mse=float((clean-sigma*noise-predicted.velocity.float()).square().mean()),
            applied_base_mse=float((base.float()+delta.detach()-clean+sigma*noise).square().mean()),
            conditional_corrected_mse=float((predicted.velocity.float()+delta.detach()-clean+sigma*noise).square().mean()),
            applied_negative_fraction=float((coefficient(delta.detach(),guidance)<-1e-6).float().mean()),
            raw_delta_rms=float(raw.detach().square().mean().sqrt()),applied_delta_rms=float(delta.detach().square().mean().sqrt()),
            calibration=target.mean(dim=(0,2,3)).detach())
        return F.mse_loss(raw,target),stats


def train(runtime,config,stage,corpus,output,reader=None,resume=None):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,MixedPrecision,ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    corpus=Pairs(corpus);output=Path(output)
    if dist.rank()==0:output.mkdir(parents=True,exist_ok=bool(resume))
    dist.barrier()
    task=Task(runtime,config,stage,reader)
    shapes={n:tuple(p.shape) for n,p in parameters(task).items()}
    identity=dict(native=runtime.identity,corpus=corpus.manifest,reader=str(Path(reader).resolve()) if reader else None,
                  observation='all_GEN_pre_block_hidden',reader_interface='layerwise_source_thw_causal_caption_v1')
    if stage=='t2i':
        # On CPU before sharding: do not materialize FP32 17B on one NPU first.
        task.float()
        task.native.train()
        if config['activation_checkpointing']:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper,CheckpointImpl
            for i,layer in enumerate(runtime.decoder.layers):
                runtime.decoder.layers[i]=checkpoint_wrapper(layer,checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        if dist.world_size()>1:
            from sensenova_u1.models.neo_unify.modeling_qwen3 import Qwen3DecoderLayer
            task=FSDP(task,auto_wrap_policy=ModuleWrapPolicy({Qwen3DecoderLayer}),use_orig_params=True,
                sharding_strategy=ShardingStrategy.FULL_SHARD,device_id=runtime.device,
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.bfloat16),
                limit_all_gathers=True)
        else:task.to(runtime.device)
    else:dist.broadcast_parameters(task)
    trainable=parameters(task)
    groups=[]
    for head in (False,True):
        params=[p for n,p in trainable.items() if ('.head.' in n)==head]
        if params:groups.append(dict(params=params,lr=config['head_lr'] if head else config['lr'],base_lr=config['head_lr'] if head else config['lr']))
    optimizer=torch.optim.AdamW(groups,betas=tuple(config['betas']),eps=config['eps'],weight_decay=0.)
    ema=EMA(task,config['ema_decay']);sampler=Sampler(corpus.ids('train'),config['seed'])
    start=checkpoint.resume(resume,task,optimizer,ema,sampler,config,identity,stage) if resume else 0
    raw_task=task.module if isinstance(task,FSDP) else task
    if dist.rank()==0:(output/'run.json').write_text(json.dumps(dict(identity=identity,stage=stage,config=config),indent=2))
    batch=config['global_batch'];world=dist.world_size()
    if batch%world:raise ValueError('Global batch must divide world size')
    try:
        for step in range(start+1,config['steps']+1):
            begin=time.monotonic();optimizer.zero_grad(set_to_none=True)
            ids=sampler.take(batch)
            count=config['num_steps'] if stage=='t2i' else config['active_steps']
            draws=[(sampler.rng.randrange(count),sampler.rng.randrange(2**31),sampler.rng.random()) for _ in ids]
            totals={};calibration=torch.zeros(config['active_steps'],4,device=runtime.device)
            for i in range(dist.rank(),batch,world):
                row=corpus.row(ids[i])
                if (row['width'],row['height'])!=(runtime.width,runtime.height):
                    raise ValueError('Corpus geometry differs; prepare for the fixed native RGB grid')
                node,noise_seed,drop=draws[i]
                with autocast(runtime.device):loss,stats=task(pixels(row,runtime.device),row['prompt'],node,noise_seed,stage=='t2i' and drop<config['text_dropout'])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                (loss/(batch/world if stage=='t2i' else batch)).backward()
                totals['loss']=totals.get('loss',0.)+float(loss.detach())
                for name,value in stats.items():
                    if name=='calibration':calibration[node,:3]+=value;calibration[node,3]+=1
                    else:totals[name]=totals.get(name,0.)+value
            if stage!='t2i':dist.sum_gradients(task)
            gradients={}
            if stage!='t2i':
                for group in ('reader','head'):
                    grads=[p.grad.float().square().sum() for n,p in trainable.items() if f'.{group}.' in n and p.grad is not None]
                    if grads:gradients['gradient_'+group]=float(torch.stack(grads).sum().sqrt())
            norm=task.clip_grad_norm_(config['gradient_clip']) if isinstance(task,FSDP) else torch.nn.utils.clip_grad_norm_(trainable.values(),config['gradient_clip'],error_if_nonfinite=True)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradients')
            for group in optimizer.param_groups:group['lr']=group['base_lr']*min(1.,step/max(1,config['warmup_steps']))
            optimizer.step();ema.update(task)
            if stage=='correct':
                if world>1:torch.distributed.all_reduce(calibration)
                raw_task.corrector.global_sums+=calibration[:,:3];raw_task.corrector.global_counts+=calibration[:,3]
            keys=sorted(totals);values=dist.sum_statistics([totals[k] for k in keys],runtime.device)
            record=dict(step=step,**{k:v/batch for k,v in zip(keys,values)},gradient_norm=float(norm),**gradients,seconds=time.monotonic()-begin)
            if dist.rank()==0:
                print(json.dumps(record),flush=True)
                with (output/'train.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            if config['validate_every'] and step%config['validate_every']==0:
                validate(task,raw_task,corpus,config,output,step)
            if step%config['save_every']==0 or step==config['steps']:
                # FSDP original parameters expose local shards outside forward.
                checkpoint.save(output/f'step_{step:06d}',task,optimizer,ema,sampler,step,config,identity,stage,shapes)
    finally:corpus.close()


@torch.no_grad()
def validate(task,raw,corpus,config,output,step):
    rt=raw.runtime;ids=corpus.ids('val')[:config['val_images']]
    if not ids:return
    records=[]
    for index in range(dist.rank(),((len(ids)+dist.world_size()-1)//dist.world_size())*dist.world_size(),dist.world_size()):
        row=corpus.row(ids[index%len(ids)]);node=min(config['active_steps']-1,config['num_steps']//4)
        with autocast(rt.device):loss,_=task(pixels(row,rt.device),row['prompt'],node,config['seed']+index)
        if index<len(ids):records.append(dict(uid=row['uid'],loss=float(loss),node=node))
    records=dist.gather_samples(records)
    if dist.rank()==0:(output/f'validation_{step:06d}.json').write_text(json.dumps(records,indent=2))
