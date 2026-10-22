"""GEN-only Table3 optimizer recipe; FULL_SHARD for original 7B generation expert."""
import json
import math
from pathlib import Path
import random
import time
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from ..bagel.accelerator import seeded_context, synchronize, max_memory_allocated
from ..bagel.modeling.bagel.qwen2_navit import Qwen2MoTDecoderLayer
from .model import NativeT2I,sequence
from .data import Pairs,Sampler,pixels
from . import checkpoint as ck


def wrap(bundle, config, device):
    bundle.model.timestep_shift=config['timestep_shift']
    native=NativeT2I(bundle.model,config['activation_checkpointing']).float()
    shapes={'model.'+n:s for n,s in native.parameter_shapes.items()}
    if ck.world()>1:
        native=FSDP(native,auto_wrap_policy=ModuleWrapPolicy({Qwen2MoTDecoderLayer}),
            sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=True,device_id=device,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,cast_root_forward_inputs=False),limit_all_gathers=True)
    else:native=native.to(device)
    bundle.vae.eval().requires_grad_(False).to(device)
    return native,shapes


def example(bundle,row,config,device,drop=False):
    prompt='' if drop else row['prompt']
    inputs=sequence(bundle.model,bundle.tokenizer,bundle.token_ids,prompt,row['height'],row['width'],device,config['max_sequence_length'])
    with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16):
        latent=bundle.vae.encode(pixels(row,device).to(next(bundle.vae.parameters()).dtype))[0].float()
    return latent,inputs


def validate(model,bundle,corpus,config,device,ema,step,output):
    ids=corpus.ids('val')[:config['val_images']]
    if not ids:raise ValueError('Empty validation selection')
    current=ck.parameters(model)
    backup={n:p.detach().clone() for n,p in current.items()}
    rows=[]
    try:
        with torch.no_grad():
            for n,p in current.items():p.copy_(ema.values[n])
        # Equal forward counts on every FSDP rank, ignoring only padded observations.
        for index in range(ck.rank(),math.ceil(len(ids)/ck.world())*ck.world(),ck.world()):
            row=corpus.row(ids[index%len(ids)])
            with seeded_context(device,config['seed']+row['id']),torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16):
                latent,inputs=example(bundle,row,config,device)
                loss,_,t=model(latent,inputs,torch.randn((),device=device))
            if index<len(ids):rows.append(dict(uid=row['uid'],mse=float(loss),t=float(t)))
    finally:
        with torch.no_grad():
            for n,p in current.items():p.copy_(backup[n])
    if ck.world()>1:
        gathered=[None]*ck.world();dist.all_gather_object(gathered,rows);rows=[r for rs in gathered for r in rs]
    if ck.rank()==0:
        (Path(output)/f'validation_{step:06d}.json').write_text(json.dumps(dict(step=step,weights='ema',mse=sum(r['mse'] for r in rows)/len(rows),records=rows),indent=2)+'\n')


def train(bundle,corpus_path,output,config,device,resume=None):
    output=Path(output)
    if ck.rank()==0:output.mkdir(parents=True,exist_ok=bool(resume))
    ck.barrier()
    corpus=Pairs(corpus_path)
    try:
        return _train(bundle,corpus,output,config,device,resume)
    finally:
        corpus.close()


def _train(bundle,corpus,output,config,device,resume):
    for k in ('min_short_side','max_long_side','max_sequence_length'):
        if config[k]!=corpus.manifest[k]:raise ValueError('Prepared data recipe differs: '+k)
    if config['global_batch']%ck.world():raise ValueError('Global batch must be divisible by world size')
    per_rank=config['global_batch']//ck.world()
    torch.manual_seed(config['seed']+ck.rank());random.seed(config['seed']+ck.rank())
    model,shapes=wrap(bundle,config,device)
    parameters=list(ck.parameters(model).values())
    optimizer=torch.optim.AdamW(parameters,lr=config['lr'],betas=tuple(config['betas']),eps=config['eps'],weight_decay=config['weight_decay'])
    ema=ck.EMA(model,config['ema_decay']);sampler=Sampler(corpus.ids('train'),config['seed'])
    provenance=dict(native_model=str(Path(bundle.model_path).resolve()),corpus=corpus.manifest['identity'])
    start=ck.resume(resume,model,optimizer,ema,sampler,config,provenance,device) if resume else 0
    if ck.rank()==0:
        (output/'run.json').write_text(json.dumps(dict(config=config,world=ck.world(),provenance=provenance,trainable_parameters=sum(math.prod(s) for s in shapes.values()),
            method='native GEN-only T2I Rectified Flow',start=start,precision='FP32 master/optimizer/EMA; BF16 compute',
            recipe_scope='Table3 SFT optimizer/resolution/15000steps; pure T2I only, no mixed tasks or 40K packed sequences'),indent=2)+'\n')
    for step in range(start+1,config['steps']+1):
        began=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        batch=sampler.take(config['global_batch']);total=dropped=0.
        for index in range(ck.rank(),len(batch),ck.world()):
            row=corpus.row(batch[index]);drop=random.random()<config['text_dropout'];dropped+=drop
            latent,inputs=example(bundle,row,config,device,drop)
            with torch.autocast(device.type,dtype=torch.bfloat16):loss,_,_=model(latent,inputs,torch.randn((),device=device))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native T2I loss')
            (loss/per_rank).backward();total+=float(loss.detach());del latent,inputs,loss
        norm=model.clip_grad_norm_(config['gradient_clip']) if isinstance(model,FSDP) else torch.nn.utils.clip_grad_norm_(parameters,config['gradient_clip'],error_if_nonfinite=True)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite native T2I gradient norm')
        lr=config['lr']*min(1.,step/max(1,config['warmup_steps']))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step();ema.update(model)
        values=torch.tensor([total,dropped],device=device)
        if ck.world()>1:dist.all_reduce(values)
        synchronize(device)
        record=dict(step=step,loss=float(values[0])/config['global_batch'],text_dropout_fraction=float(values[1])/config['global_batch'],
            gradient_norm=float(norm),lr=lr,step_seconds=time.perf_counter()-began,epoch=sampler.epoch,
            peak_memory_gib=max_memory_allocated(device)/1024**3,world=ck.world())
        if ck.rank()==0:
            with (output/'train.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
            print(json.dumps(record),flush=True)
        if step%config['validate_every']==0 or step==config['steps']:
            validate(model,bundle,corpus,config,device,ema,step,output)
        if step==1 or step%config['save_every']==0 or step==config['steps']:
            ck.save(output/f'step_{step:06d}',model,optimizer,ema,sampler,step,config,provenance,shapes,device)
