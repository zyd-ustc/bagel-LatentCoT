"""Exact-layout rank-local FSDP resume and portable native GEN/EMA export."""
import json
from pathlib import Path
import random
import torch
import torch.distributed as dist
from safetensors.torch import save_file

KIND='bagel_native_t2i_sft_v1'


def rank():return dist.get_rank() if dist.is_initialized() else 0

def world():return dist.get_world_size() if dist.is_initialized() else 1

def barrier():
    if dist.is_initialized():dist.barrier()


def parameters(model):
    return {n.replace('_fsdp_wrapped_module.',''):p for n,p in model.named_parameters() if p.requires_grad}


def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return value


def rng(device):
    return dict(cpu=torch.get_rng_state(),python=random.getstate(),
                device=getattr(torch,device.type).get_rng_state(device).cpu() if device.type!='cpu' else None)


def restore_rng(saved,device):
    torch.set_rng_state(saved['cpu']);random.setstate(saved['python'])
    if device.type!='cpu':getattr(torch,device.type).set_rng_state(saved['device'],device)


class EMA:
    def __init__(self, model, decay):
        self.decay=decay
        self.values={n:p.detach().float().clone() for n,p in parameters(model).items()}
    @torch.no_grad()
    def update(self,model):
        for n,p in parameters(model).items():self.values[n].lerp_(p.detach().float(),1-self.decay)


def save(directory, model, optimizer, ema, sampler, step, config, provenance, shapes, device):
    directory=Path(directory)
    if rank()==0:directory.mkdir(exist_ok=False)
    barrier()
    saved=dict(weights=cpu(parameters(model)),optimizer=cpu(optimizer.state_dict()),ema=cpu(ema.values),
               sampler=sampler.state_dict(),rng=rng(device))
    temporary=directory/f'rank_{rank():05d}.partial'
    torch.save(saved,temporary);temporary.replace(directory/f'rank_{rank():05d}.pt')
    barrier()
    if rank()==0:
        manifest=dict(kind=KIND,step=step,world=world(),config=config,provenance=provenance,shapes=shapes)
        temporary=directory/'manifest.partial';temporary.write_text(json.dumps(manifest,indent=2)+'\n')
        temporary.replace(directory/'manifest.json')
    barrier()


def resume(directory,model,optimizer,ema,sampler,config,provenance,device):
    directory=Path(directory);meta=json.loads((directory/'manifest.json').read_text())
    if meta['kind']!=KIND or meta['world']!=world():raise ValueError('Exact resume requires native T2I checkpoint and same world size')
    if meta['provenance']!=provenance:raise ValueError('Native model or prepared corpus changed')
    mutable={'steps','save_every','validate_every','val_images'}
    if {k:v for k,v in config.items() if k not in mutable}!={k:v for k,v in meta['config'].items() if k not in mutable}:
        raise ValueError('Optimization or training contract changed')
    saved=torch.load(directory/f'rank_{rank():05d}.pt',map_location='cpu',weights_only=False)
    current=parameters(model)
    if set(current)!=set(saved['weights']):raise ValueError('Trainable parameter inventory changed')
    with torch.no_grad():
        for n,p in current.items():
            if p.shape!=saved['weights'][n].shape:raise ValueError('Local FSDP partition changed')
            p.copy_(saved['weights'][n].to(p.device))
    optimizer.load_state_dict(saved['optimizer'])
    ema.values={n:v.to(current[n].device) for n,v in saved['ema'].items()}
    sampler.load_state_dict(saved['sampler']);restore_rng(saved['rng'],device)
    return meta['step']


def export(directory, output, use_ema=True):
    directory=Path(directory);output=Path(output)
    meta=json.loads((directory/'manifest.json').read_text())
    if meta['kind']!=KIND:raise ValueError('Not a native T2I checkpoint')
    # Read one rank at a time. Frozen parameters are reconstructed from native weights.
    chunks={n:[] for n in meta['shapes']}
    for i in range(meta['world']):
        saved=torch.load(directory/f'rank_{i:05d}.pt',map_location='cpu',weights_only=False)
        for n in chunks:chunks[n].append(saved['ema' if use_ema else 'weights'][n].reshape(-1).to(torch.bfloat16))
        del saved
    weights={n.removeprefix('model.'):torch.cat(parts).reshape(meta['shapes'][n]).contiguous() for n,parts in chunks.items()}
    output.mkdir(parents=True,exist_ok=False)
    save_file(weights,str(output/'gen.safetensors'))
    (output/'manifest.json').write_text(json.dumps(dict(meta,export_weights='ema' if use_ema else 'online'),indent=2)+'\n')
    return output
