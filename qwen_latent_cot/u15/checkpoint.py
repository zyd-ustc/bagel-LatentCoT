"""Rank-local exact resume; explicit portable exports for reader/GEN/corrector."""
import json
from pathlib import Path
import torch
from ..t2i.checkpoint import cpu,rng,restore_rng,rank,world,barrier


READER_INTERFACE='layerwise_source_thw_causal_caption_v1'

def parameters(module):
    return {n.replace('_fsdp_wrapped_module.','').replace('_checkpoint_wrapped_module.',''):p
            for n,p in module.named_parameters() if p.requires_grad}


class EMA:
    def __init__(self,module,decay):
        self.decay=decay
        self.values={n:p.detach().float().clone() for n,p in parameters(module).items()}
    @torch.no_grad()
    def update(self,module):
        for n,p in parameters(module).items(): self.values[n].lerp_(p.detach().float(),1-self.decay)


def save(directory,task,optimizer,ema,sampler,step,config,identity,stage,shapes):
    directory=Path(directory)
    if rank()==0: directory.mkdir(exist_ok=False)
    barrier()
    saved=dict(weights=cpu(parameters(task)),optimizer=cpu(optimizer.state_dict()),ema=cpu(ema.values),
        sampler=sampler.state_dict(),rng=rng(task.runtime.device),
        calibration=cpu(task.calibration_state()))
    path=directory/f'rank_{rank():05d}.pt';temp=path.with_suffix('.partial');torch.save(saved,temp);temp.replace(path)
    barrier()
    if rank()==0:
        meta=dict(kind='u15_resume_v1',world=world(),step=step,stage=stage,config=config,identity=identity,shapes=shapes)
        temp=directory/'manifest.partial';temp.write_text(json.dumps(meta,indent=2)+'\n');temp.replace(directory/'manifest.json')
    barrier()


def resume(directory,task,optimizer,ema,sampler,config,identity,stage):
    directory=Path(directory);meta=json.loads((directory/'manifest.json').read_text())
    if meta['kind']!='u15_resume_v1' or meta['world']!=world() or meta['stage']!=stage or meta['identity']!=identity:
        raise ValueError('Resume world/stage/base/corpus/reader identity differs')
    immutable=lambda c:{k:v for k,v in c.items() if k not in ('steps','save_every','validate_every','val_images')}
    if immutable(config)!=immutable(meta['config']):raise ValueError('Resume method/optimizer differs')
    saved=torch.load(directory/f'rank_{rank():05d}.pt',map_location='cpu',weights_only=False)
    current=parameters(task)
    if set(current)!=set(saved['weights']):raise ValueError('Resume parameter inventory differs')
    with torch.no_grad():
        for n,p in current.items():p.copy_(saved['weights'][n].to(p))
    optimizer.load_state_dict(saved['optimizer']);sampler.load_state_dict(saved['sampler'])
    ema.values={n:p.to(current[n].device) for n,p in saved['ema'].items()}
    restore_rng(saved['rng'],task.runtime.device);task.load_calibration(saved['calibration'])
    return meta['step']


def export(directory,output,use_ema=False):
    directory=Path(directory);meta=json.loads((directory/'manifest.json').read_text())
    if meta['kind']!='u15_resume_v1':raise ValueError('Requires U1.5 checkpoint')
    chunks={n:[] for n in meta['shapes']};calibration=None
    for rank_id in range(meta['world']):
        saved=torch.load(directory/f'rank_{rank_id:05d}.pt',map_location='cpu',weights_only=False)
        for n in chunks:
            # Corrector/reader use replicated SUM gradients. Only GEN is FSDP sharded.
            if meta['stage']=='t2i' or rank_id==0:chunks[n].append(saved['ema' if use_ema else 'weights'][n].reshape(-1))
        if rank_id==0:calibration=saved['calibration']
    weights={n:torch.cat(values).reshape(meta['shapes'][n]) for n,values in chunks.items()}
    if meta['stage']=='t2i':weights={n.removeprefix('native.'):p for n,p in weights.items()}
    elif meta['stage']=='caption':weights={n.removeprefix('corrector.reader.'):p for n,p in weights.items()}
    else:weights={n.removeprefix('corrector.'):p for n,p in weights.items()}
    base=meta['identity']['native']
    result=dict(reader_interface=READER_INTERFACE,kind='u15_portable_v1',stage=meta['stage'],step=meta['step'],identity=base,
        training_identity=meta['identity'],config=meta['config'],weights=weights,calibration=calibration,
        weights_mode='ema' if use_ema else 'online')
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    torch.save(result,path)
    return result


def load_portable(path,module,runtime,stage):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['kind']!='u15_portable_v1' or saved['stage']!=stage or saved['identity']!=runtime.identity:
        raise ValueError('Portable checkpoint stage/base differs')
    if stage in ('caption','correct') and saved.get('reader_interface')!=READER_INTERFACE:
        raise ValueError('Reader interface differs; retrain/export the current layerwise reader')
    if stage=='correct':validate_sampling(runtime.config,saved['config'])
    parameters=dict(module.named_parameters())
    if set(parameters)!=set(saved['weights']):raise ValueError('Portable checkpoint parameter inventory differs')
    with torch.no_grad():
        for n,p in parameters.items():p.copy_(saved['weights'][n].to(p))
    if stage=='correct':
        module.global_sums.copy_(saved['calibration']['sums'].to(runtime.device))
        module.global_counts.copy_(saved['calibration']['counts'].to(runtime.device))
    return saved


def validate_sampling(config,saved):
    for name in ('width','height','num_steps','timestep_shift','cfg_scale','cfg_norm','t_eps','active_steps','lora_rank'):
        if config[name]!=saved[name]:raise ValueError('Sampling contract differs: '+name)
