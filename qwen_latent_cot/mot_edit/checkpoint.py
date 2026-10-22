"""Atomic adapter-only checkpoints with optimizer, sampler and all-rank RNG."""
from pathlib import Path
import os
import torch
from ..correction.checkpoint import rng_state,restore_rng

KIND='bagel_mot_internal_edit_v1'


def read(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('kind')!=KIND:raise ValueError('Not a native-head MoT edit checkpoint')
    return saved


def apply(saved,model):
    expected=model.all_adapters()
    if set(expected)!=set(saved['adapters']):raise ValueError('Role adapter inventory differs')
    with torch.no_grad():
        for name,p in expected.items():p.copy_(saved['adapters'][name].to(p.device))
    model.set_stage(saved['stage'])


def save(path,model,step,optimizer,sampler,provenance,rngs):
    path=Path(path);temporary=path.with_suffix('.partial')
    torch.save(dict(kind=KIND,step=step,stage=model.stage,config=model.config,
        adapters={n:p.detach().cpu() for n,p in model.all_adapters().items()},
        optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),provenance=provenance,rng_by_rank=rngs),temporary)
    os.replace(temporary,path)
