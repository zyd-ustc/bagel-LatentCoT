"""Atomic adapter-only checkpoints with optimizer, sampler and all-rank RNG."""
from pathlib import Path
import os
import torch
from ..correction.checkpoint import rng_state,restore_rng

KIND='bagel_mot_state_edit_v2'


def read(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('kind') not in (KIND,'bagel_mot_cross_time_v3'):raise ValueError('Not a native-head MoT edit checkpoint')
    return saved


def apply(saved,model):
    if saved['kind']!=getattr(model,'checkpoint_kind',KIND):raise ValueError('Checkpoint pipeline differs')
    expected=model.all_adapters()
    if set(expected)!=set(saved['adapters']):raise ValueError('Role adapter inventory differs')
    with torch.no_grad():
        for name,p in expected.items():p.copy_(saved['adapters'][name].to(p.device))
    model.set_stage(saved['stage'])


def save(path,model,step,optimizer,sampler,provenance,rngs):
    path=Path(path);temporary=path.with_suffix('.partial')
    torch.save(dict(kind=getattr(model,'checkpoint_kind',KIND),step=step,stage=model.stage,config=model.config,
        adapters={n:p.detach().cpu() for n,p in model.all_adapters().items()},
        optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),provenance=provenance,rng_by_rank=rngs),temporary)
    os.replace(temporary,path)
