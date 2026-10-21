"""Trainable-only checkpoints; reconstruct frozen weights from the native model."""
from pathlib import Path
import os
import random
import torch

KIND='bagel_und_spatial_correction_v1'


def rng_state(device):
    return dict(cpu=torch.get_rng_state(),python=random.getstate(),
                accelerator=getattr(torch,device.type).get_rng_state(device).cpu() if device.type!='cpu' else None)


def restore_rng(state,device):
    torch.set_rng_state(state['cpu']);random.setstate(state['python'])
    if device.type!='cpu':getattr(torch,device.type).set_rng_state(state['accelerator'],device)


def read(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('kind')!=KIND:raise ValueError('Requires a new spatial-correction checkpoint; legacy training cannot resume')
    return saved


def apply(saved,corrector):
    expected=corrector.trainable();actual=saved['trainable']
    if set(actual)!=set(expected):raise ValueError('Corrector parameter inventory differs')
    with torch.no_grad():
        for name,p in expected.items():p.copy_(actual[name].to(p.device))


def save(path,corrector,step,optimizer,sampler,provenance,rng_by_rank):
    saved=dict(kind=KIND,step=step,config=corrector.config,
        trainable={name:p.detach().cpu() for name,p in corrector.trainable().items()},optimizer=optimizer.state_dict(),
        sampler=sampler.state_dict(),provenance=provenance,rng_by_rank=rng_by_rank)
    path=Path(path);temporary=path.with_suffix('.partial')
    torch.save(saved,temporary);os.replace(temporary,path)
