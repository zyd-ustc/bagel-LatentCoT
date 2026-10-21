"""Portable interaction checkpoints; original format-4 files are parents only."""
import os
import random
from pathlib import Path
import torch

KIND = 'bagel_full_loop_v3'
STAGE_A_KIND = 'bagel_full_loop_stage_a_v4'
KINDS = (KIND, STAGE_A_KIND)

def kind(student):
    return STAGE_A_KIND if student.config.get('stage_a') else KIND


def read(path):
    from ..diagnostics.inference import resolve_checkpoint
    return torch.load(str(resolve_checkpoint(path)), weights_only=True, map_location='cpu', mmap=True)


def cpu_state(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu()
    if isinstance(values, dict):
        return {k: cpu_state(v) for k, v in values.items()}
    if isinstance(values, (list, tuple)):
        return type(values)(cpu_state(v) for v in values)
    return values


def rng_state(device):
    state = dict(python=random.getstate(), torch=torch.get_rng_state())
    if device.type != 'cpu':
        state['accelerator'] = getattr(torch, device.type).get_rng_state(device).cpu()
    return state


def restore_rng(state, device):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if 'accelerator' in state:
        getattr(torch, device.type).set_rng_state(state['accelerator'].cpu(), device)


def save(path, student, stage, step, *, optimizer=None, sampler=None, provenance=None, rng_by_rank=None):
    path = Path(path)
    value = dict(kind=kind(student), stage=stage, step=step, parent=student.parent,
                 config=student.config, adapters=cpu_state(student.adapters.state_dict()),
                 bridge=cpu_state(student.bridge.state_dict()),
                 optimizer=cpu_state(optimizer.state_dict()) if optimizer else None,
                 sampler=sampler.state_dict() if sampler else None,
                 rng_by_rank=rng_by_rank or [rng_state(student.runtime.device)],
                 provenance=provenance or {})
    temporary = path.with_name(path.name+'.partial')
    torch.save(value, str(temporary))
    os.replace(temporary, path)


def apply(saved, student):
    if saved.get('kind') != kind(student):
        raise ValueError('Checkpoint architecture does not match this full-loop configuration')
    student.adapters.load_state_dict(saved['adapters'], strict=True)
    student.bridge.load_state_dict(saved['bridge'], strict=True)


def resume(saved, student, optimizer, sampler, provenance):
    from ..diagnostics.distributed import rank, world_size
    budget_keys = ('steps', 'save_steps', 'val_images')
    active = {k: v for k, v in student.config.items() if k not in budget_keys}
    previous = {k: v for k, v in saved['config'].items() if k not in budget_keys}
    # Model directories can move; parent native_files carry the model identity.
    stored_data = {k: v for k, v in saved['provenance'].items() if k != 'model'}
    current_data = {k: v for k, v in provenance.items() if k != 'model'}
    if stored_data != current_data or previous != active:
        raise ValueError('Resume data/config differs from the saved run')
    if saved['optimizer'] is None or len(saved['rng_by_rank']) != world_size():
        raise ValueError('Resume requires the optimizer and the same number of workers')
    optimizer.load_state_dict(saved['optimizer'])
    sampler.load_state_dict(saved['sampler'])
    restore_rng(saved['rng_by_rank'][rank()], student.runtime.device)
    return saved['step']
