"""Select the local PyTorch accelerator."""

import importlib
import torch


def resolve_device(spec='auto'):
    text=str(spec)
    if text=='auto':
        try:importlib.import_module('torch_npu')
        except ModuleNotFoundError as error:
            if error.name!='torch_npu':raise
        if hasattr(torch,'npu') and torch.npu.is_available():text='npu:0'
        elif torch.cuda.is_available():text='cuda:0'
        else:text='cpu'
    if text.startswith('npu'):
        importlib.import_module('torch_npu')
    device=torch.device(text)
    if device.type not in ('cpu','cuda','npu'):raise ValueError('unsupported execution backend: '+text)
    if device.type!='cpu' and not getattr(torch,device.type).is_available():
        raise RuntimeError('requested backend unavailable: '+text)
    return device

def set_device(device):
    device=resolve_device(device)
    if device.type!='cpu':getattr(torch,device.type).set_device(device)
    return device
