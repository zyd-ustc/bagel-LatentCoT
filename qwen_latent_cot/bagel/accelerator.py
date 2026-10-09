"""Explicit CUDA/Ascend device handling, based on the existing ModelArts port.

No transfer_to_npu monkey patch and no conversion of an explicit CUDA request
to another backend. CPU remains the numerical oracle, not a model fallback.
"""
from contextlib import nullcontext,contextmanager
import importlib
import torch


def resolve_device(spec='auto'):
    text=str(spec)
    if text=='auto':
        if torch.cuda.is_available():text='cuda:0'
        else:
            try:importlib.import_module('torch_npu')
            except ModuleNotFoundError as error:
                if error.name!='torch_npu':raise
            text='npu:0' if hasattr(torch,'npu') and torch.npu.is_available() else 'cpu'
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


def autocast_for(device):
    return torch.autocast(device.type,dtype=torch.bfloat16) if device.type in ('cuda','npu') else nullcontext()


@contextmanager
def seeded_context(device,seed):
    """Replay native posterior sampling without changing the caller's RNG.

    torch.manual_seed also seeds other accelerators. Set only CPU and the
    selected backend here; workers use one visible accelerator per process.
    """
    devices=[device.index if device.index is not None else getattr(torch,device.type).current_device()] if device.type!='cpu' else []
    with torch.random.fork_rng(devices=devices,device_type=device.type if devices else 'cuda'):
        torch.set_rng_state(torch.Generator().manual_seed(int(seed)).get_state())
        if devices:getattr(torch,device.type).manual_seed(int(seed))
        yield


def synchronize(device):
    if device.type!='cpu':getattr(torch,device.type).synchronize(device)


def reset_peak_memory_stats(device):
    if device.type!='cpu':getattr(torch,device.type).reset_peak_memory_stats(device)


def max_memory_allocated(device):
    return getattr(torch,device.type).max_memory_allocated(device) if device.type!='cpu' else 0


def device_info(device):
    info={'device_type':device.type,'device':str(device),'torch':torch.__version__,
        'name':getattr(torch,device.type).get_device_name(device) if device.type!='cpu' else 'CPU oracle',
        'attention_backend':{'cuda':'native_flash_attention','npu':'ascend_torch_sdpa_varlen','cpu':'float32_cpu_oracle'}[device.type]}
    if device.type=='npu':info['torch_npu']=importlib.import_module('torch_npu').__version__
    return info
