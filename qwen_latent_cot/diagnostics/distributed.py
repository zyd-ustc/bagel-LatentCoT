"""Synchronous data parallelism for the small writer, never for frozen BAGEL."""
import os
from datetime import timedelta
import torch
import torch.distributed as dist
from ..bagel.accelerator import set_device


def initialize(device):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world < 2:
        raise ValueError('--distributed requires torchrun with at least two workers')
    kind = str(device).split(':')[0]
    if kind not in ('npu', 'cuda', 'cpu'):
        raise ValueError('distributed training requires an explicit device backend')
    selected = set_device(f'{kind}:{os.environ["LOCAL_RANK"]}' if kind != 'cpu' else 'cpu')
    dist.init_process_group({'npu': 'hccl', 'cuda': 'nccl', 'cpu': 'gloo'}[kind],
                            timeout=timedelta(minutes=30))
    return selected


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier():
    if dist.is_initialized():
        dist.barrier()


@torch.no_grad()
def broadcast_parameters(module):
    if world_size() == 1:
        return
    for parameter in module.parameters():
        dist.broadcast(parameter, src=0)


@torch.no_grad()
def sum_gradients(module):
    """Losses are divided by GLOBAL batch before backward; sum, do not average."""
    if world_size() == 1:
        return
    parameters = list(module.parameters())
    if any(parameter.grad is None for parameter in parameters):
        raise RuntimeError('every writer parameter must receive a gradient')
    flattened = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])
    dist.all_reduce(flattened, op=dist.ReduceOp.SUM)
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        parameter.grad.copy_(flattened[offset:offset+size].view_as(parameter))
        offset += size


def sum_statistics(values, device):
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    if world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def gather_samples(samples):
    if world_size() == 1:
        return samples
    gathered = [None] * world_size()
    dist.all_gather_object(gathered, samples)
    return [sample for shard in gathered for sample in shard]
