"""DDP helpers for frozen-native Stage-1 training."""

import os
from dataclasses import replace

import torch
from torch import distributed as dist, nn
from torch.nn.parallel import DistributedDataParallel

from .accelerator import resolve_device
from .anchored_loop import direct_flow_loss
from .depth_curriculum import sample_curriculum_depth


class Stage1Objective(nn.Module):
    """Register only trainable loop modules, and expose their loss to DDP.

    Native weights are frozen and independently loaded from the same checkpoint.
    Keeping them outside this wrapper avoids broadcasting the 7B native model.
    Diagnostics detach before returning so unused-parameter detection traverses
    only the actual loss, rather than readouts that are not used by the loss.
    """

    def __init__(self, model):
        super().__init__()
        self.loop = model.t2i_loop
        self.native_forward = model.forward_t2i_loop

    def forward(self, *, target, loop_config, **inputs):
        result = self.native_forward(loop_config=loop_config, **inputs)
        loss = direct_flow_loss(result, target, loop_config)
        diagnostics = replace(
            result,
            velocity=result.velocity.detach(),
            base_velocity=None
            if result.base_velocity is None
            else result.base_velocity.detach(),
            velocities=[value.detach() for value in result.velocities],
        )
        return loss, diagnostics


def initialize_stage1_distributed(device_spec):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = resolve_device(device_spec)
    if world_size > 1:
        if device.type == "cuda":
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
        elif device.type != "cpu":
            raise ValueError("Stage-1 DDP supports CUDA/NCCL and CPU/Gloo")
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    elif device.type == "cuda":
        torch.cuda.set_device(device)
    return device, rank, world_size


def stage1_objective(model, device, world_size):
    objective = Stage1Objective(model)
    if world_size > 1:
        objective = DistributedDataParallel(
            objective,
            device_ids=[device.index] if device.type == "cuda" else None,
            # Dynamic loop depth can leave adapter parameters unused at R1.
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
    return objective


def synchronized_depth(phases, step, device, rank, world_size):
    depth, phase = sample_curriculum_depth(phases, step) if rank == 0 else (0, 0)
    if world_size > 1:
        values = torch.tensor([depth, phase], device=device, dtype=torch.long)
        dist.broadcast(values, src=0)
        depth, phase = values.tolist()
    return depth, phase


def token_weighted_backward_loss(loss, token_count, device, world_size):
    """DDP averages gradients; scale to the global mean over packed tokens.

    Samples can have different aspect ratios and token counts on each rank.
    The feature width and deep-supervision weights are the same on every rank.
    """
    if world_size == 1:
        return loss
    counts = torch.tensor(token_count, device=device, dtype=torch.long)
    dist.all_reduce(counts)
    return loss * (world_size * token_count / int(counts))


def gather_rank_metrics(metrics, rank, world_size):
    if world_size == 1:
        return [metrics]
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(metrics, gathered, dst=0)
    return gathered
