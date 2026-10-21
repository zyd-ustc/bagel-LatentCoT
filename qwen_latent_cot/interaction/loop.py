"""Full-depth dual-state recurrence at a fixed noise state and flow time."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from ..diagnostics.flow_training import conditional_gen
from ..diagnostics.feedback_source import prompt_source
from ..diagnostics.schedule import native_times
from ..diagnostics.numerics import autocast_for


def active_updates(config):
    return math.ceil((len(native_times(config))-1)/2)


def training_times(config):
    return native_times(config)[:active_updates(config)]


class ReentryBridge(nn.Module):
    def __init__(self, width, device):
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False, device=device, dtype=torch.float32)
        nn.init.eye_(self.projection.weight)
        self.strength = nn.Parameter(torch.full((), 0.15, device=device))

    def forward(self, anchor, previous):
        direction = self.projection(previous.float()).float()
        rms = lambda x: x.float().square().mean().sqrt()
        # One image-wide scale preserves relative energy between spatial patches.
        direction = direction/rms(direction).clamp_min(1e-6)*rms(anchor).detach()
        return (anchor.float()+self.strength*direction).to(anchor.dtype)


@dataclass
class LoopOutput:
    baseline: object
    exits: list
    banks: list

    @property
    def velocity(self):
        return self.exits[-1].velocity


def image_entrance(runtime, state):
    model = runtime.model
    with autocast_for(runtime.device):
        return (model.vae2llm(state.x_t)
                + model.time_embedder(torch.full((len(state.x_t),), state.t, device=runtime.device))
                + model.latent_pos_embed(state.flow['packed_vae_position_ids']))


def full_loop(student, state, *, guidance=False, feedback_updates=None, memory_masked=False, reentry=True):
    runtime = student.runtime
    if (feedback_updates is not None or memory_masked or not reentry) and torch.is_grad_enabled():
        raise ValueError('feedback_updates override is inference-only')
    updates = student.config['feedback_updates'] if feedback_updates is None else feedback_updates
    if not isinstance(updates, int) or isinstance(updates, bool) or updates < 1:
        raise ValueError('feedback_updates must be a positive integer; use native sampling for base')
    with torch.no_grad():
        baseline = conditional_gen(runtime, state, detach_source=True)
        anchor = image_entrance(runtime, state)
    prompt = state.prompt_kv if state.prompt_kv is not None else prompt_source(state.flow, state.lengths, state.positions)
    previous = baseline
    banks, exits = [], []
    for index in range(updates):
        bank = student.write(previous.source, state.t, prompt=prompt)
        with autocast_for(runtime.device):
            entrance = student.bridge(anchor, previous.hidden) if reentry else None
        output = conditional_gen(runtime, state, bank, masked=memory_masked, round_index=index+1,
                                 hidden_input=entrance, detach_source=False,
                                 capture=index < updates-1,
                                 adapter_scope=student.adapters.context('consumer'),
                                 guidance=guidance and index == updates-1,
                                 checkpoint_activations=student.config['activation_checkpointing'],
                                 velocity_baseline=baseline.velocity if hasattr(student.bridge, 'vel_scale') else None,
                                 velocity_scale=student.bridge.vel_scale[min(index, len(student.bridge.vel_scale)-1)] if hasattr(student.bridge, 'vel_scale') else None)
        banks.append(bank)
        exits.append(output)
        previous = output
    return LoopOutput(baseline, exits, banks)


@dataclass
class LoopLoss:
    total: torch.Tensor
    round_flow: list
    round_weighted_flow: list
    round_keep: list


def patch_weights(baseline, target, hard_patch_alpha):
    """Detached error-based weights; error is not a semantic/texture label."""
    with torch.no_grad():
        error = (baseline.float()-target.float()).square().mean(-1)
        relative = error/error.mean().clamp_min(1e-8)
        hard = 1+hard_patch_alpha*relative
        hard = hard/hard.mean()
        keep = 1/(1+relative)
    return hard, keep


def supervised_loss(output, state, weights, *, hard_patch_alpha=1.0, keep_weight=0.02):
    if len(weights) != len(output.exits):
        raise ValueError('Every feedback output needs a supervision weight')
    if hard_patch_alpha < 0 or keep_weight < 0:
        raise ValueError('Loss coefficients must be nonnegative')
    target = state.flow_target().detach()
    baseline = output.baseline.velocity.detach().float()
    hard, keep = patch_weights(baseline, target, hard_patch_alpha)
    raw, weighted, fidelity = [], [], []
    for value in output.exits:
        error = (value.velocity.float()-target).square().mean(-1)
        drift = (value.velocity.float()-baseline).square().mean(-1)
        raw.append(error.mean())
        weighted.append((hard*error).mean())
        fidelity.append((keep*drift).mean())
    total = sum(w*(flow+keep_weight*anchor) for w,flow,anchor in zip(weights,weighted,fidelity))
    return LoopLoss(total,raw,weighted,fidelity)
