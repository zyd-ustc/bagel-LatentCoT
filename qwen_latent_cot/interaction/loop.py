"""Full-depth dual-state recurrence at a fixed noise state and flow time."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from ..bagel.modeling.qwen2.modeling_qwen2 import Qwen2RMSNorm
from ..diagnostics.flow_training import conditional_gen, flow_mse
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
        self.norm = Qwen2RMSNorm(width, eps=1e-6).to(device=device, dtype=torch.float32)
        self.projection = nn.Linear(width, width, bias=False, device=device, dtype=torch.float32)
        nn.init.eye_(self.projection.weight)
        self.strength = nn.Parameter(torch.ones((), device=device))

    def forward(self, anchor, previous):
        direction = self.projection(self.norm(previous.float())).float()
        rms = lambda x: x.float().square().mean(-1, keepdim=True).sqrt()
        # Match units, not a small residual cap. Preserve local noise/time/position.
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


def full_loop(student, state, *, guidance=False):
    runtime = student.runtime
    with torch.no_grad():
        baseline = conditional_gen(runtime, state, detach_source=True)
        anchor = image_entrance(runtime, state)
    prompt = state.prompt_kv if state.prompt_kv is not None else prompt_source(state.flow, state.lengths, state.positions)
    previous = baseline
    banks, exits = [], []
    for index in range(student.config['feedback_updates']):
        bank = student.write(previous.source, state.t, prompt=prompt)
        with autocast_for(runtime.device):
            entrance = student.bridge(anchor, previous.hidden)
        output = conditional_gen(runtime, state, bank, round_index=index+1,
                                 hidden_input=entrance, detach_source=False,
                                 capture=index < student.config['feedback_updates']-1,
                                 adapter_scope=student.adapters.context('consumer'),
                                 guidance=guidance and index == student.config['feedback_updates']-1,
                                 checkpoint_activations=student.config['activation_checkpointing'])
        banks.append(bank)
        exits.append(output)
        previous = output
    return LoopOutput(baseline, exits, banks)


def supervised_loss(output, state, weights):
    if len(weights) != len(output.exits):
        raise ValueError('Every feedback output needs a supervision weight')
    losses = [flow_mse(value, state) for value in output.exits]
    return sum(w*loss for w, loss in zip(weights, losses)), losses
