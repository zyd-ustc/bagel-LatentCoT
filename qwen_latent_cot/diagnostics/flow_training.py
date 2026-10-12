"""Differentiable native conditional GEN, with the SAME injection kernel as inference."""
from contextlib import nullcontext
from dataclasses import dataclass
import torch
from torch.utils.checkpoint import checkpoint
from .native import capture_gen, gen_memory_call
from .memory import KV
from .numerics import autocast_for


@dataclass
class TrainingState:
    x_t: torch.Tensor
    x0: torch.Tensor | None
    epsilon: torch.Tensor | None
    t: float
    flow: dict
    lengths: tuple
    positions: list
    metadata: dict

    def flow_target(self):
        if self.x0 is None or self.epsilon is None:
            raise ValueError('trajectory has no paired epsilon-x0 flow target')
        return self.epsilon.float()-self.x0.float()


@dataclass
class FlowOutput:
    velocity: torch.Tensor
    source: tuple[KV, ...]
    round: int


def conditional_gen(runtime, state, bank=None, *, masked=False, capture=True,
                    round_index=0, detach_source=True, checkpoint_activations=False):
    """Unwrap ONLY native no_grad; all time/position/norm/heads stay native.

    Checkpoint closure owns its contexts: backward recomputation re-enters the
    same injection kernel, even after the original forward context has closed.
    Captured K/V are explicit checkpoint outputs, never a mutable backward cache.
    """
    native = getattr(runtime.model._forward_flow, '__wrapped__', None)
    if native is None:
        raise RuntimeError('native _forward_flow no_grad wrapper contract changed')
    flow = state.flow
    source_positions = flow['packed_position_ids']+(bank.slots if bank is not None else 0)
    if int(source_positions.max()) >= runtime.config['writer_position_start']:
        raise ValueError('GEN positions overlap writer slots')
    metadata = []

    def forward(x):
        context = (gen_memory_call(runtime.decoder, flow['past_key_values'], bank,
                                   state.lengths, state.positions, masked)
                   if bank is not None else nullcontext())
        observer = (capture_gen(runtime.decoder, flow['past_key_values'], t=state.t,
                                round_index=round_index, positions=source_positions,
                                detach=detach_source,
                                source_policy=runtime.config.get('visual_source_policy', 'full_query'))
                    if capture else nullcontext({}))
        with context, observer as captured, autocast_for(runtime.device):
            velocity = native(runtime.model, x_t=x,
                              timestep=torch.full((len(x),), state.t, device=runtime.device),
                              **flow, cfg_text_scale=1., cfg_img_scale=1.,
                              cfg_renorm_min=runtime.config['cfg_renorm_min'],
                              cfg_renorm_type=runtime.config['cfg_renorm_type'])
        source = tuple(captured[i] for i in range(len(runtime.decoder.layers))) if capture else ()
        if not metadata:
            metadata.extend(source)
        return (velocity, *(value for kv in source for value in (kv.keys, kv.values)))

    outputs = (checkpoint(forward, state.x_t, use_reentrant=False, preserve_rng_state=True)
               if checkpoint_activations and torch.is_grad_enabled() and bank is not None
               else forward(state.x_t))
    if not torch.isfinite(outputs[0]).all():
        raise FloatingPointError('non-finite conditional GEN velocity')
    source = tuple(KV(kv.layer, outputs[1+2*i], outputs[2+2*i], kv.lengths,
                      kv.rotated, kv.source_kind, kv.t, kv.round, kv.token_roles,
                      kv.positions, kv.expert) for i, kv in enumerate(metadata))
    return FlowOutput(outputs[0], source, round_index)


def flow_mse(output, state):
    return (output.velocity.float()-state.flow_target()).square().mean()


class ExecutionCounter:
    """Count actual decoder/layer entries, including checkpoint recomputation."""
    def __init__(self, runtime):
        self.counts = [0, 0, 0]  # GEN entries, GEN layer entries, UND layer entries
        self.handles = []
        self.layers = runtime.decoder.layers
        def decoder_hook(module, args, kwargs):
            if kwargs.get('mode') == 'gen': self.counts[0] += 1
        def layer_hook(module, args, kwargs):
            self.counts[1 if kwargs.get('mode') == 'gen' else 2] += 1
        self.handles.append(runtime.decoder.register_forward_pre_hook(decoder_hook, with_kwargs=True))
        for layer in self.layers:
            if hasattr(layer, '_diagnostic_cost_counter'):
                raise RuntimeError('cost counter already attached')
            object.__setattr__(layer, '_diagnostic_cost_counter', self)
            self.handles.append(layer.register_forward_pre_hook(layer_hook, with_kwargs=True))

    def close(self):
        for handle in self.handles: handle.remove()
        for layer in self.layers: object.__delattr__(layer, '_diagnostic_cost_counter')
