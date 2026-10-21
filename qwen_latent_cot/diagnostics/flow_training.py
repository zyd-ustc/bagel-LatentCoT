"""Differentiable native conditional GEN, with the SAME injection kernel as inference."""
from contextlib import nullcontext, contextmanager
from types import MethodType
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
    prompt_kv: tuple | None = None
    correction_target: torch.Tensor | None = None
    prepared_baseline: "FlowOutput | None" = None

    def flow_target(self):
        if self.x0 is None or self.epsilon is None or not self.metadata.get('flow_supervision', False):
            raise ValueError('trajectory has no paired epsilon-x0 flow target')
        return self.epsilon.float()-self.x0.float()


@dataclass
class FlowOutput:
    velocity: torch.Tensor
    source: tuple[KV, ...]
    round: int
    hidden: torch.Tensor | None = None


@contextmanager
def conditional_reentry(decoder, cache, image_indexes, hidden_input):
    """Replace image entrance only; observe the exit before final output norm."""
    original = decoder.forward_inference
    captured = {}
    def exit_hook(module, args, kwargs, result):
        if kwargs.get('mode') == 'gen' and kwargs.get('past_key_values') is cache:
            captured['hidden'] = result[0][image_indexes]
    def call(self, **kwargs):
        if kwargs.get('mode') == 'gen' and kwargs.get('past_key_values') is cache and hidden_input is not None:
            kwargs = dict(kwargs)
            hidden = kwargs['packed_query_sequence'].clone()
            hidden[image_indexes] = hidden_input.to(hidden.dtype)
            kwargs['packed_query_sequence'] = hidden
        return original(**kwargs)
    handle = decoder.layers[-1].register_forward_hook(exit_hook, with_kwargs=True)
    decoder.forward_inference = MethodType(call, decoder)
    try:
        yield captured
    finally:
        decoder.forward_inference = original
        handle.remove()


def conditional_gen(runtime, state, bank=None, *, masked=False, capture=True,
                    round_index=0, detach_source=True, checkpoint_activations=False,
                    adapter_scope=nullcontext, guidance=False, hidden_input=None,
                    velocity_baseline=None, velocity_scale=None, memory_gates=None, memory_gate_scale=1.0, native_scope=nullcontext):
    """One full-depth native flow call. CFG references never receive feedback.

    Reentry, Memory, adapter scopes and capture are recreated inside checkpoint
    recomputation. Hidden and captured KV are explicit differentiable outputs.
    """
    native = getattr(runtime.model._forward_flow, '__wrapped__', None)
    if native is None:
        raise RuntimeError('native _forward_flow no_grad wrapper contract changed')
    flow = state.flow
    source_positions = flow['packed_position_ids']+(bank.slots if bank is not None and memory_gates is None else 0)
    if int(source_positions.max()) >= runtime.config['writer_position_start']:
        raise ValueError('GEN positions overlap writer slots')
    metadata = []
    image_indexes = flow['packed_vae_token_indexes']

    def forward(x, entrance):
        with native_scope():
            return native_forward(x, entrance)

    def native_forward(x, entrance):
        # First llm2vae invocation is conditional; reference heads stay native.
        calls = 0
        def residual_head(module, args, value):
            nonlocal calls
            calls += 1
            if calls != 1 or velocity_baseline is None: return value
            adjusted = value.clone()
            raw = value[image_indexes].float()
            base = velocity_baseline.detach().float()
            adjusted[image_indexes] = (base+velocity_scale*(raw-base)).to(value.dtype)
            return adjusted
        handle = runtime.model.llm2vae.register_forward_hook(residual_head)
        context = (gen_memory_call(runtime.decoder, flow['past_key_values'], bank,
                                   state.lengths, state.positions, masked, adapter_scope=adapter_scope, memory_gates=memory_gates,
                                   memory_gate_scale=memory_gate_scale)
                   if bank is not None else nullcontext())
        observer = (capture_gen(runtime.decoder, flow['past_key_values'], t=state.t,
                                round_index=round_index, positions=source_positions,
                                detach=detach_source, source_policy='full_query')
                    if capture else nullcontext({}))
        try:
            with conditional_reentry(runtime.decoder, flow['past_key_values'], image_indexes, entrance) as exit_state:
                with context, observer as captured, autocast_for(runtime.device):
                    velocity = native(runtime.model, x_t=x,
                                      timestep=torch.full((len(x),), state.t, device=runtime.device),
                                      **flow, cfg_text_scale=runtime.config['cfg_text_scale'] if guidance else 1., cfg_img_scale=1.,
                                      cfg_renorm_min=runtime.config['cfg_renorm_min'],
                                      cfg_renorm_type=runtime.config['cfg_renorm_type'])
        finally:
            handle.remove()
        source = tuple(captured[i] for i in range(len(runtime.decoder.layers))) if capture else ()
        if not metadata:
            metadata.extend(source)
        return (velocity, exit_state['hidden'], *(value for kv in source for value in (kv.keys, kv.values)))

    outputs = (checkpoint(forward, state.x_t, hidden_input, use_reentrant=False, preserve_rng_state=True)
               if checkpoint_activations and torch.is_grad_enabled() and bank is not None
               else forward(state.x_t, hidden_input))
    if not torch.isfinite(outputs[0]).all():
        raise FloatingPointError('non-finite conditional GEN velocity')
    source = tuple(KV(kv.layer, outputs[2+2*i], outputs[3+2*i], kv.lengths,
                      kv.rotated, kv.source_kind, kv.t, kv.round, kv.token_roles,
                      kv.positions, kv.expert) for i, kv in enumerate(metadata))
    return FlowOutput(outputs[0], source, round_index, outputs[1])


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
