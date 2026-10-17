"""Expert-specific LoRA with explicit forward/recomputation scope."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
import math
import torch
from torch import nn
from torch.nn import functional as F


class LowRank(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, linear.in_features, device=linear.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(linear.out_features, rank, device=linear.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.register_buffer('reference_a', None, persistent=False)
        self.register_buffer('reference_b', None, persistent=False)

    def forward(self, x, mode):
        a, b = (self.reference_a, self.reference_b) if mode == 'reference' else (self.a, self.b)
        if a is None or b is None:
            raise RuntimeError('Reference policy has not been initialized')
        return F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype)) * self.scale

    @torch.no_grad()
    def snapshot_reference(self):
        self.reference_a = self.a.detach().clone()
        self.reference_b = self.b.detach().clone()


class ScopedLinear(nn.Module):
    def __init__(self, base, update, scope):
        super().__init__()
        self.base, self.update = base, update
        object.__setattr__(self, 'scope', scope)
        self.in_features, self.out_features = base.in_features, base.out_features

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        result = self.base(x)
        mode = self.scope.get()
        if mode != 'off':
            result = result + self.update(x, mode).to(result.dtype)
        return result


class Adapters(nn.Module):
    """Register only LoRA tensors in the student, never the frozen BAGEL."""
    def __init__(self, decoder, rank=16, alpha=16):
        super().__init__()
        self.writer = nn.ModuleDict()
        self.consumer = nn.ModuleDict()
        self.scopes = {name: ContextVar(f'bagel_{id(self)}_{name}', default='off') for name in ('writer', 'consumer')}
        for index, layer in enumerate(decoder.layers):
            attention = layer.self_attn
            for group, suffix in (('writer', ''), ('consumer', '_moe_gen')):
                for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                    base = getattr(attention, name+suffix)
                    if isinstance(base, ScopedLinear):
                        raise ValueError('Adapters are already installed on this decoder')
                    update = LowRank(base, rank, alpha)
                    getattr(self, group)[f'{index:02d}_{name}'] = update
                    setattr(attention, name+suffix, ScopedLinear(base, update, self.scopes[group]))

    @contextmanager
    def scope(self, group, mode='policy'):
        if mode not in ('off', 'policy', 'reference'):
            raise ValueError(mode)
        token = self.scopes[group].set(mode)
        try:
            yield
        finally:
            self.scopes[group].reset(token)

    def context(self, group, mode='policy'):
        # Capture the mode NOW; backward may run after a reference scope ends.
        return partial(self.scope, group, mode)

    def freeze_writer(self):
        self.writer.requires_grad_(False)

    def snapshot_reference(self):
        for update in self.consumer.values():
            update.snapshot_reference()

    def reference_state(self):
        return {name: dict(a=u.reference_a.detach().cpu(), b=u.reference_b.detach().cpu())
                for name, u in self.consumer.items() if u.reference_a is not None}

    def load_reference(self, state):
        for name, values in state.items():
            update = self.consumer[name]
            update.reference_a = values['a'].to(update.a.device)
            update.reference_b = values['b'].to(update.b.device)
