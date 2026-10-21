"""Scoped Write-KV LoRA; Stage A consumer modifies only independent Memory queries."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
import math
import torch
from torch import nn
from torch.nn import functional as F


class LowRank(nn.Module):
    def __init__(self, linear, rank):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, linear.in_features, device=linear.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(linear.out_features, rank, device=linear.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(F.linear(x, self.a.to(x.dtype)), self.b.to(x.dtype))


class ScopedLinear(nn.Module):
    def __init__(self, base, update, scope):
        super().__init__()
        self.base = base
        object.__setattr__(self, 'update', update)
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
        return result + self.update(x).to(result.dtype) if self.scope.get() else result


STAGE_A_CONSUMER_TARGETS = ("q_proj_moe_gen",)


class Adapters(nn.Module):
    def __init__(self, decoder, writer_rank, consumer_rank, *, stage_a=False):
        super().__init__()
        self.writer = nn.ModuleDict()
        self.consumer = nn.ModuleDict()
        self.scopes = {name: ContextVar(f'bagel_{id(self)}_{name}', default=False)
                       for name in ('writer', 'consumer')}
        for index, layer in enumerate(decoder.layers):
            for group, suffix, names, rank in (
                ('writer', '', ('k_proj', 'v_proj'), writer_rank),
                ('consumer', '_moe_gen', tuple(name.removesuffix('_moe_gen') for name in STAGE_A_CONSUMER_TARGETS) if stage_a else ('q_proj', 'v_proj', 'o_proj'), consumer_rank)):
                for name in names:
                    base = getattr(layer.self_attn, name+suffix)
                    if isinstance(base, ScopedLinear):
                        raise ValueError('Adapters already installed')
                    update = LowRank(base, rank)
                    getattr(self, group)[f'{index:02d}_{name}'] = update
                    setattr(layer.self_attn, name+suffix, ScopedLinear(base, update, self.scopes[group]))

    @contextmanager
    def scope(self, group, enabled=True):
        token = self.scopes[group].set(enabled)
        try:
            yield
        finally:
            self.scopes[group].reset(token)

    def context(self, group, enabled=True):
        return partial(self.scope, group, enabled)
