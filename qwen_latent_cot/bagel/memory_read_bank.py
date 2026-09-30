"""Detached, layer-matched native KV produced by a single strict Read."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch


@dataclass(frozen=True)
class LayerMemoryState:
    layer_idx: int
    hidden_entry: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor

    def __post_init__(self):
        if isinstance(self.layer_idx,bool) or not isinstance(self.layer_idx,int) or self.layer_idx < 0:
            raise ValueError("memory layer index must be a nonnegative integer")
        if self.hidden_entry.ndim != 2 or self.key.ndim != 3 or self.value.ndim != 3:
            raise ValueError("memory state requires hidden [K,D] and KV [K,Hkv,Dh]")
        if self.key.shape != self.value.shape or self.key.shape[0] != self.hidden_entry.shape[0]:
            raise ValueError("memory state slot counts or KV shapes differ")
        if self.key.shape[0] < 1 or self.key.dtype != self.value.dtype or len(
                {value.device for value in (self.hidden_entry,self.key,self.value)}) != 1:
            raise ValueError("memory bank requires nonempty, co-located native KV")
        if any(value.requires_grad for value in (self.hidden_entry, self.key, self.value)):
            raise ValueError("strict Read bank must be detached")

    def zero_like(self):
        return LayerMemoryState(self.layer_idx, torch.zeros_like(self.hidden_entry),
                                torch.zeros_like(self.key), torch.zeros_like(self.value))


@dataclass(frozen=True)
class MemoryReadBank:
    states: Mapping[int, LayerMemoryState]

    def __post_init__(self):
        if not self.states or any(index != state.layer_idx for index, state in self.states.items()):
            raise ValueError("Read bank must have nonempty, correctly indexed states")
        if len({(state.hidden_entry.shape,state.key.shape,state.key.device)
                for state in self.states.values()}) != 1:
            raise ValueError("Read bank layers must have the same slot/head geometry and device")
        object.__setattr__(self,"states",MappingProxyType(dict(self.states)))

    def require_layers(self, start, end):
        if set(self.states) != set(range(start, end)):
            raise ValueError(f"Read bank layers must be exactly [{start},{end})")
        return self

    def zero_like(self):
        return MemoryReadBank({index: state.zero_like() for index, state in self.states.items()})
