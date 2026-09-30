"""Symmetric bank-only GEN readout with a zero-effect OPD injection gate."""

import math

import torch
from torch import nn


class ZeroEffectOutputAdapter(nn.Module):
    def __init__(self, hidden_size, *, rank=8, alpha=16):
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("memory output rank and alpha must be positive")
        self.A = nn.Linear(hidden_size, rank, bias=False, dtype=torch.float32)
        self.B = nn.Linear(rank, hidden_size, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.scale = float(alpha) / rank

    def forward(self, value):
        return (self.B(self.A(value.float())) * self.scale).to(value.dtype)


def bank_attention(gen_query, key, value, *, num_heads, num_kv_heads, head_dim):
    """Native GQA head repetition, independent bank-only softmax (B=1)."""
    if (gen_query.ndim != 3 or key.ndim != 3 or value.shape != key.shape
            or gen_query.shape[1:] != (num_heads, head_dim)
            or key.shape[1:] != (num_kv_heads, head_dim) or key.shape[0] < 1
            or num_heads % num_kv_heads):
        raise ValueError("invalid bank-only Q/K/V geometry")
    repeats = num_heads // num_kv_heads
    k = key.repeat_interleave(repeats, dim=1).float()
    v = value.repeat_interleave(repeats, dim=1).float()
    weights = (torch.einsum("qhd,khd->hqk", gen_query.float(), k)
               / math.sqrt(head_dim)).softmax(dim=-1)
    attended = torch.einsum("hqk,khd->qhd", weights, v)
    return attended.reshape(gen_query.shape[0], num_heads * head_dim), weights


class GenMemoryReader(nn.Module):
    """Translate native memory-bank readout; OPD injects through a scalar gate."""

    def __init__(self, *, native_gen_o_proj, num_heads, num_kv_heads, head_dim,
                 o_rank=8, o_alpha=16):
        super().__init__()
        hidden = num_heads * head_dim
        self.output_adapter = ZeroEffectOutputAdapter(hidden, rank=o_rank, alpha=o_alpha)
        self.injection_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        object.__setattr__(self, "native_gen_o_proj", native_gen_o_proj)
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.last_output_rms = 0.0
        self.last_native_attn_rms = 0.0
        self.last_attention_entropy = 0.0
        self.last_attention_max = 0.0
        self.last_slot_effective_count = 0.0
        self.last_slot_mass_max = 0.0
        self.last_adapter_rms = 0.0

    def _read(self, gen_query, key, value):
        attended, weights = bank_attention(gen_query, key, value,
            num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim)
        native = self.native_gen_o_proj(attended.to(value.dtype))
        adapter = self.output_adapter(attended)
        self.last_adapter_rms = float(adapter.detach().float().square().mean().sqrt())
        # Preserve small translation updates in the side-head MSE. Native O
        # retains its BF16 numerics; injection is cast to the native stream.
        return native.float() + adapter.float(), weights

    def forward(self, *, gen_query, memory_key, memory_value):
        result, weights = self._read(gen_query, memory_key, memory_value)
        self.last_output_rms = float(result.detach().float().square().mean().sqrt())
        probability = weights.detach().float()
        self.last_attention_entropy = float(-(probability * probability.clamp_min(1e-12).log()).sum(dim=-1).mean())
        self.last_attention_max = float(probability.amax(dim=-1).mean())
        slot_mass = probability.mean(dim=(0,1))
        self.last_slot_effective_count = float((-(slot_mass * slot_mass.clamp_min(1e-12).log()).sum()).exp())
        self.last_slot_mass_max = float(slot_mass.max())
        return result

    def diagnostics(self):
        return dict(memory_readout_rms=self.last_output_rms,
            adapter_residual_rms=self.last_adapter_rms,
            attention_entropy=self.last_attention_entropy,
            attention_max=self.last_attention_max,
            slot_effective_count=self.last_slot_effective_count,
            slot_mass_max=self.last_slot_mass_max,
            injection_gate=float(self.injection_gate.detach()))

    def prompt_target(self, *, gen_query, prompt_key, prompt_value):
        with torch.no_grad():
            attended, _ = bank_attention(gen_query.detach(), prompt_key.detach(),
                prompt_value.detach(), num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads, head_dim=self.head_dim)
            return self.native_gen_o_proj(attended.to(prompt_value.dtype)).detach()


def install_memory_readers(model, *, start=12, end=20, rank=8, alpha=16,
                           stage="warmup"):
    decoder = model.language_model.model
    if not 0 <= start < end <= len(decoder.layers):
        raise ValueError("invalid MemoryReader body range")
    if stage not in ("warmup", "opd"):
        raise ValueError("reader stage must be warmup or opd")
    for layer_idx in range(start, end):
        layer = decoder.layers[layer_idx]
        attention = layer.self_attn
        if not all(hasattr(attention, name) for name in
                   ("q_proj_moe_gen", "o_proj_moe_gen", "k_proj", "v_proj")):
            raise ValueError(f"layer {layer_idx} is not a MoT attention layer")
        layer.memory_reader = GenMemoryReader(
            native_gen_o_proj=attention.o_proj_moe_gen,
            num_heads=attention.num_heads, num_kv_heads=attention.num_key_value_heads,
            head_dim=attention.head_dim, o_rank=rank, o_alpha=alpha)
    model.requires_grad_(False)
    for layer in decoder.layers[start:end]:
        if stage == "warmup":
            layer.memory_reader.output_adapter.requires_grad_(True)
        else:
            layer.memory_reader.injection_gate.requires_grad_(True)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
