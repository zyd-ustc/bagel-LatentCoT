"""Memory-only GEN residual: pretrained Q/K/V geometry, trainable zero-effect O."""

import math
import torch
from torch import nn
from torch.nn import functional as F


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


class GenMemoryReader(nn.Module):
    """Cross-attend GEN queries only to frozen Read-memory slots.

    Native projections are referenced, not registered/copied. The returned
    residual is added after native attention; native softmax/KV is untouched.
    """
    def __init__(self, *, native_gen_q_proj, native_und_k_proj,
                 native_und_v_proj, num_heads, num_kv_heads, head_dim,
                 q_norm=None, k_norm=None, o_rank=8, o_alpha=16,
                 train_q_lora=False):
        super().__init__()
        if train_q_lora:
            raise ValueError("T0 only supports frozen native GEN-Q; Q_mem LoRA is Phase 1A.1")
        if num_heads % num_kv_heads:
            raise ValueError("GEN heads must be divisible by UND KV heads")
        hidden = num_heads * head_dim
        self.output_adapter = ZeroEffectOutputAdapter(hidden, rank=o_rank, alpha=o_alpha)
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        for key, value in (("native_gen_q_proj", native_gen_q_proj),
                           ("native_und_k_proj", native_und_k_proj),
                           ("native_und_v_proj", native_und_v_proj),
                           ("q_norm", q_norm or nn.Identity()),
                           ("k_norm", k_norm or nn.Identity())):
            object.__setattr__(self, key, value)
        self.last_output_rms = 0.0
        self.last_native_attn_rms = 0.0
        self.last_attention_entropy = 0.0
        self.last_attention_max = 0.0

    def forward(self, gen_hidden, memory_hidden):
        if gen_hidden.ndim != 2 or memory_hidden.ndim != 2 or memory_hidden.shape[0] < 1:
            raise ValueError("reader needs [GEN,D] and nonempty [K,D] memory")
        if gen_hidden.shape[-1] != memory_hidden.shape[-1]:
            raise ValueError("GEN/memory hidden widths differ")
        q = self.q_norm(self.native_gen_q_proj(gen_hidden).reshape(-1, self.num_heads,
                                                                  self.head_dim))
        k = self.k_norm(self.native_und_k_proj(memory_hidden).reshape(-1, self.num_kv_heads,
                                                                       self.head_dim))
        v = self.native_und_v_proj(memory_hidden).reshape(-1, self.num_kv_heads,
                                                        self.head_dim)
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1).float()
        v = v.repeat_interleave(repeats, dim=1).float()
        scores = torch.einsum("qhd,khd->hqk", q.float(), k) / math.sqrt(self.head_dim)
        weights = scores.softmax(dim=-1)
        attended = torch.einsum("hqk,khd->qhd", weights, v).reshape(-1, gen_hidden.shape[-1])
        delta = self.output_adapter(attended).to(gen_hidden.dtype)
        if torch.is_grad_enabled():
            self.last_output_rms = float(delta.detach().float().square().mean().sqrt())
            probability = weights.detach().float()
            self.last_attention_entropy = float(-(probability * probability.clamp_min(1e-12).log()).sum(dim=-1).mean())
            self.last_attention_max = float(probability.amax(dim=-1).mean())
        return delta


def install_memory_readers(model, *, start=12, end=20, rank=8, alpha=16):
    """Install only the dedicated trainable branch on the selected MoT layers."""
    decoder = model.language_model.model
    if not 0 <= start < end <= len(decoder.layers):
        raise ValueError("invalid MemoryReader body range")
    for layer_idx, layer in enumerate(decoder.layers):
        if not start <= layer_idx < end:
            continue
        attention = layer.self_attn
        if not all(hasattr(attention, name) for name in
                   ("q_proj_moe_gen", "k_proj", "v_proj")):
            raise ValueError(f"layer {layer_idx} is not a MoT attention layer")
        layer.memory_reader = GenMemoryReader(
            native_gen_q_proj=attention.q_proj_moe_gen,
            native_und_k_proj=attention.k_proj,
            native_und_v_proj=attention.v_proj,
            q_norm=attention.q_norm_moe_gen,
            k_norm=attention.k_norm,
            num_heads=attention.num_heads, num_kv_heads=attention.num_key_value_heads,
            head_dim=attention.head_dim, o_rank=rank, o_alpha=alpha)
    model.requires_grad_(False)
    for layer in decoder.layers[start:end]:
        layer.memory_reader.output_adapter.requires_grad_(True)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
