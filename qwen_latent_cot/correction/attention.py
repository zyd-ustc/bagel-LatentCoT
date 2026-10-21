"""Spatial UND attention over frozen prompt/GEN context and current patch tokens."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .numerics import autocast_for


@dataclass
class Context:
    keys: torch.Tensor
    values: torch.Tensor


def rotate(value, rope):
    cos, sin = rope
    half = value.shape[-1]//2
    orthogonal = torch.cat((-value[..., half:], value[..., :half]), -1)
    return (value*cos[:, None]+orthogonal*sin[:, None]).to(value.dtype)


def rope_for(decoder, hidden, positions):
    cos, sin = decoder.rotary_emb(hidden, positions.unsqueeze(0))
    return cos.squeeze(0), sin.squeeze(0)


def project(attention, hidden, *, generation=False):
    suffix = '_moe_gen' if generation else ''
    q = getattr(attention, 'q_proj'+suffix)(hidden).view(-1, attention.num_heads, attention.head_dim)
    k = getattr(attention, 'k_proj'+suffix)(hidden).view(-1, attention.num_key_value_heads, attention.head_dim)
    v = getattr(attention, 'v_proj'+suffix)(hidden).view_as(k)
    q = getattr(attention, 'q_norm'+suffix)(q.float())
    k = getattr(attention, 'k_norm'+suffix)(k.float())
    return q, k, v


def attend(q, k, v):
    groups = q.shape[1]//k.shape[1]
    q = q.transpose(0, 1)[None]
    k = k.transpose(0, 1)[None].repeat_interleave(groups, 1)
    v = v.transpose(0, 1)[None].repeat_interleave(groups, 1)
    if q.device.type == 'cpu': q, k, v = q.float(), k.float(), v.float()
    with torch.autocast('cpu', enabled=False) if q.device.type == 'cpu' else autocast_for(q.device):
        result = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
    return result[0].transpose(0, 1).reshape(q.shape[2], -1)


def spatial_block(layer, hidden, rope, context):
    attention = layer.self_attn
    q, k, v = project(attention, layer.input_layernorm(hidden))
    q, k = rotate(q, rope).to(hidden.dtype), rotate(k, rope).to(hidden.dtype)
    keys = torch.cat((context.keys.to(k.dtype), k))
    values = torch.cat((context.values.to(v.dtype), v))
    hidden = hidden+attention.o_proj(attend(q, keys, values).to(hidden.dtype))
    return hidden+layer.mlp(layer.post_attention_layernorm(hidden))
