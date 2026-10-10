"""Project the native mixed GEN query's layer-input KV, including boundary tokens."""
import torch
from .modeling.qwen2.modeling_qwen2 import apply_rotary_pos_emb


def project_gen(layer, hidden, rope, text_indexes, image_indexes):
    attention = layer.self_attn
    normalized = torch.zeros_like(hidden)
    normalized[text_indexes] = layer.input_layernorm(hidden[text_indexes])
    normalized[image_indexes] = layer.input_layernorm_moe_gen(hidden[image_indexes])
    normalized = normalized.to(torch.bfloat16)
    shape = (len(hidden), attention.num_key_value_heads * attention.head_dim)
    keys = normalized.new_zeros(shape)
    values = normalized.new_zeros(shape)
    keys[text_indexes] = attention.k_proj(normalized[text_indexes])
    keys[image_indexes] = attention.k_proj_moe_gen(normalized[image_indexes])
    values[text_indexes] = attention.v_proj(normalized[text_indexes])
    values[image_indexes] = attention.v_proj_moe_gen(normalized[image_indexes])
    keys = keys.view(-1, attention.num_key_value_heads, attention.head_dim).float()
    values = values.view(-1, attention.num_key_value_heads, attention.head_dim)
    keys[text_indexes] = attention.k_norm(keys[text_indexes])
    keys[image_indexes] = attention.k_norm_moe_gen(keys[image_indexes])
    _, keys = apply_rotary_pos_emb(torch.zeros_like(keys), keys, *rope, unsqueeze_dim=1)
    return keys.to(torch.bfloat16), values.to(torch.bfloat16)
