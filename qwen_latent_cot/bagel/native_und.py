"""Project UND hidden states with the native layer's parameters and RoPE."""
import torch


def project_und(layer, hidden, rope, query=False):
    from .modeling.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    attention=layer.self_attn
    normalized=layer.input_layernorm(hidden)
    keys=attention.k_norm(attention.k_proj(normalized).view(-1,attention.num_key_value_heads,attention.head_dim))
    values=attention.v_proj(normalized).view(-1,attention.num_key_value_heads,attention.head_dim)
    queries=(attention.q_norm(attention.q_proj(normalized).view(-1,attention.num_heads,attention.head_dim))
             if query else torch.zeros_like(keys))
    queries,keys=apply_rotary_pos_emb(queries,keys,*rope,unsqueeze_dim=1)
    return queries.to(torch.bfloat16) if query else None,keys.to(torch.bfloat16),values.to(torch.bfloat16)
