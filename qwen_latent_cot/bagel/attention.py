"""Native FlashAttention dispatch; a slow CPU oracle supports contract tests."""
import torch
from torch.nn.functional import scaled_dot_product_attention

try:
    from flash_attn import flash_attn_varlen_func as _flash
except ImportError:
    _flash = None


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q, max_seqlen_k, causal=False, **kwargs):
    if q.is_cuda:
        if _flash is None:
            raise RuntimeError('CUDA inference requires native flash-attn; no silent backend fallback')
        return _flash(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q,
                      cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q,
                      max_seqlen_k=max_seqlen_k, causal=causal, **kwargs)
    outputs = []
    for i in range(len(cu_seqlens_q)-1):
        qi = q[int(cu_seqlens_q[i]):int(cu_seqlens_q[i+1])].transpose(0, 1)[None].float()
        ki = k[int(cu_seqlens_k[i]):int(cu_seqlens_k[i+1])].transpose(0, 1)[None].float()
        vi = v[int(cu_seqlens_k[i]):int(cu_seqlens_k[i+1])].transpose(0, 1)[None].float()
        if not qi.shape[2]:
            continue
        groups = qi.shape[1] // ki.shape[1]
        ki = ki.repeat_interleave(groups, dim=1)
        vi = vi.repeat_interleave(groups, dim=1)
        mask = None
        if causal:
            # FlashAttention aligns causal masks to the bottom right for KV caches.
            nq, nk = qi.shape[2], ki.shape[2]
            mask = torch.arange(nk)[None, :] <= torch.arange(nq)[:, None] + nk-nq
        out = scaled_dot_product_attention(qi, ki, vi, attn_mask=mask)
        outputs.append(out[0].transpose(0, 1).to(q.dtype))
    return torch.cat(outputs) if outputs else q.new_empty(q.shape)
