"""CUDA FlashAttention; Ascend SDPA from the old port; float32 CPU oracle."""
import torch
from torch.nn.functional import scaled_dot_product_attention

try:
    from flash_attn import flash_attn_varlen_func as _flash
except ImportError:
    _flash = None


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q, max_seqlen_k, causal=False, **kwargs):
    if q.device.type=='cuda':
        if _flash is None:
            raise RuntimeError('CUDA inference requires native flash-attn; no silent backend fallback')
        return _flash(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q,
                      cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q,
                      max_seqlen_k=max_seqlen_k, causal=causal, **kwargs)
    if q.device.type not in ('cpu','npu'):raise ValueError('unsupported attention device')
    if kwargs.get('dropout_p',0.)!=0.:raise ValueError('inference attention requires zero dropout')
    if set(kwargs)-{'dropout_p','softmax_scale'}:raise ValueError('unsupported attention options')
    qends=cu_seqlens_q.tolist();kends=cu_seqlens_k.tolist()
    if len(qends)!=len(kends) or qends[0]!=0 or kends[0]!=0 or qends[-1]!=len(q) or kends[-1]!=len(k):
        raise ValueError('packed attention lengths do not cover Q/K')
    if k.shape!=v.shape or q.shape[-1]!=k.shape[-1] or q.shape[1]%k.shape[1]:
        raise ValueError('invalid packed attention head dimensions')
    outputs = []
    for i in range(len(qends)-1):
        qi = q[qends[i]:qends[i+1]].transpose(0, 1)[None]
        ki = k[kends[i]:kends[i+1]].transpose(0, 1)[None]
        vi = v[kends[i]:kends[i+1]].transpose(0, 1)[None]
        if q.device.type=='cpu':qi,ki,vi=qi.float(),ki.float(),vi.float()
        if not qi.shape[2]:
            continue
        groups = qi.shape[1] // ki.shape[1]
        ki = ki.repeat_interleave(groups, dim=1)
        vi = vi.repeat_interleave(groups, dim=1)
        mask = None
        if causal:
            # FlashAttention aligns causal masks to the bottom right for KV caches.
            nq, nk = qi.shape[2], ki.shape[2]
            mask = torch.arange(nk,device=q.device)[None, :] <= torch.arange(nq,device=q.device)[:, None] + nk-nq
        out = scaled_dot_product_attention(qi, ki, vi, attn_mask=mask,dropout_p=0.,is_causal=False,
            scale=kwargs.get('softmax_scale'))
        outputs.append(out[0].transpose(0, 1).to(q.dtype))
    return torch.cat(outputs) if outputs else q.new_empty(q.shape)
