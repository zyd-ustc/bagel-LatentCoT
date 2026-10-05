"""Optional diagnostics; never executed by ordinary generation or budget timing."""
import math
import torch


def memory_slot_stats(hidden):
    x=hidden.detach().float()
    if not len(x):return {'slots':0,'effective_rank':None,'mean_pairwise_cosine':None,'slot_std':None,'sigma1_ratio':None}
    sigma=torch.linalg.svdvals(x)
    probabilities=sigma/sigma.sum().clamp_min(1e-12)
    entropy=-(probabilities*probabilities.clamp_min(1e-12).log()).sum()
    unit=torch.nn.functional.normalize(x,dim=-1)
    cosine=unit@unit.T
    n=len(x)
    return {'slots':n,'effective_rank':float(entropy.exp()) if sigma.sum()>0 else 0.,
        'mean_pairwise_cosine':float((cosine.sum()-cosine.diagonal().sum())/(n*(n-1))) if n>1 else None,
        'slot_std':float(x.std(dim=0,unbiased=False).mean()),
        'sigma1_ratio':float(sigma[0]/sigma.sum().clamp_min(1e-12))}


def sampled_read_mass(q,k,glen,klen,plen,mlen,read,maximum_queries=32):
    """Exact dense attention over a deterministic sample of native queries."""
    results=[];qi=ki=0
    for g,kl,p,m in zip(glen,klen,plen,mlen):
        if not read or not m:results.append(0.)
        else:
            ids=torch.linspace(0,g-1,min(g,maximum_queries),device=q.device).long()
            qs=q[qi:qi+g][ids].float().transpose(0,1)
            ks=k[ki:ki+kl].float().repeat_interleave(q.shape[1]//k.shape[1],dim=1).transpose(0,1)
            probabilities=(qs@ks.transpose(-1,-2)/math.sqrt(q.shape[-1])).softmax(-1)
            results.append(float(probabilities[:,:,p:p+m].sum(-1).mean()))
        qi+=g;ki+=kl
    return results
