"""Read-only diagnostics for complete prompt Memory slots."""
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
