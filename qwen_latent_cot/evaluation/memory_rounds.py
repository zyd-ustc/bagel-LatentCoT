"""Read-only, same-input comparisons of layer KV and denoiser velocity."""
import torch


def tensor_metrics(reference, candidate):
    if reference.shape != candidate.shape or reference.numel() == 0:
        raise ValueError('comparison requires equal, nonempty tensor shapes')
    a, b = reference.detach().float(), candidate.detach().float()
    delta = b-a
    an, bn, dn = a.norm(), b.norm(), delta.norm()
    return dict(equal=torch.equal(reference, candidate),
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        reference_norm=float(an), candidate_norm=float(bn), delta_norm=float(dn),
        relative_l2=float(dn/an.clamp_min(1e-12)), max_abs=float(delta.abs().max()),
        cosine=float((a.flatten()*b.flatten()).sum()/(an*bn).clamp_min(1e-12)))


def kv_metrics(reference, candidate, special_mask, **labels):
    rows=[]
    for subset, mask in [('all', torch.ones_like(special_mask)), ('content', ~special_mask),
                         ('special', special_mask)]:
        if not bool(mask.any()):
            continue
        for component, a, b in zip(('K','V'), reference, candidate):
            rows.append(dict(labels, subset=subset, component=component,
                             slots=int(mask.sum()), **tensor_metrics(a[mask],b[mask])))
    return rows


class MemoryRoundCapture:
    """Copy final GEN reads and body updates; never mutate observed tensors.

    UND suffix continuation runs once after the final body writer. Only body
    updates are compared across rounds. All complete prompt slots are retained.
    """
    def __init__(self, deepest):
        self.deepest=deepest
        self.reads={}
        self.native={}
        self.masks={}
        self.phases={}
        self.read_kinds={}
        self.suffix_writer_count=0
        self.writer_rows=[]
        self.hidden_rows=[]

    def __call__(self, *, event, layer, phase, depth, current, reference, special_mask,
                 from_round=None, to_round=None, hidden_before=None, hidden_after=None, read_kind=None):
        mask=special_mask.detach().cpu().clone()
        a=(reference.keys.detach().cpu().clone(),reference.values.detach().cpu().clone())
        b=(current.keys.detach().cpu().clone(),current.values.detach().cpu().clone())
        if event=='final_read':
            if layer in self.reads.setdefault(depth,{}):
                raise ValueError('duplicate final Memory read; capture conditional branch once')
            self.reads[depth][layer]=b
            self.native[layer]=a
            self.masks[layer]=mask
            self.phases[layer]=phase
            self.read_kinds[layer]=read_kind or 'memory'
        elif event=='writer_update' and phase=='suffix':
            self.suffix_writer_count+=1
        elif event=='writer_update' and depth==self.deepest and phase=='body':
            self.writer_rows.extend(kv_metrics(a,b,mask,scope='within_deepest_call_writer',
                layer=layer,phase=phase,depth=depth,from_round=from_round,to_round=to_round))
            if hidden_before is not None:
                a=hidden_before.detach().cpu();b=hidden_after.detach().cpu()
                for subset,selected in [('all',torch.ones_like(mask)),('content',~mask),('special',mask)]:
                    if bool(selected.any()):
                        self.hidden_rows.append(dict(layer=layer,phase=phase,subset=subset,
                            from_round=from_round,to_round=to_round,**tensor_metrics(a[selected],b[selected])))
