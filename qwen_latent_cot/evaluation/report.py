"""Pair first, then bootstrap complete prompt clusters (all seeds and atoms)."""
from collections import defaultdict
import math
import numpy as np
from .io import identity


def gm(atoms):
    return 0. if any(v==0 for v in atoms) else math.exp(sum(math.log(v) for v in atoms)/len(atoms))


def paired_report(reference, candidate, resamples=10000, seed=20261005):
    if set(reference)!=set(candidate): raise ValueError('paired coverage mismatch')
    groups=defaultdict(lambda:np.zeros(10))
    repair=damage=basefail=basepass=0
    for key,a in reference.items():
        b=candidate[key]
        for field in ('prompt','noise_sha256','height','width'):
            if a[field]!=b[field]:raise ValueError(f'paired {field} differs')
        aa,bb=np.array(a['semantic_atoms'])>=.5,np.array(b['semantic_atoms'])>=.5
        if len(aa)!=len(bb):raise ValueError('paired constraints differ')
        r=int((~aa & bb).sum());d=int((aa & ~bb).sum())
        failed=int((~aa).sum());passed=int(aa.sum())
        groups[key[0]]+=np.array([r,d,len(aa),gm(b['semantic_atoms'])-gm(a['semantic_atoms']),1,
            b['quality_proxy']-a['quality_proxy'],float(b['invalid'])-float(a['invalid']),
            float(bb.all())-float(aa.all()),failed,passed])
        repair+=r;damage+=d;basefail+=failed;basepass+=passed
    values=np.array([groups[k] for k in sorted(groups)])
    def metrics(totals):
        return np.array([(totals[0]-totals[1])/totals[2],totals[3]/totals[4],
                         totals[5]/totals[4],totals[6]/totals[4],totals[7]/totals[4]])
    rng=np.random.default_rng(seed)
    samples=np.array([metrics(values[rng.integers(0,len(values),len(values))].sum(0)) for _ in range(resamples)])
    point=metrics(values.sum(0));lower,upper=np.quantile(samples,[.025,.975],axis=0)
    names=('net_repair','semantic_gm_delta','quality_proxy_delta','invalid_delta','all_pass_delta')
    out={n:{'mean':float(p),'ci95':[float(l),float(u)]} for n,p,l,u in zip(names,point,lower,upper)}
    out.update(repair_count=repair,damage_count=damage,paired_atoms=int(values[:,2].sum()),
        repair_given_base_fail=repair/basefail if basefail else None,
        damage_given_base_pass=damage/basepass if basepass else None,
        prompt_clusters=len(groups),paired_images=len(reference),bootstrap_replicates=resamples)
    return out


def summarize(rows,resamples=10000):
    groups=defaultdict(dict)
    for r in rows:
        key=identity(r)
        if key in groups[r['arm']]:raise ValueError('duplicate scored record')
        if r['semantic_atoms'] is None or r['quality_proxy'] is None or r['invalid'] is None:
            raise ValueError('incomplete scoring; no silent pair removal')
        groups[r['arm']][key]=r
    if 'BASE' not in groups:raise ValueError('BASE required')
    output={}
    for arm,values in groups.items():
        records=list(values.values())
        output[arm]={'semantic_gm':float(np.mean([gm(r['semantic_atoms']) for r in records])),
            'quality_proxy':float(np.mean([r['quality_proxy'] for r in records])),
            'invalid_rate':float(np.mean([r['invalid'] for r in records])),
            'latency_median_seconds':float(np.median([r['generation_seconds'] for r in records])),
            'latency_p95_seconds':float(np.quantile([r['generation_seconds'] for r in records],.95)),
            'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in records),
            'timing_is_engineering_only':True,
            'vs_BASE':paired_report(groups['BASE'],values,resamples)}
        buckets=set(r['bucket'] for r in records)
        output[arm]['buckets']={}
        for bucket in buckets:
            own={k:r for k,r in values.items() if r['bucket']==bucket}
            base={k:r for k,r in groups['BASE'].items() if r['bucket']==bucket}
            output[arm]['buckets'][bucket]=paired_report(base,own,resamples)
    return output


def repair_retention(base, full, candidate):
    """Keep the reference arm's repairs and avoid its damage, using Base as the atom anchor."""
    if set(base)!=set(full) or set(base)!=set(candidate):
        raise ValueError('retention requires complete three-arm pairs')
    repairs=retained=damages=avoided=0
    for key,a in base.items():
        aa=np.array(a['semantic_atoms'])>=.5
        ff=np.array(full[key]['semantic_atoms'])>=.5
        cc=np.array(candidate[key]['semantic_atoms'])>=.5
        if aa.shape!=ff.shape or aa.shape!=cc.shape:
            raise ValueError('retention constraints differ')
        repaired=~aa & ff;damaged=aa & ~ff
        repairs+=int(repaired.sum());retained+=int((repaired & cc).sum())
        damages+=int(damaged.sum());avoided+=int((damaged & cc).sum())
    return {'reference_repair_atoms':repairs,'retained_reference_repair_atoms':retained,
        'retained_reference_repair_fraction':retained/repairs if repairs else None,
        'reference_damage_atoms':damages,'avoided_reference_damage_atoms':avoided,
        'avoided_reference_damage_fraction':avoided/damages if damages else None,
        'threshold':.5,'status':'VLM_proxy_pending_manual_review'}
