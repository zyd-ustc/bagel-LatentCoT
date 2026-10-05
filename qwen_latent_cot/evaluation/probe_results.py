"""Observed-vs-desired labels are separate. Probe accuracy is not T2I quality."""
from collections import defaultdict
import itertools,math
import numpy as np


def row_key(row):
    return row['prompt_id'],row['seed'],row['step'],row['question_id']


def label_answer(raw,candidates,minimum_confidence=.8):
    from ..bagel.memory_probe import canonical_answer
    if not isinstance(raw,dict) or isinstance(raw.get('confidence'),bool) or not isinstance(raw.get('confidence'),(int,float)):
        raise ValueError('observed label requires answer and numeric confidence')
    confidence=float(raw['confidence'])
    if not math.isfinite(confidence) or not 0<=confidence<=1:raise ValueError('confidence outside [0,1]')
    answer=canonical_answer(raw['answer'])
    if answer not in candidates:raise ValueError('observed answer outside registered candidates')
    return {'observed_answer':answer if confidence>=minimum_confidence else 'unknown',
            'raw_observed_answer':answer,'judge_self_reported_confidence':confidence,
            'confidence_is_calibrated':False}


def summarize_probe(rows,labels,resamples=10000):
    groups=defaultdict(dict);labelmap={row_key(r):r for r in labels}
    if len(labelmap)!=len(labels):raise ValueError('duplicate observed labels')
    for r in rows:
        key=row_key(r)
        if r['source'] in groups[key]:raise ValueError('duplicate probe source')
        groups[key][r['source']]=r
    if set(groups)!=set(labelmap):raise ValueError('QA/label coverage mismatch')
    metrics=defaultdict(lambda:defaultdict(list));clusters=defaultdict(list);known=0;mismatch_count=0
    coverage=defaultdict(lambda:{'known':0,'unknown':0})
    sources={'DYNAMIC','SEED','EMPTY','VIT_IMAGE'}
    for key,g in groups.items():
        if set(g)!=sources:raise ValueError('probe requires Dynamic/Seed/Empty/VIT sources')
        label=labelmap[key];actual=label['observed_answer']
        if any(r['image_sha256']!=label['image_sha256'] or r['candidates']!=label['candidates'] for r in g.values()):raise ValueError('label/QA image or candidate mismatch')
        row=g['DYNAMIC']
        for scope in ('all',f"step_{row['step']}",f"skill_{row['skill']}"):
            coverage[scope]['unknown' if actual=='unknown' else 'known']+=1
        if actual=='unknown':continue
        known+=1;desired=g['DYNAMIC']['desired_answer'];mismatch=desired!=actual;mismatch_count+=mismatch
        for name,row in g.items():
            for scope in ('all',f"step_{row['step']}",f"skill_{row['skill']}"):
                metrics[scope][name].append((row['prediction']==actual,mismatch,row['prediction']==desired,row['choice_probability'][actual]))
        clusters[key[0]].append((float(g['DYNAMIC']['prediction']==actual)-float(g['SEED']['prediction']==actual),
            g['DYNAMIC']['choice_probability'][actual]-g['SEED']['choice_probability'][actual],mismatch))
    result={'known_labels':known,'unknown_labels':len(groups)-known,'prompt_actual_mismatches':mismatch_count,
            'label_source':'offline_image_judge_over_guided_x0_proxy','quality_claim':False,'by_scope':{},'label_coverage':dict(coverage)}
    for scope,values in metrics.items():
        result['by_scope'][scope]={}
        for name,items in values.items():
            subset=[r for r in items if r[1]]
            result['by_scope'][scope][name]={'questions':len(items),'accuracy_vs_observed':float(np.mean([r[0] for r in items])),
                'observed_answer_choice_probability':float(np.mean([r[3] for r in items])),
                'mismatch_questions':len(subset),'accuracy_on_prompt_actual_mismatch':float(np.mean([r[0] for r in subset])) if subset else None,
                'prompt_echo_rate_on_mismatch':float(np.mean([r[2] for r in subset])) if subset else None}
    if clusters:
        arrays=np.array([[sum(v[0] for v in items),sum(v[1] for v in items),len(items)] for _,items in sorted(clusters.items())])
        def value(total):return total[:2]/total[2]
        rng=np.random.default_rng(20261005)
        samples=np.array([value(arrays[rng.integers(0,len(arrays),len(arrays))].sum(0)) for _ in range(resamples)])
        point=value(arrays.sum(0));lo,hi=np.quantile(samples,[.025,.975],axis=0)
        result['dynamic_vs_seed']={n:{'mean':float(p),'ci95':[float(l),float(u)]} for n,p,l,u in zip(('accuracy_delta','observed_choice_probability_delta'),point,lo,hi)}
        result['prompt_clusters']=len(clusters)
    varying=defaultdict(dict)
    for key,g in groups.items():
        actual=labelmap[key]['observed_answer']
        if actual!='unknown':varying[key[0],key[2],key[3]][key[1]]=(actual,g)
    pairs=defaultdict(list)
    for values in varying.values():
        for (a,ga),(b,gb) in itertools.combinations(values.values(),2):
            if a==b:continue
            for name in sources:
                pa,pb=ga[name]['choice_probability'],gb[name]['choice_probability']
                contrast=(pb[b]-pa[b]+pa[a]-pb[a])/2
                pairs[name].append((ga[name]['prediction']==a and gb[name]['prediction']==b,contrast))
    result['different_observed_states_same_prompt']={n:{'seed_pairs':len(v),'both_states_correct_rate':float(np.mean([x[0] for x in v])),
        'mean_correct_direction_probability_contrast':float(np.mean([x[1] for x in v]))} for n,v in pairs.items()}
    vit=result['by_scope'].get('all',{}).get('VIT_IMAGE',{})
    result['reader_calibration_status']='not_established' if not vit else 'report_VIT_image_accuracy_before_interpreting_Memory_failure'
    return result
