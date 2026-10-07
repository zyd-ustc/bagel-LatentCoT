#!/usr/bin/env python
"""Merge completed diagnostic shards; summarize measured changes without gain claims."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def stats(values):
    values=sorted(values)
    if not values or not all(math.isfinite(v) for v in values):raise ValueError('empty/nonfinite summary values')
    return dict(n=len(values),mean=statistics.mean(values),median=statistics.median(values),
                p90=values[min(len(values)-1,math.ceil(.9*len(values))-1)],max=max(values))


def summarize(records):
    memory=defaultdict(list);velocity=defaultdict(list);contraction=defaultdict(list);hidden=defaultdict(list)
    for record in records:
        mode=record['memory_mode']
        for row in record.get('hidden',[]):
            hidden[(mode,row['layer'],row['subset'],row['from_round'],row['to_round'],record['step'])].append(row)
        for row in record['memory']:
            key=(mode,)+tuple(row[k] for k in ('scope','phase','layer','subset','component','from_round','to_round'))+(record['step'],)
            memory[key].append(row)
        for row in record['velocity']:
            velocity[(mode,row['branch'],row['from_round'],row['to_round'],record['step'])].append(row)
        for component in ('K','V'):
            updates=defaultdict(dict)
            for row in record['memory']:
                if row['scope']=='within_deepest_call_writer' and row['subset']=='content' and row['component']==component:
                    updates[row['layer']][row['to_round']]=row['delta_norm']
            # Combine layer delta norms by sum of squares, including the first layer.
            norms={r:math.sqrt(sum(u[r]**2 for u in updates.values())) for r in (1,2,3)}
            for before,after in ((1,2),(2,3)):
                if norms[before]>0:
                    contraction[(mode,component,before,after,record['step'])].append(norms[after]/norms[before])
    memory_summary=[]
    for key,rows in sorted(memory.items()):
        labels=dict(zip(('memory_mode','scope','phase','layer','subset','component','from_round','to_round','step'),key))
        memory_summary.append(dict(labels,relative_l2=stats([r['relative_l2'] for r in rows]),
            delta_norm=stats([r['delta_norm'] for r in rows]),cosine=stats([r['cosine'] for r in rows]),
            equal_fraction=sum(r['equal'] for r in rows)/len(rows)))
    velocity_summary=[]
    for key,rows in sorted(velocity.items()):
        labels=dict(zip(('memory_mode','branch','from_round','to_round','step'),key))
        velocity_summary.append(dict(labels,relative_l2=stats([r['relative_l2'] for r in rows]),
            delta_over_first_effect=stats([r['delta_over_first_effect'] for r in rows]),
            equal_fraction=sum(r['equal'] for r in rows)/len(rows)))
    return dict(memory=memory_summary,velocity=velocity_summary,
        hidden=[dict(memory_mode=k[0],layer=k[1],subset=k[2],from_round=k[3],to_round=k[4],step=k[5],
            relative_l2=stats([r['relative_l2'] for r in v]),candidate_norm=stats([r['candidate_norm'] for r in v]),
            delta_norm=stats([r['delta_norm'] for r in v]),
            equal_fraction=sum(r['equal'] for r in v)/len(v)) for k,v in sorted(hidden.items())],
        writer_update_contraction=[dict(memory_mode=k[0],component=k[1],previous_update=k[2],next_update=k[3],step=k[4],
            ratio=stats(v)) for k,v in sorted(contraction.items())])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    args=p.parse_args();root=Path(args.run_dir)
    plan=json.loads((root/'plan.json').read_text());records=[];seen=set();inputs={}
    modes=plan['memory_modes']
    for shard in range(plan['num_shards']):
        folder=root/f'worker_{shard}'
        meta=json.loads((folder/'run.json').read_text())
        if meta!=dict(plan,shard_index=shard):raise ValueError('shard provenance mismatch')
        done=json.loads((folder/'complete.json').read_text())
        if not done['prompt_cache_unchanged']:raise ValueError('prompt cache changed')
        local=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
        if len(local)!=done['probes']:raise ValueError('incomplete shard')
        for record in local:
            mode=record['memory_mode']
            key=(mode,record['prompt_id'],record['seed'],record['step'])
            input_key=key[1:];input_value=(record['x_t_sha256'],record['timestep'],record['input_noise_sha256'])
            if input_key in inputs and inputs[input_key]!=input_value:raise ValueError('cross-mode inputs differ')
            inputs[input_key]=input_value
            if key in seen:raise ValueError('duplicate prompt/seed/probe')
            if record['index']%plan['num_shards']!=shard:raise ValueError('wrong prompt shard')
            if not all(record['sanity'].values()):raise ValueError('diagnostic contract failed')
            if not all(r['finite'] for r in record['memory']+record['velocity']+record.get('hidden',[])):raise ValueError('nonfinite metrics')
            seen.add(key);records.append(record)
    expected={(m,p,plan['seed'],s) for m in modes for p in plan['prompt_ids'] for s in plan['probe_steps']}
    if seen!=expected:raise ValueError('prompt/probe coverage is incomplete')
    records.sort(key=lambda r:(r['index'],r['step'],r.get('memory_mode','')))
    summary=dict(plan,probes=len(records),contracts_passed=True,semantic_gain_verified=False,
        quality_retention_verified=False,**summarize(records))
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    lines=[f"# Memory round diagnosis: {len(plan['prompt_ids'])} prompts / {len(records)} probes",'',
        'All depths use the same x_t on the native Base trajectory. No dynamic trajectory, VAE decode or quality scorer is run.',
        'R0 parity, special-slot pinning, persistent-state contracts and input/cache immutability checks passed.',
        'Numbers describe tensor changes. They do not establish semantic gain or quality retention.','',
        '|Mode|Step|Actual t|Branch|Depth pair|Velocity relative L2 mean / median / p90|Delta / first effect mean|',
        '|---|---|---:|---|---|---:|---:|']
    for row in summary['velocity']:
        if (row['from_round'],row['to_round']) not in [(0,1),(1,2),(2,3)]:continue
        ts={round(r['timestep'],8) for r in records if r['step']==row['step']}
        if len(ts)!=1:raise ValueError('timestep differs across matched probes')
        st=row['relative_l2']
        lines.append(f"|{row['memory_mode']}|{row['step']}|{next(iter(ts)):.6f}|{row['branch']}|R{row['from_round']}→R{row['to_round']}|"
                     f"{st['mean']:.6f} / {st['median']:.6f} / {st['p90']:.6f}|{row['delta_over_first_effect']['mean']:.6f}|")
    lines+=['','Content-only Memory KV actually read by GEN at each independent depth:',
        'Every body layer reads dynamic Memory. Values pool per-layer relative changes; they are not an attention sensitivity measure.',
        '', '|Mode|Step|Phase|Component|Depth pair|Relative L2 mean / median / p90|Exact-equal fraction|',
        '|---|---|---|---|---|---:|---:|']
    pooled=defaultdict(list)
    for record in records:
        for row in record['memory']:
            if (row['scope']=='independent_depth_final_read' and row['subset']=='content'
                    and row['layer']>=plan['start_layer']):
                pooled[(record['memory_mode'],record['step'],row['phase'],row['component'],row['from_round'],row['to_round'])].append(row)
    for key,rows in sorted(pooled.items()):
        mode,step,phase,component,previous,depth=key
        s=stats([r['relative_l2'] for r in rows]);equal=sum(r['equal'] for r in rows)/len(rows)
        lines.append(f'|{mode}|{step}|{phase}|{component}|R{previous}→R{depth}|'
                     f"{s['mean']:.6f} / {s['median']:.6f} / {s['p90']:.6f}|{equal:.6f}|")
    lines+=['','Within one R3 call, content-only Memory update norm ratios across body layers:',
            '', '|Mode|Step|Component|Update ratio|Mean|Median|p90|','|---|---|---|---|---:|---:|---:|']
    for r in summary['writer_update_contraction']:
        s=r['ratio'];lines.append(f"|{r['memory_mode']}|{r['step']}|{r['component']}|ΔM{r['next_update']} / ΔM{r['previous_update']}|"
                                  f"{s['mean']:.6f}|{s['median']:.6f}|{s['p90']:.6f}|")
    lines+=['','Persistent UND hidden updates (content tokens; mean relative L2):',
        '', '|Mode|Step|Layer|Round pair|Hidden relative L2|Hidden update norm|Hidden norm|',
        '|---|---|---|---|---:|---:|---:|']
    for r in summary['hidden']:
        if r['subset']=='content':
            lines.append(f"|{r['memory_mode']}|{r['step']}|{r['layer']}|{r['from_round']}→{r['to_round']}|{r['relative_l2']['mean']:.6f}|{r['delta_norm']['mean']:.6f}|{r['candidate_norm']['mean']:.6f}|")
    lines+=['','Here ΔMr = Mr − M(r−1), with M0 = native prompt KV. Ratios below 1 mean a smaller measured update.',
        'Per-layer K/V relative L2, cosine, exact equality and independent-depth final-read comparisons are in summary.json.',
        'The writer suffix runs once after the final body update; it is excluded from within-call recurrence ratios.',
        'Fixed special slots are reported separately so they do not dilute content-only changes.','']
    (root/'summary.md').write_text('\n'.join(lines))
    print(root/'summary.md',flush=True)


if __name__=='__main__':main()
