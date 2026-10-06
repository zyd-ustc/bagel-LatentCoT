#!/usr/bin/env python
import argparse
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from qwen_latent_cot.evaluation.io import load_manifests,read_jsonl,sha256,identity
from qwen_latent_cot.evaluation.scoring import LocalScorer
from qwen_latent_cot.evaluation.report import summarize


def main():
    p=argparse.ArgumentParser(description='Paired GenEval2/TIIF semantics, quality proxy, Repair/Damage and prompt-cluster CI')
    p.add_argument('--manifests',nargs='+',required=True);p.add_argument('--benchmark',required=True)
    p.add_argument('--judge-model',required=True);p.add_argument('--geneval2-source',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--output-dir',required=True)
    p.add_argument('--bootstrap-replicates',type=int,default=10000)
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    args=p.parse_args()
    if not 0<=args.shard_index<args.num_shards:raise ValueError('invalid scoring shard')
    records,run=load_manifests(args.manifests)
    if run['benchmark_sha256']!=sha256(args.benchmark):raise ValueError('benchmark differs from generation')
    data=read_jsonl(args.benchmark)
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    work={key:i for i,key in enumerate(sorted({identity(r) for r in records}))}
    records=[r for r in records if work[identity(r)]%args.num_shards==args.shard_index]
    scorer=LocalScorer(args.judge_model,args.geneval2_source,args.device)
    provenance={'run':run,'scorer':scorer.provenance,'scoring_shard':[args.shard_index,args.num_shards]}
    binding=output/'scorer.json'
    if binding.exists() and json.loads(binding.read_text())!=provenance:raise ValueError('scorer binding changed')
    binding.write_text(json.dumps(provenance,indent=2)+'\n')
    scorefile=output/'scores.jsonl';done={}
    if scorefile.exists():
        for r in read_jsonl(scorefile):
            key=(r['arm'],*identity(r))
            if key in done:raise ValueError('duplicate scoring cache')
            done[key]=r
    scored=[]
    for row in records:
        key=(row['arm'],*identity(row));benchmark=data[row['index']]
        if benchmark['prompt']!=row['prompt']:raise ValueError('benchmark prompt mismatch')
        if key in done:
            result=done[key]
            if result['image_sha256']!=row['image_sha256']:raise ValueError('scored image changed')
        else:
            result=scorer.score(row,benchmark)
            with scorefile.open('a') as f:f.write(json.dumps(result)+'\n');f.flush()
        scored.append(result);print(f"scored {row['arm']} {row['prompt_id']} seed={row['seed']}",flush=True)
    if args.num_shards>1:
        print(f'Completed scoring shard: {scorefile}',flush=True)
        return
    write_summary(scored,run,output,args.bootstrap_replicates)


def write_summary(scored,run,output,resamples=10000):
    summary={'status':'engineering_scored' if run['stage']=='engineering' else 'paired_scored',
        'training_admitted':False,'training_admission_status':'not_assessed_here; assess_image_gain_and_memory_content_evidence',
        'stage':run['stage'],'source_sha256':run['source_sha256'],'model_sha256':run['model_sha256'],
        'quality_is_proxy':True,'blind_review':'pending','statistics_unit':'prompt_cluster_all_seeds_and_atoms',
        'arms':summarize(scored,resamples)}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Frozen BAGEL denoiser loop evaluation','','Quality is a VLM proxy; blind review is pending. Engineering timing does not establish budget compliance.','',
        '|Arm|Semantic GM|Quality proxy|Invalid|Net Repair vs Base (95% CI)|Repair / Damage|',
        '|---|---:|---:|---:|---|---:|']
    for arm,s in summary['arms'].items():
        delta=s['vs_BASE']['net_repair'];lo,hi=delta['ci95']
        lines.append(f"|{arm}|{s['semantic_gm']:.4f}|{s['quality_proxy']:.4f}|{s['invalid_rate']:.4f}|{delta['mean']:.4f} [{lo:.4f}, {hi:.4f}]|{s['vs_BASE']['repair_count']} / {s['vs_BASE']['damage_count']}|")
    controls=[(arm,key[3:],value) for arm,s in summary['arms'].items()
              for key,value in s.items() if key.startswith('vs_') and key!='vs_BASE']
    if controls:
        lines+=['','|Candidate / Reference|Semantic GM delta (95% CI)|Quality proxy delta (95% CI)|Repair / Damage|',
                '|---|---|---|---:|']
        def interval(value):
            lo,hi=value['ci95']
            return f"{value['mean']:.4f} [{lo:.4f}, {hi:.4f}]"
        for arm,reference,value in controls:
            lines.append(f"|{arm} / {reference}|{interval(value['semantic_gm_delta'])}|{interval(value['quality_proxy_delta'])}|{value['repair_count']} / {value['damage_count']}|")
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    from qwen_latent_cot.evaluation.blind_review import make_blind_pack
    make_blind_pack(scored,output/'blind_review')
    print(output/'summary.md',flush=True)


if __name__=='__main__':main()
