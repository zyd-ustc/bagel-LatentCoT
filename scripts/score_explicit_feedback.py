"""Score existing feedback images with upstream GenEval2 and an independent VLM.

Workers score whole prompt groups. Aggregation bootstraps paired prompt deltas;
it never selects the best early window separately for each image.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def mean(values):
    return sum(values)/len(values)


def gm(values):
    return math.exp(mean([math.log(v) for v in values])) if all(v>0 for v in values) else 0.


def collect(run, benchmark):
    by_prompt={r['prompt']:r for r in map(json.loads,Path(benchmark).read_text().splitlines()) if r}
    groups=[]
    for path in sorted(Path(run).glob('device_*/*.json')):
        row=json.loads(path.read_text())
        if not row.get('images') or not row.get('feedback'):continue
        if row['prompt'] not in by_prompt:raise ValueError('run prompt missing from benchmark')
        b=by_prompt[row['prompt']]
        if len(b['skills'])!=len(b['vqa_list']):raise ValueError('VQA skill layout mismatch')
        groups.append(dict(record=str(path),sample_id=row['sample_id'],prompt=row['prompt'],
            origin=row['feedback']['origin'],feedback=row['feedback'],initial_noise_hash=row['initial_noise_hash'],
            benchmark=b,images={a:str(path.parent/f) for a,f in row['images'].items()}))
    if not groups:raise ValueError('no completed feedback image groups')
    return groups


def collect_shallow(run,benchmark):
    by_prompt={r['prompt']:r for r in map(json.loads,Path(benchmark).read_text().splitlines()) if r}
    plan=json.loads((Path(run)/'selection.json').read_text());groups=[]
    for i,row in enumerate(plan['rows']):
        files={'Base':str(Path(run)/f'p{i:02d}_base.png')}
        for early in plan['early']:
            for passes in plan['passes']:
                arm=f'early{early}_R{passes}';files[arm]=str(Path(run)/f'p{i:02d}_{arm}.png')
        if any(not Path(p).is_file() for p in files.values()):raise ValueError('missing shallow image')
        groups.append(dict(sample_id=row['prompt_id'],prompt=row['prompt'],origin='historical_C2_observe',
                           benchmark=by_prompt[row['prompt']],images=files))
    return groups


def paired_interval(values,seed=0):
    rng=random.Random(seed)
    samples=sorted(mean(rng.choices(values,k=len(values))) for _ in range(10000))
    return dict(mean=mean(values),low=samples[249],high=samples[9749])


def summarize(rows):
    origins=sorted({r['origin'] for r in rows}); result={}
    for origin in origins:
        selected=[r for r in rows if r['origin']==origin]
        by_arm={a:[r for r in selected if r['arm']==a] for a in sorted({r['arm'] for r in selected})}
        summaries={}
        for arm,items in by_arm.items():
            skills={}
            for r in items:
                for skill,value in zip(r['skills'],r['atoms']):skills.setdefault(skill,[]).append(value)
            summaries[arm]=dict(prompts=len(items),soft_tifa_am=100*mean([mean(r['atoms']) for r in items]),
                soft_tifa_gm=100*mean([gm(r['atoms']) for r in items]),
                tifa=100*mean([mean(r['hard_atoms']) for r in items]),
                skills={s:100*mean(v) for s,v in skills.items()})
        pairs={}
        for arm,items in by_arm.items():
            if arm=='Base':continue
            controls=['Base']
            if 'Text-memory' in arm:controls.append(arm.replace('Text-memory','Layout-memory'))
            if arm.endswith('_Text'):controls.append(arm.replace('_Text','_Layout-text'))
            for control in controls:
                if control not in by_arm:continue
                references={r['sample_id']:r for r in by_arm[control]}
                if {r['sample_id'] for r in items}!=set(references):raise ValueError('incomplete paired images')
                differences=[]; fixes=damage=correct=wrong=0
                for r in items:
                    before=references[r['sample_id']]
                    differences.append(100*(gm(r['atoms'])-gm(before['atoms'])))
                    for a,b in zip(before['hard_atoms'],r['hard_atoms']):
                        correct+=int(a);wrong+=int(not a);fixes+=int(not a and b);damage+=int(a and not b)
                pairs[arm+' vs '+control]=dict(gm_delta_pp=paired_interval(differences),
                    improved_prompts=sum(v>1e-6 for v in differences),worse_prompts=sum(v< -1e-6 for v in differences),
                    judge_atom_fixes=fixes,judge_atom_damage=damage,baseline_correct_atoms=correct,baseline_wrong_atoms=wrong,
                    scope='hard VQA transitions are judge predictions, not human verified facts')
        result[origin]=dict(arms=summaries,paired=pairs)
    return result


def scorer(judge, source, device):
    from qwen_latent_cot.interaction.rewards import scorer as atom_scorer
    return atom_scorer(judge, source, device)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['score','aggregate'])
    parser.add_argument('--run',required=True);parser.add_argument('--benchmark',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--shallow-run',help='optional matched historical implicit-memory comparison')
    parser.add_argument('--judge');parser.add_argument('--official-source')
    parser.add_argument('--device',default='npu:0')
    parser.add_argument('--rank',type=int,default=0);parser.add_argument('--world',type=int,default=1)
    args=parser.parse_args(argv);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    groups=collect(args.run,args.benchmark)
    if args.shallow_run:groups+=collect_shallow(args.shallow_run,args.benchmark)
    if args.command=='aggregate':
        rows=[json.loads(line) for p in sorted(output.glob('scores_rank_*.jsonl')) for line in p.read_text().splitlines()]
        expected={(r['sample_id'],r['origin'],a) for r in groups for a in r['images']}
        keys=[(r['sample_id'],r['origin'],r['arm']) for r in rows]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('incomplete or duplicate scored arms')
        report=dict(run=args.run,shallow_run=args.shallow_run,benchmark_sha256=digest(args.benchmark),scored_images=len(rows),
            groups=summarize(rows),metric='upstream GenEval2 functions; NPU SDPA Qwen3-VL-8B-Instruct',
            scope='selected prompts and one seed; no claim of a mathematical performance upper bound')
        if args.shallow_run:
            report['text_teacher_vs_historical_reader']={}
            explicit=[r for r in rows if r['origin']=='bagel_auto']
            implicit=[r for r in rows if r['origin']=='historical_C2_observe']
            bases={r['sample_id']:r for r in implicit if r['arm']=='Base'}
            for r in explicit:
                if r['arm']=='Base' and r['image_sha256']!=bases[r['sample_id']]['image_sha256']:
                    raise ValueError('explicit and shallow baseline images differ')
            for window in (10,20):
                teachers={r['sample_id']:r for r in explicit if r['arm']==f'early{window}_Text-memory'}
                for passes in (2,3,4):
                    students=[r for r in implicit if r['arm']==f'early{window}_R{passes}']
                    deltas=[100*(gm(teachers[r['sample_id']]['atoms'])-gm(r['atoms'])) for r in students]
                    report['text_teacher_vs_historical_reader'][f'early{window}_Text-memory vs R{passes}']=paired_interval(deltas)
        (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False,indent=2));return
    if not args.judge or not args.official_source:parser.error('score requires --judge and --official-source')
    target=output/f'scores_rank_{args.rank:02d}.jsonl'
    if target.exists():raise ValueError('use a fresh worker output')
    score=scorer(args.judge,args.official_source,args.device)
    meta=dict(judge=args.judge,official_source=args.official_source,official_source_sha256=digest(args.official_source),
        judge_config_sha256=digest(Path(args.judge)/'config.json'),code_sha256=digest(__file__),device=args.device)
    (output/f'manifest_rank_{args.rank:02d}.json').write_text(json.dumps(meta,indent=2)+'\n')
    with target.open('w') as stream:
        cache={}
        for group in groups[args.rank::args.world]:
            b=group['benchmark']
            for arm,path in group['images'].items():
                start=time.monotonic();image_hash=digest(path)
                key=(image_hash,json.dumps(b['vqa_list']))
                if key not in cache:cache[key]=score(b['vqa_list'],path)
                atoms,hard=cache[key]
                row=dict(sample_id=group['sample_id'],prompt=group['prompt'],origin=group['origin'],
                    arm=arm,path=path,image_sha256=image_hash,skills=b['skills'],questions=b['vqa_list'],
                    atoms=atoms,hard_atoms=hard,seconds=time.monotonic()-start)
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                print(f"{group['sample_id']} {arm}: GM={100*gm(atoms):.3f}",flush=True)


if __name__=='__main__':main()
