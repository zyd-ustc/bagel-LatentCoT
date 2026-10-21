"""GenEval adapter for native BAGEL versus Stage A, using the deployed scorer."""
import argparse
from collections import Counter,defaultdict
from functools import partial
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

ARMS=('native_full','loop_full')
TAGS=('single_object','two_object','counting','colors','position','color_attr')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare(args):
    from qwen_latent_cot.diagnostics.inference import resolve_checkpoint
    source=Path(args.tools)/'geneval/src/geneval/prompts/evaluation_metadata.jsonl'
    metadata=read_jsonl(source)
    counts=Counter(row['tag'] for row in metadata)
    if len(metadata)!=553 or set(counts)!=set(TAGS):raise ValueError('Expected original 553-prompt, six-category GenEval')
    selected=[];seen=set()
    for i,row in enumerate(metadata):
        if args.pilot and row['tag'] in seen:continue
        seen.add(row['tag']);selected.append((i,row))
    run=Path(args.run)
    if run.exists():raise FileExistsError('Use a new benchmark run directory')
    checkpoint=resolve_checkpoint(args.checkpoint)
    seeds=[0] if args.pilot else args.seeds
    if len(set(seeds))!=len(seeds) or not seeds:raise ValueError('Seeds must be nonempty and unique')
    cases=[dict(id=f'geneval_{i:05d}',group='geneval',category=row['tag'],source='GenEval original',
                benchmark='geneval',official_item_index=i,benchmark_metadata=row,
                full_prompt=row['prompt'],partial_prompt=row['prompt'],removed_clauses='',
                checks=['官方标签：'+json.dumps(row,ensure_ascii=False)]) for i,row in selected]
    run.mkdir(parents=True)
    (run/'prompts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases))
    plan=dict(benchmark='geneval',source=str(source),source_prompts=553,source_categories=dict(counts),
              selected_prompts=len(cases),selected_categories=dict(Counter(r['category'] for r in cases)),
              subset=args.pilot,seeds=seeds,checkpoint=str(checkpoint),arms=list(ARMS),
              expected_images=len(cases)*len(seeds)*len(ARMS),
              protocol='official prompt/labels; native BAGEL sampler and Stage A checkpoint settings; CPU-adapted official scorer',
              formal_default_protocol=not args.pilot and seeds==[0,1,2,3])
    (run/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(plan,ensure_ascii=False,indent=2))


def export(run):
    run=Path(run);plan=json.loads((run/'plan.json').read_text());cases=read_jsonl(run/'prompts.jsonl')
    exported=0
    for seed in plan['seeds']:
        directory=run/'samples'/f'seed_{seed:04d}'
        for index,case in enumerate(cases):
            record=json.loads((directory/f'case_{index:03d}.json').read_text())
            if record['id']!=case['id'] or record['seed']!=seed or record['checkpoint']!=plan['checkpoint']:
                raise ValueError('Generation identity differs from the pinned plan')
            for arm in ARMS:
                result=record['arms'][arm]
                if result['prompt']!=case['full_prompt'] or result['native']!=(arm=='native_full'):
                    raise ValueError('Benchmark arm does not match its native/full-prompt contract')
                path=run/'official'/arm/f'{case["official_item_index"]:05d}'
                (path/'samples').mkdir(parents=True,exist_ok=True)
                (path/'metadata.jsonl').write_text(json.dumps(case['benchmark_metadata'])+'\n')
                shutil.copyfile(directory/result['image'],path/'samples'/f'{seed:04d}.png')
                exported+=1
    if exported!=plan['expected_images']:raise ValueError('Incomplete benchmark export')
    print(json.dumps(dict(exported_images=exported,expected_images=plan['expected_images'])))


def load_geneval(tools):
    root=Path(tools)/'geneval'
    path=root/'src/geneval/evaluation/evaluate_images.py'
    spec=importlib.util.spec_from_file_location('official_geneval',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.args=SimpleNamespace(options={},model_path=str(root/'models'),
        model_config=str(Path(module.mmdet.__file__).parent/'../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'))
    module.object_detector,(module.clip_model,module.transform,module.tokenizer),module.classnames=module.load_models(module.args)
    module.THRESHOLD,module.COUNTING_THRESHOLD=.3,.9
    module.MAX_OBJECTS,module.NMS_THRESHOLD,module.POSITION_THRESHOLD=16,1.,.1
    if module.DEVICE=='cpu':
        module.zsc.zero_shot_classifier=partial(module.zsc.zero_shot_classifier,amp=False)
        module.zsc.run_classification=partial(module.zsc.run_classification,amp=False)
    return module


def summarize(run):
    run=Path(run);plan=json.loads((run/'plan.json').read_text());cases=read_jsonl(run/'prompts.jsonl')
    expected={(r['official_item_index'],seed) for r in cases for seed in plan['seeds']}
    result=dict(benchmark='geneval',subset=plan['subset'],source_prompts=553,seeds=plan['seeds'],
                formal_default_protocol=plan['formal_default_protocol'],arms={})
    for arm in ARMS:
        path=run/'scores'/f'{arm}.jsonl'
        rows=read_jsonl(path) if path.exists() else []
        keys=[(r['official_item_index'],r['seed']) for r in rows]
        if len(set(keys))!=len(keys) or not set(keys)<=expected:raise ValueError('Duplicate/unexpected scored images')
        by=defaultdict(list)
        for row in rows:by[row['tag']].append(float(row['correct']))
        categories={tag:sum(values)/len(values) for tag,values in by.items()}
        complete=set(keys)==expected
        item=dict(scored_images=len(rows),expected_images=len(expected),coverage_complete=complete,
                  category_accuracy=categories,scored_subset_macro_accuracy=sum(categories.values())/len(categories) if categories else None)
        if complete and not plan['subset']:item['full_553_prompt_macro_accuracy']=item['scored_subset_macro_accuracy']
        if complete and plan['formal_default_protocol']:item['geneval_macro_accuracy']=item['scored_subset_macro_accuracy']
        result['arms'][arm]=item
    (run/'benchmark_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def score(args):
    run=Path(args.run);plan=json.loads((run/'plan.json').read_text());cases=read_jsonl(run/'prompts.jsonl')
    jobs=[(arm,case,seed,run/'official'/arm/f'{case["official_item_index"]:05d}'/'samples'/f'{seed:04d}.png')
          for arm in ARMS for case in cases for seed in plan['seeds']]
    if any(not path.is_file() for _,_,_,path in jobs):raise ValueError('Exported images incomplete; run export first')
    scores=run/'scores';scores.mkdir(exist_ok=True)
    completed={}
    for arm in ARMS:
        file=scores/f'{arm}.jsonl'
        completed[arm]={(r['official_item_index'],r['seed']) for r in read_jsonl(file)} if file.exists() else set()
    if all((case['official_item_index'],seed) in completed[arm] for arm,case,seed,_ in jobs):
        summarize(run);return
    judge=load_geneval(args.tools)
    for arm,case,seed,path in jobs:
        key=(case['official_item_index'],seed)
        if key in completed[arm]:continue
        result=judge.evaluate_image(str(path),case['benchmark_metadata'])
        record=dict(result,arm=arm,official_item_index=case['official_item_index'],seed=seed,
                    checkpoint=plan['checkpoint'] if arm=='loop_full' else None,
                    image=str(path),scorer_device=judge.DEVICE,clip_cpu_amp=False)
        with (scores/f'{arm}.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(dict(arm=arm,index=key[0],seed=seed,correct=result['correct'],reason=result.get('reason'))),flush=True)
    summarize(run)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','export','score','summary'))
    parser.add_argument('--run',required=True)
    parser.add_argument('--tools',default='/root/npu-eval-tools')
    parser.add_argument('--checkpoint')
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3])
    args=parser.parse_args()
    if args.command=='prepare':
        if not args.checkpoint:parser.error('prepare requires --checkpoint')
        prepare(args)
    elif args.command=='export':export(args.run)
    elif args.command=='score':score(args)
    else:summarize(args.run)


if __name__=='__main__':main()
