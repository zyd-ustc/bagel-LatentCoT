"""Prepare, generate and score native Memory-loop benchmarks on modelarts-job."""
import argparse
import ast
import csv
from functools import partial
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARM = 'early20_L28_R2'
SAMPLING = dict(resolution=512, sampler_steps=50, timestep_shift=3.0,
                cfg_text_scale=4.0, cfg_renorm_type='global', cfg_renorm_min=0.0)
READY = ['geneval', 'geneval2', 'dpg', 'dsg']
ALL = READY + ['geneval-pp', 'unigen-en', 'unigen-zh', 'unigen-en-long', 'unigen-zh-long']


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def annotations(path):
    groups = {}
    for row in csv.DictReader(Path(path).open()):
        item = groups.setdefault(row['item_id'], dict(prompt=row['text'], questions={}, dependencies={}, tuples={},
                                                     dependency_errors={}))
        qid = int(row['proposition_id'])
        item['questions'][qid] = row['question_natural_language']
        try:
            item['dependencies'][qid] = [int(x.strip()) for x in row['dependency'].split(',')]
        except ValueError:
            item['dependencies'][qid] = []
            item['dependency_errors'][qid] = row['dependency']
        item['tuples'][qid] = row['tuple']
    for item in groups.values():
        item['missing_parents'] = {q: [p for p in parents if p and p not in item['questions']]
                                   for q, parents in item['dependencies'].items()
                                   if any(p and p not in item['questions'] for p in parents)}
    return groups


def catalog(args):
    root = Path(args.tools)
    datasets = {}
    paths = dict(geneval=root/'geneval/src/geneval/prompts/evaluation_metadata.jsonl',
                 geneval2=Path(args.geneval2),
                 **{'geneval-pp': root/'geneval-pp/src/Geneval++.jsonl'})
    for name, path in paths.items():
        datasets[name] = [(f'{i:05d}', row['prompt'], row) for i, row in enumerate(read_jsonl(path))]
    for name, relative in [('dpg', 'dpg/src/dpg_bench/dpg_bench.csv'),
                           ('dsg', 'dsg/src/dsg/data/dsg-1k-anns.csv')]:
        datasets[name] = [(key, row['prompt'], row) for key, row in annotations(root/relative).items()]
    for language in ('en', 'zh', 'en_long', 'zh_long'):
        path = root/f'unigen/src/data/test_prompts_{language}.csv'
        name = 'unigen-' + language.replace('_', '-')
        datasets[name] = [(r['index'], r[f'prompt_{language[:2]}'], r)
                          for r in csv.DictReader(path.open())]
    return datasets


def prepare(args):
    output = Path(args.run).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Resolve a training directory at preparation time; workers use this pinned
    # file even if training saves a newer step during benchmark generation.
    from qwen_latent_cot.diagnostics.inference import resolve_checkpoint
    args.checkpoint = str(resolve_checkpoint(args.checkpoint))
    arm = getattr(args, 'arm', None)
    if arm is None:
        from qwen_latent_cot.interaction.checkpoint import read, KIND
        arm = 'interaction_memory' if read(args.checkpoint).get('kind') == KIND else 'c2_memory'
    identity = dict(arm=arm, checkpoint=str(Path(args.checkpoint).resolve()),
                    checkpoint_sha256=getattr(args, 'checkpoint_sha256', None) or digest(args.checkpoint), seeds=args.seeds,
                    sampling_config=SAMPLING, passes=1 if arm in ('bagel', 'c2_no_memory', 'parent_no_memory') else 2,
                    early=getattr(args, 'early', 20), loop_layers=getattr(args, 'loop_layers', 28),
                    donor_seed_offset=getattr(args, 'donor_seed_offset', 1), euler_updates=49,
                    weight_source='native_bagel' if arm == 'bagel' else 'interaction_checkpoint' if arm.startswith('interaction_') else 'und_parent',
                    model=str(Path(args.model).resolve()))
    jobs = []
    for name, items in catalog(args).items():
        for item_id, prompt, metadata in items:
            for seed in args.seeds:
                image = str(output/f'images/{name}/seed_{seed:04d}/{item_id}.png')
                jobs.append(dict(benchmark=name, item_id=item_id, seed=seed, prompt=prompt,
                                 metadata=metadata, image=image))
    plan = dict(identity, jobs=jobs)
    plan_path = output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Existing plan differs; choose a new output directory')
    write_json(plan_path, plan)
    reused = 0
    if args.reuse and arm == 'c2_memory' and identity['early'] == 20 and identity['loop_layers'] == 28:
        matches = {(j['prompt'], j['seed']): [] for j in jobs}
        for job in jobs:
            matches[job['prompt'], job['seed']].append(job)
        for path in Path(args.reuse).glob(f'*_{ARM}.json'):
            row = json.loads(path.read_text())
            if (row.get('checkpoint_sha256') != identity['checkpoint_sha256'] or
                row.get('smoke_steps') is not None or row.get('sampling_config') != SAMPLING or
                row.get('early') != 20 or row.get('loop_layers') != 28 or row.get('passes') != 2 or
                row.get('euler_updates') != 49):
                continue
            for job in matches.get((row['prompt'], row['seed']), []):
                image = Path(job['image'])
                if image.exists():
                    continue
                image.parent.mkdir(parents=True, exist_ok=True)
                image.symlink_to(path.with_suffix('.png').resolve())
                write_json(image.with_suffix('.json'), dict(arm=ARM, checkpoint_sha256=identity['checkpoint_sha256'],
                           prompt=job['prompt'], seed=job['seed'], reused_from=str(path)))
                reused += 1
    print(json.dumps(dict(images={name: sum(j['benchmark'] == name for j in jobs) for name in ALL},
                               reused=reused), indent=2), flush=True)


def selected(args):
    plan = json.loads((Path(args.run)/'plan.json').read_text())
    jobs = [j for j in plan['jobs'] if j['benchmark'] in args.benchmarks]
    return plan, jobs


def generate(args, engine=None):
    plan, jobs = selected(args)
    jobs = jobs[args.rank::args.world]
    if args.limit:
        jobs = jobs[:args.limit]
    jobs = [j for j in jobs if not Path(j['image']).is_file()]
    if not jobs:
        print('All assigned images are complete', flush=True)
        return engine
    from qwen_latent_cot.diagnostics.inference import InferenceEngine
    identity = (plan['checkpoint'], plan['checkpoint_sha256'], plan.get('model', args.model))
    if engine is None:
        if digest(plan['checkpoint']) != plan['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after preparation; use the pinned file')
        engine = InferenceEngine.load(plan.get('model', args.model), plan['checkpoint'], args.device)
        engine.benchmark_identity = identity
    elif engine.benchmark_identity != identity:
        raise ValueError('Shared inference engine and arm checkpoint/model differ')
    config = engine.runtime.config
    if {key: config[key] for key in SAMPLING} != plan['sampling_config']:
        raise ValueError('Checkpoint sampling settings differ from the prepared plan')
    arm = plan['arm'] if plan['arm'] != ARM else 'c2_memory'
    worker = Path(args.run)/f'generation/worker_{args.rank:02d}.jsonl'
    worker.parent.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        start = time.time()
        donor_seed = (job['seed']+plan.get('donor_seed_offset', 1)) % (2**63-1)
        image, nodes = engine.generate(job['prompt'], job['seed'], arm=arm,
                                       early=plan['early'], loop_layers=plan['loop_layers'],
                                       donor_seed=donor_seed if arm.endswith('_swap') else None)
        path = Path(job['image'])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp.png')
        image.save(temporary)
        temporary.replace(path)
        result = dict(benchmark=job['benchmark'], item_id=job['item_id'], seed=job['seed'],
                      prompt=job['prompt'], image=job['image'], arm=arm,
                      weight_source='native_bagel' if arm == 'bagel' else 'interaction_checkpoint' if arm.startswith('interaction_') else 'und_parent',
                      checkpoint_sha256=plan['checkpoint_sha256'] if arm != 'bagel' else None,
                      checkpoint_step=engine.step if arm != 'bagel' else None,
                      trained_stage=engine.stage if arm != 'bagel' else None,
                      parent_step=engine.parent_step if arm != 'bagel' else None,
                      parent_stage=engine.parent_stage if arm != 'bagel' else None,
                      control_checkpoint_sha256=plan['checkpoint_sha256'],
                      sampling_config=plan['sampling_config'], early=plan['early'], passes=plan['passes'],
                      loop_layers=plan['loop_layers'],
                      donor_seed=donor_seed if arm.endswith('_swap') else None,
                      seconds=time.time()-start, nodes=nodes)
        write_json(path.with_suffix('.json'), result)
        with worker.open('a') as stream:
            stream.write(json.dumps({k: v for k, v in result.items() if k != 'nodes'})+'\n')
        print(f"{job['benchmark']} {job['item_id']} seed={job['seed']} {result['seconds']:.1f}s", flush=True)
    return engine


def upstream(path, names, namespace):
    nodes = [n for n in ast.parse(Path(path).read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f'Missing upstream definitions: {path}')
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def mplug(model_path, device):
    """Use the ModelScope model and preprocessor without importing unused OFA."""
    import torch
    import typing
    import modelscope
    from PIL import Image
    from modelscope.models.multi_modal.mplug_for_all_tasks import MPlugForAllTasks
    from modelscope.preprocessors.base import Preprocessor
    from modelscope.utils.config import Config
    from modelscope.utils.constant import ModeKeys, ModelFile, Tasks
    if device.startswith('npu'):
        import torch_npu
        torch.npu.set_device(device)
    source = Path(modelscope.__file__).parent/'preprocessors/multi_modal.py'
    ns = dict(vars(typing), torch=torch, Image=Image, osp=os.path, Preprocessor=Preprocessor,
              Config=Config, ModeKeys=ModeKeys, ModelFile=ModelFile, Tasks=Tasks)
    preprocessor = upstream(source, {'MPlugPreprocessor'}, ns)['MPlugPreprocessor'](model_path)
    model = MPlugForAllTasks(model_path).eval().to(device)
    def to_device(data):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        if isinstance(data, Mapping):
            return type(data)({key: to_device(value) for key, value in data.items()})
        return data
    class VQA:
        @torch.inference_mode()
        def vqa(self, image, question):
            return model(to_device(preprocessor({'image': image, 'question': question})))['text']
    return VQA()


def judge(args):
    root = Path(args.tools)
    if args.scorer == 'geneval2':
        from score_explicit_feedback import scorer
        score = scorer(args.judge, root/'GenEval2/evaluation.py', args.device)
        def evaluate(job):
            atoms, hard = score(job['metadata']['vqa_list'], job['image'])
            gm = math.exp(sum(math.log(v) for v in atoms)/len(atoms)) if all(v > 0 for v in atoms) else 0.
            return dict(score=gm, am=sum(atoms)/len(atoms), atoms=atoms, hard_atoms=hard,
                        skills=job['metadata']['skills'])
        return evaluate
    if args.scorer == 'geneval':
        path = root/'geneval/src/geneval/evaluation/evaluate_images.py'
        spec = importlib.util.spec_from_file_location('official_geneval', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.args = SimpleNamespace(options={}, model_path=str(root/'geneval/models'),
            model_config=str(Path(module.mmdet.__file__).parent/'../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'))
        module.object_detector, (module.clip_model, module.transform, module.tokenizer), module.classnames = module.load_models(module.args)
        module.THRESHOLD, module.COUNTING_THRESHOLD = 0.3, 0.9
        module.MAX_OBJECTS, module.NMS_THRESHOLD, module.POSITION_THRESHOLD = 16, 1.0, 0.1
        if module.DEVICE == 'cpu':
            # clip_benchmark defaults to CPU BF16 autocast; Torch 2.1's ARM
            # matmul kernel rejects it. Use the loaded FP32 CLIP weights.
            module.zsc.zero_shot_classifier = partial(module.zsc.zero_shot_classifier, amp=False)
            module.zsc.run_classification = partial(module.zsc.run_classification, amp=False)
        def evaluate(job):
            row = module.evaluate_image(job['image'], job['metadata'])
            return dict(score=float(row['correct']), **row)
        return evaluate
    model = mplug(args.mplug, args.device)
    ns = upstream(root/'dsg/src/dsg/vqa_utils.py', {'calc_vqa_score'},
                  dict(Dict=dict, Any=object, deepcopy=deepcopy))
    # DPG's dependency propagation differs from DSG; retain its own function.
    dpg = upstream(root/'dpg/src/dpg_bench/compute_dpg_bench.py', {'crop_image', 'compute_dpg_one_sample'},
                   dict(Image=__import__('PIL.Image', fromlist=['Image']), osp=os.path))
    details = Path(args.scores or Path(args.run)/'scores')/'dpg_native.txt'
    def evaluate(job):
        meta = job['metadata']
        questions = {int(k): v for k, v in meta['questions'].items()}
        dependencies = {int(k): v for k, v in meta['dependencies'].items()}
        if job['benchmark'] == 'dpg':
            data = {job['item_id']: dict(qid2tuple={int(k): v for k, v in meta['tuples'].items()},
                    qid2question=questions, qid2dependency=dependencies)}
            score, tuples, scores = dpg['compute_dpg_one_sample'](
                SimpleNamespace(pic_num=1, res_path=str(details)), data, job['image'], model, 512)
            return dict(score=score, tuples=tuples, question_scores=scores)
        from PIL import Image
        with Image.open(job['image']) as image:
            answers = {qid: model.vqa(image.convert('RGB'), q) for qid, q in questions.items()}
        valid = not meta['missing_parents'] and not meta['dependency_errors']
        result = ns['calc_vqa_score'](answers, dependencies if valid else None)
        return dict(score=result['average_score_with_dependency'] if valid else None,
                    no_dependency_score=result['average_score_without_dependency'],
                    dependency_valid=valid, missing_parents=meta['missing_parents'],
                    dependency_errors=meta['dependency_errors'], **result)
    return evaluate


def score(args):
    plan, jobs = selected(args)
    allowed = {'dpg', 'dsg'} if args.scorer == 'mplug' else {args.scorer}
    if set(args.benchmarks) - allowed:
        raise ValueError('Selected benchmarks and scorer differ')
    if args.available_only:
        jobs = [j for j in jobs if Path(j['image']).is_file()]
    elif any(not Path(j['image']).is_file() for j in jobs):
        raise ValueError('Generation incomplete; use --available-only for a partial pilot')
    jobs = jobs[args.rank::args.world]
    if args.limit:
        jobs = jobs[:args.limit]
    output = Path(args.scores or Path(args.run)/'scores')/f'{args.scorer}_{args.rank:02d}.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = [r for path in output.parent.glob(f'{args.scorer}_*.jsonl') for r in read_jsonl(path)]
    key = lambda row: (row['benchmark'], row['item_id'], row['seed'])
    done = {key(row) for row in previous}
    jobs = [j for j in jobs if key(j) not in done]
    if not jobs:
        print('No unscored completed images', flush=True)
        return
    from importlib.metadata import version
    provenance = dict(device=args.device, torch_version=version('torch'),
                      transformers_version=version('transformers'))
    if args.scorer == 'geneval2':
        provenance['judge_model'] = args.judge
        provenance['official_source_sha256'] = digest(Path(args.tools)/'GenEval2/evaluation.py')
    elif args.scorer == 'mplug':
        provenance['judge_model'] = args.mplug
        provenance['modelscope_version'] = version('modelscope')
    else:
        provenance['official_source_sha256'] = digest(Path(args.tools)/'geneval/src/geneval/evaluation/evaluate_images.py')
        provenance['clip_cpu_amp'] = False
    evaluate = judge(args)
    with output.open('a') as stream:
        for job in jobs:
            start = time.time()
            row = dict(benchmark=job['benchmark'], item_id=job['item_id'], seed=job['seed'],
                       image=job['image'], arm=plan['arm'],
                       checkpoint_sha256=plan['checkpoint_sha256'] if plan['arm'] != 'bagel' else None,
                       control_checkpoint_sha256=plan['checkpoint_sha256'],
                       scorer=args.scorer, judge_runtime=provenance, **evaluate(job))
            row['seconds'] = time.time()-start
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
            stream.flush()
            print(f"{row['benchmark']} {row['item_id']} score={row['score']} {row['seconds']:.1f}s", flush=True)


def summary(args):
    plan, jobs = selected(args)
    rows = [r for path in Path(args.scores or Path(args.run)/'scores').glob('*.jsonl') for r in read_jsonl(path)]
    result = {}
    mean = lambda xs: sum(xs)/len(xs) if xs else None
    for name in args.benchmarks:
        selected_jobs = [j for j in jobs if j['benchmark'] == name]
        expected = {(j['item_id'], j['seed']) for j in selected_jobs}
        found = [r for r in rows if r['benchmark'] == name]
        keys = [(r['item_id'], r['seed']) for r in found]
        if len(keys) != len(set(keys)) or not set(keys) <= expected:
            raise ValueError(f'Duplicate or unexpected results for {name}')
        report = dict(expected_images=len(expected), generated_images=sum(Path(j['image']).is_file() for j in selected_jobs),
                      scored_images=len(found), coverage_complete=set(keys) == expected,
                      seeds=plan['seeds'], arm=plan['arm'])
        scores = [r['score'] for r in found if r['score'] is not None]
        report['scored_subset_mean_percent'] = 100*mean(scores) if scores else None
        if name in ('geneval', 'geneval-pp') and found:
            categories = defaultdict(list)
            for row in found:
                categories[row['tag']].append(row['score'])
            report['categories_percent'] = {k: 100*mean(v) for k, v in categories.items()}
            report['scored_subset_macro_percent'] = 100*mean([mean(v) for v in categories.values()])
        if name == 'geneval2' and found:
            report['scored_subset_soft_tifa_am_percent'] = 100*mean([r['am'] for r in found])
        if name == 'dsg' and found:
            report['dependency_valid_images'] = len(scores)
            report['dependency_coverage_complete'] = len(scores) == len(expected)
            report['scored_subset_without_dependency_percent'] = 100*mean([r['no_dependency_score'] for r in found])
        result[name] = report
    write_json(Path(args.scores or args.run)/'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


def export(args):
    """Create the upstream API evaluators' image layouts after generation."""
    plan, jobs = selected(args)
    if any(not Path(j['image']).is_file() for j in jobs):
        raise ValueError('Generate all selected images before exporting')
    root = Path(args.run)/'exports'
    notes = []
    for name in args.benchmarks:
        items = [j for j in jobs if j['benchmark'] == name]
        if name == 'geneval-pp':
            for seed in plan['seeds']:
                folder = root/name/f'seed_{seed:04d}'
                folder.mkdir(parents=True, exist_ok=True)
                rows = [j for j in items if j['seed'] == seed]
                (folder/'Geneval++.jsonl').write_text(''.join(json.dumps(j['metadata'])+'\n' for j in rows))
                for i, job in enumerate(rows):
                    path = folder/f'image/{i+1}.jpg'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not path.exists():
                        # PIL decodes the PNG pixels; upstream encodes JPEG once.
                        path.symlink_to(job['image'])
        elif name.startswith('unigen-'):
            folder = root/name
            folder.mkdir(parents=True, exist_ok=True)
            rows = [dict(j['metadata']) for j in items if j['seed'] == plan['seeds'][0]]
            lang = name.split('-')[1]
            field = f'sub_dims_{lang}'
            for row in rows:
                try:
                    json.loads(row[field])
                except json.JSONDecodeError:
                    # One upstream en_long row lacks this separating comma.
                    original = row[field]
                    repaired = original.replace('"Grammar - Negation" "Attribute - Color"',
                                                '"Grammar - Negation", "Attribute - Color"')
                    json.loads(repaired)
                    row[field] = repaired
                    notes.append(dict(benchmark=name, index=row['index'], field=field,
                                      original=original, repaired=repaired))
            with (folder/'prompts.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            for job in items:
                sample_index = plan['seeds'].index(job['seed'])
                path = folder/f"{job['item_id']}_{sample_index}.png"
                if not path.exists():
                    path.symlink_to(job['image'])
        else:
            raise ValueError('Export supports geneval-pp and the unigen variants')
    write_json(root/'format_repairs.json', notes)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'generate', 'score', 'summary', 'export'])
    parser.add_argument('--run', required=True)
    parser.add_argument('--tools', default='/root/npu-eval-tools')
    parser.add_argument('--geneval2', default='data/geneval2_800.jsonl')
    parser.add_argument('--checkpoint', default='/root/outputs/feedback_nativeund_attn_full_20k/step_002000.pt')
    parser.add_argument('--model', default='/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--benchmarks', nargs='+', choices=ALL, default=READY)
    parser.add_argument('--reuse')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=1)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--scorer', choices=['geneval', 'geneval2', 'mplug'])
    parser.add_argument('--scores', help='score directory; default: RUN/scores')
    parser.add_argument('--judge', default='/data/zyd_workspace/models/Qwen3-VL-8B-Instruct')
    parser.add_argument('--mplug', default='/root/npu-eval-tools/dpg/models/mplug-large')
    parser.add_argument('--available-only', action='store_true')
    from qwen_latent_cot.diagnostics.inference import ALL_ARMS
    parser.add_argument('--arm', choices=ALL_ARMS,
                        help='default: trained interaction, or C2 memory for UND-only checkpoints')
    parser.add_argument('--early', type=int, default=20)
    parser.add_argument('--loop-layers', type=int, default=28)
    parser.add_argument('--donor-seed-offset', type=int, default=1)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or not 0 <= args.rank < args.world:
        parser.error('Seeds must be unique and rank must be in [0, world)')
    if args.command == 'score' and not args.scorer:
        parser.error('--scorer is required for score')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
