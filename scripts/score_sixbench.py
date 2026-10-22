"""Bind BAGEL PNGs to the original six local27B scorers without changing rubrics."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from urllib.request import urlopen
from benchmark_sixbench import COUNTS, SOURCE, PACKAGE, atomic, digest, rows, report_path

ARM = 'ours_step2000'
NAMESPACE = 'bagel_guidance_safe_step2000_sixbench_20261022_v1'
CLIENT = '/data/250010019/unimoe_vg_16npu/runtime/conda_training/bin/python'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prepare(run, api):
    helper = load(PACKAGE/'code/prepare_sixbench_scoring.py', '_original_preparation')
    helper.endpoint(api)
    original = json.loads((SOURCE/'qwen38_27b_rescore_config.json').read_text())
    root = run/'scoring_qwen27b_v1'
    config = {k: v for k, v in original.items() if k not in ('jobs','source_max_config_sha256')}
    plan = json.loads((run/'PLAN.json').read_text())
    config.update(schema='bagel_step2000_sixbench_v1', api_base=api, python=CLIENT,
        repo_root=str(SOURCE.parents[1]), control_root=str(root/'control'), jobs=[], arms=[ARM],
        workers_per_cell=8, max_global_inflight=8, expected_image_records=sum(COUNTS.values()),
        protocol='bagel_step2000_original_sixbench_rubric_local27b_v1',
        artifact_prefix='BAGEL_STEP2000_27B', official_comparable=False, judge_kind='local_vllm',
        external_api_calls=False, proxy_namespace=NAMESPACE,
        proxy_protocol='qwen38_27b_local_exact_rubric_wire_v1', generation=plan,
        reason_not_official='Original rubric and aggregation, with independent local27B judge and BAGEL sampling.',
        payload_cache='New run exact-wire cache, no historical votes or training reward gateway')
    inputs = set()
    for bench in COUNTS:
        helper.manifest_pair(run, bench)
        template = next(x for x in original['jobs'] if x['benchmark'] == bench and x['arm'] == 'friend17500')
        job = json.loads(json.dumps(template).replace('/mnt/sfs/', '/data/'))
        view = root/'views'/ARM/bench
        job.update(id=ARM+':'+bench, arm=ARM, display_name='BAGEL guidance-safe Step2000',
            score_root=str(root/'scores'/ARM/bench), ready_marker=str(view/'VIEW_COMPLETE.json'),
            manifest=str(run/'score_manifests'/(bench+('.json' if bench.endswith('_pp') else '.jsonl'))))
        for key in ('ready_markers','pipeline_root','image_root'): job.pop(key, None)
        job['pipeline_root' if bench.endswith('_pp') else 'image_root'] = str(view)
        config['jobs'].append(job)
        inputs.update((run/'manifests'/f'{bench}.jsonl', run/'score_manifests'/f'{bench}.jsonl', Path(job['manifest'])))
    dependencies = helper.source_dependencies(SOURCE, config)
    config['frozen_files_sha256'] = {str(p): digest(p) for p in sorted(inputs | dependencies | {Path(__file__)})}
    helper.locked_json(root/'scoring_config.json', config)
    print(json.dumps(dict(config=str(root/'scoring_config.json'), cells=6, images=6378, api_calls=0)))


def validate(config):
    if config['arms'] != [ARM] or len(config['jobs']) != 6: raise ValueError('Wrong run contract')
    for path, expected in config['frozen_files_sha256'].items():
        if digest(path) != expected: raise ValueError('Frozen scoring source/input changed: '+path)


def materialize(run, config):
    from PIL import Image
    helper = load(PACKAGE/'code/prepare_sixbench_scoring.py', '_original_preparation')
    plan = json.loads((run/'PLAN.json').read_text())
    for job in config['jobs']:
        bench = job['benchmark']; generated, scoring = helper.manifest_pair(run, bench)
        bindings = []; view = Path(job.get('pipeline_root', job.get('image_root')))
        for item in generated:
            image = run/'images'/item['output_rel']; receipt = report_path(image)
            value = json.loads(receipt.read_text()); identity = value['identity']
            if value['status'] != 'ok' or value['input'] != item: raise ValueError('Image input mismatch')
            for key in ('checkpoint_sha256','source_revision','adapter_sha256','arm'):
                if identity[key] != plan[key]: raise ValueError('Image generation identity mismatch: '+key)
            with Image.open(image) as png:
                if png.size != (512, 512): raise ValueError('Wrong PNG geometry')
                png.verify()
            if digest(image) != value['image_sha256']: raise ValueError('PNG/receipt integrity differs')
            bindings.append(dict(record_id=item['record_id'], index=item.get('index'), image=str(image),
                image_sha256=value['image_sha256'], report=str(receipt), report_sha256=digest(receipt),
                scorer_output_rel=scoring[item['record_id']]['output_rel']))
        for b in bindings:
            if bench.endswith('_pp'):
                helper.locked_json(view/'cases'/f"case-{b['index']:03d}"/'report.json',
                    dict(schema='bagel_final_image_scorer_view_v1', status='ok', final_image=b['image'],
                         final_image_sha256=b['image_sha256'], provenance=b))
            else: helper.symlink_locked(view/b['scorer_output_rel'], b['image'])
        helper.locked_json(view/'VIEW_COMPLETE.json', dict(complete=True, status='complete',
            arm=ARM, benchmark=bench, expected=COUNTS[bench], present=len(bindings), bindings=bindings))
    print(json.dumps(dict(materialized=6, images=6378, api_calls=0)))


def runner():
    sys.path.insert(0, str(SOURCE/'code'))
    module = load(SOURCE/'code/run_max_rescore.py', '_original_runner')
    module.ARMS = (ARM,); module.BENCHMARKS = tuple(COUNTS); module.EXPECTED_COUNTS = COUNTS
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','materialize','dry-run','score','status'])
    p.add_argument('--run', type=Path, required=True); p.add_argument('--api-base', default='http://127.0.0.1:18364/v1')
    a = p.parse_args(); run = a.run.resolve(); path = run/'scoring_qwen27b_v1/scoring_config.json'
    if a.action == 'prepare': return prepare(run, a.api_base)
    config = json.loads(path.read_text()); validate(config)
    if a.action == 'materialize': return materialize(run, config)
    source = runner()
    if a.action == 'status':
        print(json.dumps({j['benchmark']: source.check_complete(j) for j in config['jobs']}, indent=2)); return
    if a.action == 'score':
        materialize(run, config)
        with urlopen(config['api_base'].removesuffix('/v1')+'/health', timeout=10) as response: health = json.load(response)
        if any(health.get(k) != config[v] for k, v in
               [('model','model'),('namespace','proxy_namespace'),('protocol_version','proxy_protocol')]):
            raise ValueError('Wrong independent local27B proxy')
        if health.get('auth_blocked'): raise ValueError('Judge unavailable')
    sys.argv = [source.__file__, '--config', str(path), '--parallel-cells', '2', '--max-passes', '3']
    if a.action == 'dry-run': sys.argv.append('--dry-run')
    source.main()


if __name__ == '__main__': main()
