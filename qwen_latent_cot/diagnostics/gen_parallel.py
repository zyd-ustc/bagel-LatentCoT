"""Sixteen independent NPU workers for eight complete color pairs and two seeds."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time

DATA_ROOT = Path('/data/zyd_workspace/bagel-LatentCoT/datasets')
CHECKPOINT = '/root/outputs/c1_pilot_20261012_113553/C/best.pt'
ARMS = {'G-base', 'G-layout', 'G-fixed', 'G-neutral', 'G-text', 'G-TC', 'G-swap-TC'}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def prepare(data, config, output, model, checkpoint, c0):
    from .data import load_dataset
    data = Path(data).resolve()
    if not data.is_file():
        raise FileNotFoundError(f'Missing verified color manifest: {data}; raw CoRT/edit manifests are not caption ground truth')
    rows = load_dataset(data)
    if len(rows) != 16 or any(row['split'] != 'test' or row['label_type'] != 'clean'
                              or not row.get('image') or row.get('state') for row in rows):
        raise ValueError('GEN color pilot requires exactly 16 verified clean test images, not trajectory states')
    groups = {}
    for row in rows:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', row['sample_id']):
            raise ValueError('sample_id must be a safe filename identifier')
        if not row.get('pair_id') or not row.get('prompt'):
            raise ValueError('every color row requires pair_id and the same color-free pair prompt')
        groups.setdefault(row['pair_id'], []).append(row)
    if len(groups) != 8:
        raise ValueError('GEN pilot requires eight scene pairs')
    for pair in groups.values():
        if len(pair) != 2 or pair[0].get('swap_id') != pair[1]['sample_id'] or pair[1].get('swap_id') != pair[0]['sample_id']:
            raise ValueError('each pair must contain two reciprocal swap partners')
        colors = []
        for row in pair:
            selected = {alias.lower() for fact in row['facts'] if fact['category'] == 'attribute'
                        for alias in fact['aliases'] if alias.lower() in ('red', 'blue')}
            if len(selected) != 1:
                raise ValueError('each row must assert exactly one target color: red or blue')
            asserted = set(re.findall(r'\b(?:red|blue)\b', row['caption'].lower()))
            if asserted != selected:
                raise ValueError('caption and target color fact disagree')
            colors.extend(selected)
        if set(colors) != {'red', 'blue'}:
            raise ValueError('each pair must contrast red and blue')
        if re.search(r'\b(?:red|blue)\b', pair[0]['prompt'].lower()):
            raise ValueError('original pair prompt must omit target color')
    calibration = json.loads(Path(c0).read_text())
    if calibration.get('passed') is not True:
        raise ValueError('C0 semantic readout gate has not been accepted')
    common = json.loads(Path(config).read_text())
    common.update(eval_t=[0.5, 1.0], noise_seeds=[0, 1])
    checkpoint_hash = digest(checkpoint)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'inputs').mkdir()
    (output/'workers').mkdir()
    tasks = []
    for pair_id, pair in sorted(groups.items()):
        for seed in (0, 1):
            device = len(tasks)
            name = f'device_{device:02d}'
            shard = output/'inputs'/f'{name}.jsonl'
            shard.write_text(''.join(json.dumps(row)+'\n' for row in pair))
            worker_config = dict(common, noise_seeds=[seed])
            config_path = output/'inputs'/f'{name}.json'
            write_json(config_path, worker_config)
            tasks.append(dict(device=device, pair_id=pair_id, seed=seed,
                              sample_ids=[row['sample_id'] for row in pair],
                              data=str(shard), config=str(config_path),
                              output=str(output/'workers'/name)))
    plan = dict(data=str(data), dataset_hash=digest(data), common_config=common,
                model=str(Path(model).resolve()), checkpoint=str(Path(checkpoint).resolve()),
                checkpoint_hash=checkpoint_hash, c0=str(Path(c0).resolve()), tasks=tasks,
                samples=16, states=32, expected_records=224, topology='C', continuation=False)
    write_json(output/'plan.json', plan)
    return plan


def aggregate(output, plan):
    output = Path(output)
    records = []; timings = []; manifests = []
    reference = None
    for task in plan['tasks']:
        worker = Path(task['output'])
        manifest = json.loads((worker/'manifest.json').read_text())
        contract = {key: manifest[key] for key in ('model_files', 'source_hashes', 'depth', 'slots', 'topology', 'checkpoints')}
        if reference is not None and contract != reference:
            raise ValueError('workers used different models, source, capacity or checkpoints')
        reference = contract
        expected_config = dict(plan['common_config'], noise_seeds=[task['seed']])
        if manifest['config'] != expected_config or manifest['topology'] != 'C' or not manifest['status'].startswith('executed'):
            raise ValueError('worker configuration/status differs from the GEN plan')
        if manifest['dataset_hash'] != digest(task['data']) or manifest['checkpoints']['text_checkpoint'] != plan['checkpoint_hash']:
            raise ValueError('worker dataset or retained checkpoint changed')
        rows = read_jsonl(worker/'gen.jsonl')
        keys = {(row['sample_id'], row['t'], row['noise_seed'], row['arm']) for row in rows}
        expected = {(sample, 0.5, task['seed'], arm) for sample in task['sample_ids'] for arm in ARMS}
        if len(rows) != 14 or keys != expected or any(row['frozen_guard'] != 'passed' for row in rows):
            raise ValueError('worker did not produce every requested sample/arm at the frozen fixed state')
        for row in rows:
            name = f"{row['sample_id']}_000_{row['arm']}_x0_hat.png"
            image = worker/name
            if not image.is_file():
                raise FileNotFoundError(image)
            row.update(worker_device=task['device'], predicted_image=str(image.relative_to(output)),
                       image_kind='x0_hat', evaluation_pair_id=task['pair_id'])
        records.extend(rows)
        timings.extend(dict(row, worker_device=task['device']) for row in read_jsonl(worker/'timing.jsonl'))
        manifests.append(str((worker/'manifest.json').relative_to(output)))
    if len(records) != plan['expected_records']:
        raise ValueError('GEN aggregation is incomplete')
    (output/'gen.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in records))
    (output/'timing.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in timings))
    groups = {}
    for row in records:
        groups.setdefault(row['arm'], []).append(row)
    numeric = []
    for arm, rows in sorted(groups.items()):
        numeric.append(dict(arm=arm, states=len(rows),
                            conditional_difference=sum(row['conditional']['normalized_difference'] for row in rows)/len(rows),
                            cfg_difference=sum(row['cfg']['normalized_difference'] for row in rows)/len(rows),
                            cfg_norm_ratio=sum(row['cfg']['norm_ratio'] for row in rows)/len(rows),
                            repeat_cfg_error=max(row['repeat_error']['cfg']['normalized_difference'] for row in rows)))
    blind = list(records)
    random.Random(plan['common_config']['seed']).shuffle(blind)
    (output/'local_blind').mkdir()
    mapping = []
    for index, row in enumerate(blind):
        name = f'local_blind/{index:04d}.png'
        shutil.copyfile(output/row['predicted_image'], output/name)
        mapping.append(dict(blind_file=name, sample_id=row['sample_id'], pair_id=row['evaluation_pair_id'],
                            seed=row['noise_seed'], t=row['t'], arm=row['arm'], image_kind='x0_hat',
                            supplementary_caption=row['supplementary_caption'], target_facts=row['target_facts']))
    write_json(output/'local_blind_key.json', mapping)
    with (output/'local_blind_annotations.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['blind_file', 'observed_color', 'unchanged_facts', 'quality', 'artifacts', 'notes'])
        writer.writerows([[row['blind_file'], '', '', '', '', ''] for row in mapping])
    write_json(output/'summary.json', dict(samples=16, scene_pairs=8, states=32, records=len(records),
               topology='C', checkpoint_hash=plan['checkpoint_hash'], numeric_by_arm=numeric,
               frozen_guard='passed', continuation=False, image_kind='x0_hat',
               semantics='pending factual review; numeric sensitivity is not semantic success',
               worker_manifests=manifests))
    write_json(output/'manifest.json', dict(reference, dataset_hash=plan['dataset_hash'], config=plan['common_config'],
               plan='plan.json', worker_manifests=manifests, status='executed; local semantic review pending'))


def launch(output, plan):
    root = Path(__file__).resolve().parents[2]
    processes = []; logs = []
    try:
        for task in plan['tasks']:
            env = os.environ.copy()
            env.pop('CUDA_VISIBLE_DEVICES', None)
            env['ASCEND_RT_VISIBLE_DEVICES'] = str(task['device'])
            env.setdefault('OMP_NUM_THREADS', '4')
            env['PYTHONPATH'] = str(root)+os.pathsep+env.get('PYTHONPATH', '')
            log = (Path(output)/'workers'/f'device_{task["device"]:02d}.log').open('w')
            logs.append(log)
            command = [sys.executable, '-u', '-m', 'qwen_latent_cot.diagnostics.cli', 'gen',
                       '--model', plan['model'], '--data', task['data'], '--config', task['config'],
                       '--device', 'npu:0', '--split', 'test', '--topology', 'C',
                       '--text-checkpoint', plan['checkpoint'], '--c0-pass', plan['c0'], '--output', task['output']]
            processes.append(subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT))
        print(f'Launched 16 GEN workers; logs: {output}/workers', flush=True)
        while any(process.poll() is None for process in processes):
            failed = [i for i, process in enumerate(processes) if process.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'GEN workers failed: {failed}; inspect worker logs')
            time.sleep(1)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError('GEN worker failed; inspect worker logs')
        aggregate(output, plan)
        print(f'GEN completed: {output}/summary.json; local semantic review pending', flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default=str(DATA_ROOT))
    parser.add_argument('--data', help='verified 16-image color manifest; defaults to DATASET_ROOT/memory_gen_color16.jsonl')
    parser.add_argument('--model', default='/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT')
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--c0-pass', default='/root/outputs/c0_text16_20261012_111241/calibration.json')
    parser.add_argument('--config', default=str(root/'configs/c1_pilot.json'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--prepare-only', action='store_true', help='write shards and plan without loading BAGEL')
    args = parser.parse_args()
    data = args.data or str(Path(args.dataset_root)/'memory_gen_color16.jsonl')
    plan = prepare(data, args.config, args.output, args.model, args.checkpoint, args.c0_pass)
    print(f'Prepared 16 tasks, 32 states and 224 prediction images: {args.output}', flush=True)
    if not args.prepare_only:
        launch(Path(args.output).resolve(), plan)


if __name__ == '__main__':
    main()
