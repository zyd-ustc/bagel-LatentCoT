"""Run the existing C0 command on 16 NPUs, one caption per device."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data', default=str(root/'data/c0_text16.jsonl'))
    parser.add_argument('--config', default=str(root/'configs/memory_topology.json'))
    args = parser.parse_args()
    data = Path(args.data).resolve()
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    if len(rows) != 16 or any(row['split'] != 'val' for row in rows):
        raise ValueError('C0 requires exactly 16 validation captions')
    if any(row.get('image') or row.get('state') or row.get('swap_id') for row in rows):
        raise ValueError('C0 shards accept independent text-only calibration rows')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output/'inputs'
    inputs.mkdir()
    workers = output/'workers'
    workers.mkdir()
    processes = []
    logs = []
    started = time.monotonic()
    try:
        for device, row in enumerate(rows):
            shard = inputs/f'device_{device:02d}.jsonl'
            shard.write_text(json.dumps(row)+'\n')
            env = os.environ.copy()
            env.pop('CUDA_VISIBLE_DEVICES', None)
            env['ASCEND_RT_VISIBLE_DEVICES'] = str(device)
            env.setdefault('OMP_NUM_THREADS', '4')
            env['PYTHONPATH'] = str(root)+os.pathsep+env.get('PYTHONPATH', '')
            log = (workers/f'device_{device:02d}.log').open('w')
            logs.append(log)
            command = [sys.executable, '-m', 'qwen_latent_cot.diagnostics.cli', 'calibrate',
                       '--model', str(Path(args.model).resolve()), '--data', str(shard),
                       '--config', str(Path(args.config).resolve()), '--device', 'npu:0',
                       '--split', 'val', '--output', str(workers/f'device_{device:02d}')]
            processes.append(subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT))
        print(f'Launched 16 C0 workers; logs: {workers}', flush=True)
        while any(process.poll() is None for process in processes):
            failed = [i for i, process in enumerate(processes) if process.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'C0 workers failed: {failed}; inspect their logs')
            time.sleep(1)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError('C0 worker failed; inspect worker logs')
        calibrations = [json.loads((workers/f'device_{i:02d}/calibration.json').read_text()) for i in range(16)]
        first = calibrations[0]
        for other in calibrations[1:]:
            if any(first[key] != other[key] for key in ('model_files', 'config_hash', 'instruction', 'instruction_token_ids')):
                raise RuntimeError('C0 workers used different models or decode protocols')
        captions = []
        for i in range(16):
            captions.extend(json.loads(line) for line in (workers/f'device_{i:02d}/captions.jsonl').read_text().splitlines())
        if len(captions) != 16 or {r['sample_id'] for r in captions} != {r['sample_id'] for r in rows}:
            raise RuntimeError('C0 aggregation does not contain all 16 captions')
        (output/'captions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in captions))
        first.update(passed=False, samples=16, devices=list(range(16)),
                     status='Inspect all 16 reference/free-text pairs, then record passed=true and reviewer')
        (output/'calibration.json').write_text(json.dumps(first, indent=2)+'\n')
        summary = dict(samples=16, nll=sum(r['nll'] for r in captions)/16,
                       exact_text_matches=sum(r['text'].strip() == r['caption'].strip() for r in captions),
                       elapsed_seconds=time.monotonic()-started, passed=False,
                       worker_manifests=[str(workers/f'device_{i:02d}/manifest.json') for i in range(16)])
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(f'C0 completed: {output}/captions.jsonl; manual review pending', flush=True)
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


if __name__ == '__main__':
    main()
