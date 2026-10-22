"""Single-arm BAGEL generation adapter for the existing six-benchmark package."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

COUNTS = dict(geneval=553, geneval_pp=280, t2i_compbench_pp=2400,
              dpg=1065, t2i_corebench=1080, wise_verified=1000)
BASE = Path('/data/zyd_workspace/bagel-LatentCoT')
PYTHON = str(BASE/'venvs/bagel-NPU/bin/python')
SOURCE = Path('/data/250010019/unimoe_r3/outputs/0_qwen38max_fullbench_20261002')
PACKAGE = Path('/data/250010019/unimoe_r3/outputs/0_eval62326_fourarm_handoff_20261018')
CHECKPOINT = BASE/'outputs/und_guidance_safe_64npu_20261022_5000/step_002000.pt'
MODEL = BASE/'models/Bagel-7B-MoT'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    os.replace(temporary, path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare(run, code):
    if run.exists(): raise FileExistsError(run)
    spec = importlib.util.spec_from_file_location('source_prepare', PACKAGE/'code/prepare_bundle.py')
    source = importlib.util.module_from_spec(spec); spec.loader.exec_module(source)
    combined = []; files = {}; hashes = {}
    for bench, expected in COUNTS.items():
        original = SOURCE/'manifests'/f'{bench}.jsonl'; values = rows(original)
        if len(values) != expected or len({str(x['record_id']) for x in values}) != expected:
            raise ValueError('Incomplete/duplicate source: '+bench)
        normal = [source.normalize(x, bench, len(combined)+i) for i, x in enumerate(values)]
        combined.extend(normal)
        files[f'manifests/{bench}.jsonl'] = ''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in normal)
        files[f'score_manifests/{bench}.jsonl'] = original.read_text(); hashes[str(original)] = digest(original)
        if bench.endswith('_pp'):
            binding = SOURCE/'bindings'/f'{bench}_scorer_manifest.json'
            if len(json.loads(binding.read_text())['samples']) != expected: raise ValueError(bench)
            files[f'score_manifests/{bench}.json'] = binding.read_text(); hashes[str(binding)] = digest(binding)
    if len({x['output_rel'] for x in combined}) != sum(COUNTS.values()): raise ValueError('Output collision')
    files['manifests/combined.jsonl'] = ''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in combined)
    run.mkdir()
    for relative, content in files.items():
        path = run/relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
    atomic(run/'PLAN.json', dict(arm='ours_step2000', benchmarks=COUNTS, total=len(combined),
        checkpoint=str(CHECKPOINT), checkpoint_sha256=digest(CHECKPOINT), code=str(code),
        source_revision=(code/'DEPLOYED_COMMIT').read_text().strip(), adapter_sha256=digest(__file__),
        model=str(MODEL), source_manifests_sha256=hashes, samples_per_prompt=1,
        sampler='checkpoint config; corrected; unchanged correction.sampling.sample',
        judge='local Qwen3.8-27B internal protocol', official_leaderboard_comparable=False))
    print(json.dumps(dict(prepared=str(run), counts=COUNTS, total=len(combined))), flush=True)


def report_path(image): return image.with_suffix('.report.json')


def worker(run, rank, world, limit):
    import fcntl
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from qwen_latent_cot.correction import checkpoint
    from qwen_latent_cot.correction.model import SpatialCorrector
    from qwen_latent_cot.correction.runtime import Runtime
    from qwen_latent_cot.correction.objective import guidance_safe
    from qwen_latent_cot.correction.sampling import sample
    from qwen_latent_cot.bagel.accelerator import set_device
    from qwen_latent_cot.bagel.backbone import load_native
    import random
    import torch
    plan = json.loads((run/'PLAN.json').read_text()); saved = checkpoint.read(plan['checkpoint'])
    cfg = dict(saved['config'])
    if saved['step'] != 2000 or not guidance_safe(cfg): raise ValueError('Expected guidance-safe Step2000')
    if saved['provenance']['model_path'] != str(Path(plan['model']).resolve()): raise ValueError('Model path differs')
    lock_path = run/'logs'/f'rank{rank:02d}.lock'; lock_path.parent.mkdir(exist_ok=True)
    lock = lock_path.open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    device = set_device('npu:0'); torch.manual_seed(cfg['seed']); random.seed(cfg['seed'])
    corrector = SpatialCorrector(Runtime(load_native(plan['model'], device, cfg['timestep_shift']), cfg), cfg).eval()
    checkpoint.apply(saved, corrector)
    identity = dict(checkpoint_sha256=plan['checkpoint_sha256'], source_revision=plan['source_revision'],
                    adapter_sha256=plan['adapter_sha256'], config=cfg, arm=plan['arm'])
    assigned = [x for x in rows(run/'manifests/combined.jsonl') if x['global_index'] % world == rank]
    if limit: assigned = assigned[:limit]
    print(json.dumps(dict(event='loaded', rank=rank, count=len(assigned), config=cfg)), flush=True)
    for index, item in enumerate(assigned):
        image_path = run/'images'/item['output_rel']; receipt = report_path(image_path)
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old['identity'] != identity or old['input'] != item: raise ValueError('Resume identity differs')
            with Image.open(image_path) as image: image.verify()
            if digest(image_path) != old['image_sha256']: raise ValueError('Resume image integrity differs')
            continue
        started = time.time(); image, trace = sample(corrector, item['instruction'], item['seed'])
        if len(trace) != cfg['sampler_steps']-1 or sum(x['und_passes'] for x in trace) != 25:
            raise ValueError('Unexpected sampler contract')
        if any(x.get('guidance_projection_applied', 0) < -1e-5 for x in trace):
            raise ValueError('Applied anti-guidance projection')
        image_path.parent.mkdir(parents=True, exist_ok=True)
        partial = image_path.with_suffix('.partial.png'); image.save(partial); os.replace(partial, image_path)
        atomic(receipt, dict(status='ok', identity=identity, input=item, trace=trace,
            image=str(image_path), image_sha256=digest(image_path), seconds=time.time()-started))
        value = dict(event='image', rank=rank, completed=index+1, assigned=len(assigned),
                     global_index=item['global_index'], benchmark=item['benchmark'], seconds=time.time()-started)
        atomic(run/'logs'/f'rank{rank:02d}.progress.json', value); print(json.dumps(value), flush=True)
    atomic(run/'logs'/f'rank{rank:02d}.complete.json', dict(count=len(assigned), rank=rank))


def node(run, node_rank):
    processes = []
    for local in range(16):
        rank = node_rank*16+local
        env = os.environ.copy(); env.update(ASCEND_RT_VISIBLE_DEVICES=str(local), OMP_NUM_THREADS='1',
            TOKENIZERS_PARALLELISM='false', PYTORCH_NPU_ALLOC_CONF='expandable_segments:True')
        log = (run/'logs'/f'rank{rank:02d}.log').open('ab')
        process = subprocess.Popen([PYTHON, __file__, 'worker', '--run', str(run), '--rank', str(rank), '--world', '64'],
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
        processes.append((rank, process)); log.close()
    atomic(run/'logs'/f'node{node_rank}.workers.json', {str(rank): p.pid for rank, p in processes})
    results = {str(rank): p.wait() for rank, p in processes}
    atomic(run/'logs'/f'node{node_rank}.exit.json', results)
    if any(results.values()): raise RuntimeError('Worker failed: '+str(results))


def launch(run):
    hosts = os.environ['VC_WORKER_HOSTS'].split(',')
    if len(hosts) != 4: raise ValueError('Requires four nodes')
    def remote(host, source):
        command = [PYTHON, '-'] if host == hosts[0] else ['ssh', '-o', 'BatchMode=yes', '-p', '2222',
            'root@'+host, shlex.join([PYTHON, '-'])]
        value = subprocess.run(command, input=source, text=True, capture_output=True, timeout=120)
        if value.returncode: raise RuntimeError(value.stdout+value.stderr)
        return json.loads(value.stdout)
    # Never evict: npu-smi must show no running NPU processes before using these nodes.
    scan = """import subprocess,json,os
p=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True)
print(json.dumps(dict(host=os.uname().nodename,npu_smi=p.stdout)))
"""
    with ThreadPoolExecutor(4) as pool: audit = list(pool.map(lambda h: remote(h, scan), hosts))
    atomic(run/'PREFLIGHT.json', audit)
    if not all(x['npu_smi'].count('No running processes found in NPU ') == 8 for x in audit):
        raise RuntimeError('NPU idle status not confirmed; inspect PREFLIGHT.json')
    (run/'logs').mkdir(exist_ok=True)
    def start(pair):
        i, host = pair
        source = f'''import subprocess,json,os
from pathlib import Path
log=Path({str(run/'logs')!r})/"node{i}.log"
with log.open('xb') as stream:
 p=subprocess.Popen({[PYTHON, str(Path(__file__).resolve()), 'node', '--run', str(run), '--rank', str(i)]!r},stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(node={i},host=os.uname().nodename,pid=p.pid)))
'''
        return remote(host, source)
    with ThreadPoolExecutor(4) as pool: started = list(pool.map(start, enumerate(hosts)))
    atomic(run/'LAUNCH.json', started); print(json.dumps(started), flush=True)


def status(run):
    counts = {b: 0 for b in COUNTS}; seconds = []; failures = []
    for p in (run/'images').rglob('*.report.json'):
        receipt = json.loads(p.read_text()); counts[receipt['input']['benchmark']] += 1; seconds.append(receipt['seconds'])
    for p in (run/'logs').glob('node*.exit.json'):
        failures.extend((rank, code) for rank, code in json.loads(p.read_text()).items() if code)
    result = dict(counts=counts, expected=COUNTS, completed=sum(counts.values()), total=sum(COUNTS.values()),
        workers_complete=len(list((run/'logs').glob('rank*.complete.json'))), failures=failures,
        mean_seconds=sum(seconds)/len(seconds) if seconds else None)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','worker','node','launch','status'])
    parser.add_argument('--run', type=Path, required=True); parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=64); parser.add_argument('--limit', type=int, default=0)
    a = parser.parse_args(); run = a.run.resolve()
    if a.action == 'prepare': prepare(run, Path(__file__).resolve().parents[1])
    elif a.action == 'worker': worker(run, a.rank, a.world, a.limit)
    elif a.action == 'node': node(run, a.rank)
    elif a.action == 'launch': launch(run)
    else: status(run)


if __name__ == '__main__': main()
