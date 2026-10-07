import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def source_hash(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob('*.py')):
        if any(s in ('.git', '__pycache__', '.venv') for s in p.parts): continue
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def identity(row):
    return row['prompt_id'], row['seed']


def load_manifests(paths):
    records, provenance, seen = [], [], set()
    for path in paths:
        run = json.loads((Path(path).parent/'run.json').read_text())
        provenance.append(run)
        for row in read_jsonl(path):
            key = (row['arm'], *identity(row))
            if key in seen: raise ValueError('duplicate arm/prompt/seed record')
            seen.add(key); records.append(row)
    if not records: raise ValueError('empty manifests')
    for key in ('schema', 'architecture', 'source_sha256', 'model_sha256', 'benchmark_sha256', 'sampling', 'loop', 'seeds', 'arms', 'prompt_ids', 'stage', 'diagnostics', 'memory_topologies', 'loop_depths', 'arm_configs'):
        if any(run.get(key) != provenance[0].get(key) for run in provenance):
            raise ValueError(f'incompatible shard provenance: {key}')
    expected = {(p, s) for p in provenance[0]['prompt_ids'] for s in provenance[0]['seeds']}
    for arm in provenance[0]['arms']:
        actual = {identity(r) for r in records if r['arm'] == arm}
        if actual != expected: raise ValueError(f'incomplete arm coverage: {arm}')
    for row in records:
        if row['arm'] not in provenance[0]['arms']: raise ValueError('unknown arm')
        if row['valid_file'] and sha256(row['path']) != row['image_sha256']:
            raise ValueError('image changed after generation')
    return records, provenance[0]
