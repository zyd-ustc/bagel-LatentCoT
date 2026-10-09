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


def validate_observation_record(row):
    events=row['observation_events']
    if row['arm']=='BASE':
        if events:raise ValueError('Base has an observation update')
        return
    # A failed decode can stop before the selected step. Invalid arms remain in
    # quality/invalid-rate reporting, without inventing a successful update.
    if row['valid_file'] and len(events)!=1:raise ValueError('expected one observation event')
    for event in events:
        if event['step_index']!=row['observation_step'] or event['updates']!=1:
            raise ValueError('observation step/update count differs')
        if not event['x_t_unchanged'] or not event['timestep_unchanged']:
            raise ValueError('observation changed sampler state')
        for path,digest in (('source_preview','preview_sha256'),('saved_state','state_sha256')):
            if sha256(event[path])!=event[digest]:raise ValueError('observation artifact changed: '+path)


def validate_observation_pair(row,other):
    if row['noise_sha256']!=other['noise_sha256']:raise ValueError('paired initial noise differs')
    if not row['observation_events'] or not other['observation_events']:return
    a=row['observation_events'][0];b=other['observation_events'][0]
    for field in ('step_index','x_t_sha256','timestep_sha256','preview_sha256','active_edit_cfg'):
        if a[field]!=b[field]:raise ValueError('static/observed pre-update conditions differ: '+field)
    for field in ('text_ids','text_positions','memory_length','visual_prefix_length','conditional_lengths','conditional_rope','text_removed_lengths','image_removed_lengths','visual_posterior_seed'):
        if a['contexts'][field]!=b['contexts'][field]:raise ValueError('static/observed context contract differs: '+field)


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
    for key in ('schema', 'architecture', 'source_sha256', 'model_sha256', 'benchmark_sha256', 'sampling', 'loop', 'seeds', 'arms', 'prompt_ids', 'stage', 'diagnostics', 'memory_topologies', 'arm_configs', 'config', 'config_sha256', 'plan_sha256', 'native_depth', 'time_windows', 'accelerator', 'precision', 'kernel', 'torch'):
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
    if provenance[0]['architecture']=='observation_memory':
        indexed={(r['arm'],*identity(r)):r for r in records}
        for row in records:
            validate_observation_record(row)
            if row['arm'].endswith('_OBSERVED'):
                validate_observation_pair(row,indexed[(row['arm'].replace('_OBSERVED','_STATIC'),*identity(row))])
    return records, provenance[0]
