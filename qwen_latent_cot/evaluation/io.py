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
    if row['arm']=='BASE' or row.get('arm_type')=='legacy_memory_loop':
        if events:raise ValueError('Base has an observation update')
        if row.get('arm_type')=='legacy_memory_loop' and (
            row['arm']!='LEGACY_EARLY_20_R2' or row['loop_step_indexes']!=list(range(20))
            or row['extra_rounds']!=2 or row['start_layer']!=0 or row['end_layer']!=8):
            raise ValueError('legacy Early20 R2 control changed')
        return
    # A failed decode can stop before the selected step. Invalid arms remain in
    # quality/invalid-rate reporting, without inventing a successful update.
    start=row['observation_step'];end=row.get('conditioning_end_step',start+1)
    if row['valid_file'] and [e['step_index'] for e in events]!=list(range(start,end)):
        raise ValueError('observation conditioning coverage differs')
    for index,event in enumerate(events):
        if event['step_index']!=start+index or event['updates']!=int(index==0):
            raise ValueError('observation step/update count differs')
        if not event['x_t_unchanged'] or not event['timestep_unchanged']:
            raise ValueError('observation changed sampler state')
        if index==0:
            for path,digest in (('source_preview','preview_sha256'),('saved_state','state_sha256')):
                if sha256(event[path])!=event[digest]:raise ValueError('observation artifact changed: '+path)
        elif event['contexts']!=events[0]['contexts'] or event.get('probes'):
            raise ValueError('held context changed or probe repeated')
    if events and 'memory_fingerprint' in events[0] and row['valid_file'] and not events[-1].get('held_memory_unchanged'):
        raise ValueError('held Memory immutability was not verified')


def validate_observation_pair(row,other):
    if row['noise_sha256']!=other['noise_sha256']:raise ValueError('paired initial noise differs')
    if row.get('conditioning_end_step')!=other.get('conditioning_end_step'):
        raise ValueError('paired conditioning coverage differs')
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
        anchors={}
        for row in records:
            validate_observation_record(row)
            config=provenance[0]['arm_configs'][row['arm']]
            if 'observation_step' in config:
                if row['observation_step']!=config['observation_step'] or row.get('conditioning_end_step',row['observation_step']+1)!=config.get('conditioning_end_step',config['observation_step']+1):
                    raise ValueError('recorded conditioning window differs from plan')
                if provenance[0]['config'].get('conditioning_durations') and row['observation_events']:
                    key=(*identity(row),row['observe_image']);event=row['observation_events'][0]
                    signature={k:event[k] for k in ('x_t_sha256','timestep_sha256','preview_sha256','contexts','memory_fingerprint')}
                    if key in anchors and signature!=anchors[key]:raise ValueError('duration arms did not start from the same fixed Memory')
                    anchors[key]=signature
            if row['arm'].endswith('_OBSERVED'):
                validate_observation_pair(row,indexed[(row['arm'].replace('_OBSERVED','_STATIC'),*identity(row))])
    return records, provenance[0]
