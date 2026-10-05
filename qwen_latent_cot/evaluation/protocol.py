"""Hard guards for independent confirmation. Engineering runs stay explicitly labeled."""
import json
import unicodedata
from pathlib import Path
from .io import sha256


def _normalized(prompt):
    return ' '.join(unicodedata.normalize('NFC',prompt).casefold().split())


def validate_confirmation(data,seeds,arms,audit_path,manifest_path):
    if len(data)!=384 or seeds!=[11,23,37]:raise ValueError('confirmation requires 384 prompts and seeds 11,23,37')
    if sum(r.get('bucket')=='structural' for r in data)!=256 or sum(r.get('bucket') in ('ordinary','easy_noop') for r in data)!=128:
        raise ValueError('confirmation bucket counts must be 256 structural + 128 ordinary/easy')
    if len({_normalized(r['prompt']) for r in data})!=384:raise ValueError('confirmation prompts must be unique')
    expected={'BASE','GEN_LAYERWISE','MEMORY_DYNAMIC','MEMORY_STATIC','BASE_MATCHED_LATENCY'}
    if set(arms)!=expected:raise ValueError('confirmation requires five registered arms')
    if not audit_path:raise ValueError('confirmation requires a split audit')
    audit=json.loads(Path(audit_path).read_text())
    if audit.get('manifest_sha256')!=sha256(manifest_path):raise ValueError('split audit refers to a different manifest')
    if audit.get('disjoint_from')!=['historical_evaluation','development','training']:
        raise ValueError('split audit does not cover required exclusions')
    for key in ('historical_evaluation','development','training'):
        source=Path(audit['exclusion_manifests'][key]['path'])
        if sha256(source)!=audit['exclusion_manifests'][key]['sha256']:raise ValueError('exclusion manifest changed')
        prompts={_normalized(json.loads(line)['prompt']) for line in source.read_text().splitlines() if line.strip()}
        if prompts & {_normalized(r['prompt']) for r in data}:raise ValueError('confirmation prompt leakage')
