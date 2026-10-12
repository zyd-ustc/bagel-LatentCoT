"""Frozen split, explicit fact lexicon, clustered uncertainty, reproducible hashes."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import torch

BUCKETS = (0.05, 0.25, 0.50, 0.75, 0.90, 1.00)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


LABEL_KINDS = ('target_clean', 'observation_proxy', 'target_observation_gap')


def visual_label(row):
    """Normalize new labels without upgrading unreviewed proxies to confirmations."""
    kind = row.get('label_kind', 'target_clean')
    if kind not in LABEL_KINDS:
        raise ValueError('unknown visual label_kind')
    row = dict(row, label_kind=kind)
    row.setdefault('family_id', row.get('scene_id', row.get('image_id')))
    row.setdefault('scene_id', row.get('family_id'))
    row.setdefault('image_id', row.get('scene_id'))
    row.setdefault('prompt', 'An image.')
    row.setdefault('label_type', 'observed' if row.get('state') else 'clean')
    if kind == 'target_clean':
        row.setdefault('target_caption', row.get('caption'))
        row.setdefault('target_facts', row.get('facts', []))
        row['caption'], row['facts'] = row['target_caption'], row['target_facts']
        row['confirmed_label'] = row.get('verified') is True
    else:
        required = ('t', 'noise_seed', 'round', 'observation_caption', 'observation_facts',
                    'observation_method', 'review_status', 'observation_state_hash')
        if any(key not in row for key in required):
            raise ValueError('observation labels require actual state, method and review metadata')
        if not 0 <= row['t'] <= 1 or not isinstance(row['noise_seed'], int) or row['round'] != 0:
            raise ValueError('first visual label interface requires S0 labels with actual t/seed')
        if not re.fullmatch('[0-9a-f]{64}', row['observation_state_hash']):
            raise ValueError('observation_state_hash must be SHA256 of labelled x_t')
        if not row.get('observation_image_hash') and not row.get('state'):
            raise ValueError('observation proxy requires source image hash')
        if row.get('state') and not row.get('trajectory_checkpoint_hash'):
            raise ValueError('trajectory label must identify generating checkpoint')
        row['confirmed_label'] = row.get('verified') is True and row['review_status'] == 'approved'
        if kind == 'target_observation_gap':
            if not row.get('target_caption') or not row.get('gap_caption') or 'gap_facts' not in row:
                raise ValueError('gap label requires explicit reviewed gap and target references')
            row['caption'], row['facts'] = row['gap_caption'], row['gap_facts']
        else:
            row['caption'], row['facts'] = row['observation_caption'], row['observation_facts']
        row.setdefault('uncertain_facts', [])
        row['facts'] = row['facts'] + [dict(f, unknown=True) for f in row['uncertain_facts']]
    if not isinstance(row.get('caption'), str) or not row['caption'].strip() or not row.get('family_id'):
        raise ValueError('visual labels require nonempty caption and family_id')
    return row


def load_dataset(path, *, visual=False, allow_unreviewed_proxy=False):
    root = Path(path).resolve().parent
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if visual:
        rows = [visual_label(row) for row in rows]
    ids = {}; families = {}; pairs = {}; image_splits = {}; image_hashes = {}
    for row in rows:
        if visual and not (row.get('image') or row.get('state')):
            raise ValueError('visual data requires a paired image or provenance-locked trajectory state')
        required = ('sample_id', 'scene_id', 'split', 'caption', 'label_type', 'verified', 'facts')
        if any(key not in row for key in required) or row['split'] not in ('train', 'val', 'test'):
            raise ValueError('dataset requires IDs, frozen split, caption, labels, verification and facts')
        proxy_allowed = visual and allow_unreviewed_proxy and row['label_kind'] == 'observation_proxy'
        if row['sample_id'] in ids or row['label_type'] not in ('clean', 'observed') or (row['verified'] is not True and not proxy_allowed):
            raise ValueError('duplicate sample or unverified/unknown label type')
        ids[row['sample_id']] = row
        family = row.get('family_id', row['scene_id'])
        if families.setdefault(family, row['split']) != row['split']:
            raise ValueError('scene family leaks across splits')
        if row.get('pair_id') and pairs.setdefault(row['pair_id'], row['split']) != row['split']:
            raise ValueError('counterfactual pair leaks across splits')
        for fact in row['facts']:
            if fact.get('category') not in ('object', 'attribute', 'count', 'relation'):
                raise ValueError('fact category must be fixed before evaluation')
            if (not isinstance(fact.get('aliases'), list) or not fact['aliases']
                or not all(isinstance(a, str) and a.strip() for a in fact['aliases'])
                or not isinstance(fact.get('alternatives', []), list)
                or not all(isinstance(a, str) and a.strip() for a in fact.get('alternatives', []))):
                raise ValueError('facts require predetermined aliases and alternative assertions')
        for key in ('image', 'state', 'observation_preview') if visual else ('image', 'state'):
            if row.get(key):
                row[key] = str((root / row[key]).resolve())
                if not Path(row[key]).is_file():
                    raise ValueError(f'missing dataset {key}: {row[key]}')
        if visual and row.get('observation_preview_hash') and row.get('observation_preview') and file_hash(row['observation_preview']) != row['observation_preview_hash']:
            raise ValueError('observation preview hash differs from reviewed image')
        if row.get('image'):
            path = row['image']
            if path not in image_hashes:
                image_hashes[path] = file_hash(path)
            digest = image_hashes[path]
            if image_splits.setdefault(digest, row['split']) != row['split']:
                raise ValueError('same original image content leaks across splits')
        if row.get('state') and row['label_type'] != 'observed':
            raise ValueError('trajectory diagnostic rows require verified observed labels')
    for row in rows:
        swap = row.get('swap_id')
        if swap:
            other = ids.get(swap)
            if other is None or other['split'] != row['split'] or other['sample_id'] == row['sample_id']:
                raise ValueError('swap must name a distinct sample in the same split')
            if row.get('prompt') != other.get('prompt'):
                raise ValueError('GEN counterfactuals must preserve prompt')
    return rows


def controlled_noise(x0, t, seed):
    if not 0 <= t <= 1:
        raise ValueError('t must be the ACTUAL flow interpolation value')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    eps = torch.randn(x0.shape, generator=generator, dtype=torch.float32).to(x0.device, x0.dtype)
    # Branch explicitly: pure noise must be independent of x0, including NaNs.
    return eps if t == 1 else (1-t)*x0+t*eps


def score_facts(text, facts):
    """Locked literal assertion rule, not a semantic judge or image-quality score.

    Dataset aliases encode accepted assertions; alternatives encode false ones.
    Negated clauses are not credited. Keep free captions for human audit.
    """
    clauses = re.split(r'[.!?;,]|\b(?:but|however|whereas)\b', text.lower())
    def contains(phrase):
        pattern = r'(?<!\w)' + re.escape(phrase.lower()) + r'(?!\w)'
        return any(re.search(pattern, clause) and not re.search(r'\b(no|not|without|unknown|unclear)\b', clause)
                   for clause in clauses)
    items = []
    for fact in facts:
        correct = any(contains(a) for a in fact['aliases'])
        incorrect = any(contains(a) for a in fact.get('alternatives', []))
        unknown = fact.get('unknown', False)
        items.append(dict(category=fact['category'], unknown=unknown,
                          correct=bool(correct and not incorrect and not unknown),
                          incorrect=bool(incorrect or (correct and unknown)), asserted=bool(correct or incorrect)))
    known = [f for f in items if not f['unknown']]
    unknown = [f for f in items if f['unknown']]
    return dict(fact_accuracy=sum(f['correct'] for f in known)/len(known) if known else None,
                unknown_assertion_rate=sum(f['asserted'] for f in unknown)/len(unknown) if unknown else None,
                facts=items, scoring='locked_literal_assertion_v2')


def cluster_interval(rows, key, seed=0, repeats=1000):
    groups = {}
    for row in rows:
        value = row.get(key)
        if value is not None:
            groups.setdefault(row.get('family_id', row['scene_id']), []).append(value)
    if not groups:
        return None
    values = np.array([np.mean(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return dict(mean=float(values.mean()), low=float(np.quantile(samples, .025)),
                high=float(np.quantile(samples, .975)), clusters=len(values))


def velocity_stats(value, reference):
    a, b = value.float().flatten(), reference.float().flatten()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError('non-finite velocity')
    return dict(normalized_difference=float((a-b).norm()/b.norm().clamp_min(1e-12)),
                cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
                norm_ratio=float(a.norm()/b.norm().clamp_min(1e-12)))
