"""Caption-image inputs and deterministic, resumable global sampling."""
import hashlib
import json
import random
from pathlib import Path


def identity(caption, image):
    return hashlib.sha256(json.dumps([caption, str(image)], ensure_ascii=False).encode()).hexdigest()


def load_pairs(path):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        caption, image = value['caption'].strip(), Path(value['image'])
        if not caption or not image.is_absolute():
            raise ValueError(f'{path}:{number}: require nonempty caption and absolute image')
        rows.append(dict(caption=caption, image=str(image), prompt=caption,
                         sample_id=identity(caption, image), task='generation',
                         flow_supervision=True, label_kind='target_clean',
                         **{k:value[k] for k in ('family_id','source','split') if k in value}))
    if not rows:
        raise ValueError(f'Empty caption-image input: {path}')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate caption-image pairs: {path}')
    return rows


def validate_splits(train, val):
    if {r['family_id'] for r in train if r.get('family_id')} & {r['family_id'] for r in val if r.get('family_id')}:
        raise ValueError('Train and validation families overlap')
    if {r['image'] for r in train} & {r['image'] for r in val}:
        raise ValueError('Train and validation image paths overlap')


class PairSampler:
    def __init__(self, rows, seed):
        self.rows = rows
        self.rng = random.Random(seed)
        self.order = []
        self.cursor = 0

    def take(self, count):
        result = []
        for _ in range(count):
            if self.cursor == len(self.order):
                self.order = list(range(len(self.rows)))
                self.rng.shuffle(self.order)
                self.cursor = 0
            result.append(self.rows[self.order[self.cursor]])
            self.cursor += 1
        return result

    def state_dict(self):
        return dict(order=self.order, cursor=self.cursor, rng=self.rng.getstate())

    def load_state_dict(self, state):
        self.order, self.cursor = state['order'], state['cursor']
        self.rng.setstate(state['rng'])


def audit_pairs(source, output, *, seed=17, val_fraction=.05, max_pairs=None,
                excluded_images=(), min_size=512, max_aspect=1.2):
    """Filter assets, hash-deduplicate and split; semantic approval stays explicit.

    Optional source sidecar `<source>.audit.jsonl` carries family_id/source and
    semantic_approved. Output candidates require semantic and human review.
    """
    from PIL import Image
    rows = load_pairs(source)
    sidecar = Path(str(source)+'.audit.jsonl')
    audits = {r['sample_id']: r for r in map(json.loads, sidecar.read_text().splitlines())} if sidecar.exists() else {}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    excluded = set(excluded_images)
    seen, accepted, rejected, families = set(), [], {}, {}
    rng = random.Random(seed)
    rng.shuffle(rows)
    for row in rows:
        if max_pairs is not None and len(accepted) >= max_pairs:
            break
        try:
            path = Path(row['image'])
            from ..diagnostics.data import file_hash
            digest = file_hash(path)
            with Image.open(path) as image:
                image.load()
                width, height = image.size
            reason = ('benchmark_overlap' if digest in excluded else 'duplicate' if digest in seen else
                      'small' if min(width, height) < min_size else
                      'aspect' if max(width, height)/min(width, height) > max_aspect else None)
        except (OSError, ValueError) as error:
            reason = type(error).__name__
        if reason:
            rejected[reason] = rejected.get(reason, 0)+1
            continue
        seen.add(digest)
        audit = dict(audits.get(row['sample_id'], {}), sample_id=row['sample_id'], image_sha256=digest,
                     width=width, height=height)
        family = audit.get('family_id', digest)
        inherited = audit.get('split')
        if inherited is not None and inherited not in ('train', 'val', 'test'):
            raise ValueError('Unknown inherited split: '+inherited)
        if family in families and inherited and families[family] != inherited:
            raise ValueError('Family has conflicting inherited splits')
        if family not in families:
            families[family] = inherited or ('val' if rng.random() < val_fraction else 'train')
        audit.update(family_id=family, split=families[family])
        accepted.append((row, audit))
    for split in ('train', 'val', 'test'):
        selected = [(r, a) for r, a in accepted if a['split'] == split]
        (output/f'{split}.jsonl').write_text(''.join(json.dumps({k: r[k] for k in ('caption', 'image')}, ensure_ascii=False)+'\n' for r, _ in selected))
        (output/f'{split}.audit.jsonl').write_text(''.join(json.dumps(a, ensure_ascii=False)+'\n' for _, a in selected))
    report = dict(candidates=len(rows), accepted=len(accepted), rejected=rejected, seed=seed,
                  benchmark_exclusion_hashes=len(excluded), semantic_approved=sum(bool(a.get('semantic_approved')) for _, a in accepted),
                  review='Asset filtering and exact hashes only; semantic/human review and near-duplicate exclusion are separate.')
    (output/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def export_candidates(corpus_root, output, count, seed):
    """Bounded index sampling, preserving original family splits and provenance."""
    from ..diagnostics.corpus import REF, resolve_path
    root = Path(corpus_root)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['format'] not in ('annotation_index_v1', 'annotation_index_v2'):
        raise ValueError('Unsupported caption-image export index')
    rng = random.Random(seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen, rows, audit = set(), [], []
    views = [v for v in manifest['views'] if v['task'] == 'generation' and v['count']]
    for index in range(count*20):
        if len(rows) == count or not views:
            break
        view = views[index % len(views)]
        with (root/'views'/view['path']).open('rb') as stream:
            stream.seek(rng.randrange(view['count'])*REF.size)
            shard, offset, role = REF.unpack(stream.read(REF.size))
        if role != 2:
            continue
        with (root/'records'/manifest['shards'][shard]['path']).open() as stream:
            stream.seek(offset)
            row = json.loads(stream.readline())
        caption = row['target_caption'].strip()
        image = resolve_path(row['target_image'], row['original']['path'], manifest['registry'].get('roots', {}))
        if not caption or not image:
            continue
        sample_id = identity(caption, image)
        if sample_id in seen:
            continue
        seen.add(sample_id)
        rows.append(dict(caption=caption, image=image))
        audit.append(dict(sample_id=sample_id, source_sample_id=row['record_id'], source=row['source'],
                          family_id=row['family_id'], split=row['split'], semantic_approved=False))
    output.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    Path(str(output)+'.audit.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in audit))
    return dict(exported=len(rows), seed=seed, output=str(output))


class GenerationCorpus:
    """Read only paired generation views in the existing v1/v2 corpus.

    Reuse the binary indices and record LRU; do not rebuild or load 37M rows.
    Original family splits and target-image meaning are preserved.
    """
    def __init__(self, root):
        from collections import OrderedDict
        from ..diagnostics.corpus import TaskView
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if self.manifest['format'] not in ('annotation_index_v1', 'annotation_index_v2'):
            raise ValueError('Stage A requires the paired annotation index v1/v2')
        self.handles = OrderedDict()
        self.views = [TaskView(self, info) for info in self.manifest['views']
                      if info['task'] == 'generation' and info['count']]
        self.fingerprint = self.manifest['fingerprint']

    def record(self, shard, offset):
        from ..diagnostics.corpus import Corpus
        return Corpus.record(self, shard, offset)

    def groups(self, split, tasks=None):
        return [v for v in self.views if v.info['split'] == split
                and (tasks is None or 'generation' in tasks)]

    def materialize(self, value, role):
        from ..diagnostics.corpus import resolve_path
        if role != 2: raise ValueError('Stage A accepts paired generation records only')
        caption = value['target_caption'].strip()
        image = resolve_path(value['target_image'], value['original']['path'], self.manifest['registry'].get('roots', {}))
        if not caption or not image: raise ValueError('Generation record has no caption/target image')
        return dict(caption=caption, prompt=caption, image=image, task='generation',
                    flow_supervision=True, label_kind='target_clean',
                    sample_id=value['record_id'], family_id=value['family_id'],
                    source=value['source'], split=value['split'], annotation=value['annotation'])

    def close(self):
        from ..diagnostics.corpus import Corpus
        Corpus.close(self)


class CorpusPairs:
    """Lightweight train inventory with the established source-coverage sampler."""
    def __init__(self, corpus, config):
        self.corpus, self.config = corpus, config

    def __len__(self):
        return sum(len(v) for v in self.corpus.groups('train'))

    def sampler(self):
        from ..diagnostics.curriculum import CoverageSampler
        # Generation-only selection. Reuse its affine permutations and resume state.
        cfg = dict(self.config, task_mix={'generation': 1.}, loss_weights=dict(sem=0.,teacher=0.,flow=1.))
        base = CoverageSampler(self.corpus, cfg, 'C2-feedback')
        class Sampler:
            @property
            def rng(self): return base.rng
            def take(self, count): return [base.next(1)[0] for _ in range(count)]
            def state_dict(self): return base.state_dict()
            def load_state_dict(self, state): base.load_state_dict(state)
        return Sampler()


def corpus_validation(corpus, count):
    """Deterministic, family-distinct validation, never draw from train/test."""
    rows, families = [], set()
    views = corpus.groups('val')
    for position in range(count):
        for view in views:
            if position >= len(view): continue
            row = view[position]
            if row['family_id'] in families: continue
            rows.append(row); families.add(row['family_id'])
            if len(rows) >= count: return rows
    return rows
