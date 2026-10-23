"""CLI for the training-free explicit closed-loop upper bound."""
import argparse
import json
import math
from pathlib import Path

from .atoms import score_atoms, image_pass, correction_text
from .gallery import write_gallery
from .manifests import normalize
from .metrics import evaluate, summary_lines
from . import protocol


def load_manifest(path, limit=None, ids=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    records = []
    for index, row in enumerate(rows):
        record = normalize(row, index)
        if record is None:
            raise ValueError(f'manifest row {index}: expected vqa_list or GenEval include metadata')
        records.append(record)
    if ids:
        wanted = set(ids)
        records = [r for r in records if r['prompt_id'] in wanted]
    if limit:
        records = records[:int(limit)]
    return records


def times_py(steps, shift):
    nodes = [1 - i / (steps - 1) for i in range(steps)]
    return [shift * u / (1 + (shift - 1) * u) for u in nodes]


class FakeSampler:
    """Pillow-only stand-in for --dry-run."""

    def __init__(self, config):
        self.config = config
        steps = int(config.get('sampler_steps', config.get('num_steps', 50)))
        self.grid = times_py(steps, float(config['timestep_shift']))

    def _image(self, tag, step):
        from PIL import Image, ImageDraw
        color = (60 + (hash(tag) % 150), 90, 140 + (hash(str(step)) % 100))
        image = Image.new('RGB', (256, 256), color)
        draw = ImageDraw.Draw(image)
        draw.text((8, 8), f'{tag}\nstep {step}', fill=(255, 255, 255))
        image.info['tag'] = f'{tag}:{step}'
        return image

    def run(self, prompt, seed, save_indexes=(), save_velocity=()):
        states = {i: self._image(prompt, i) for i in save_indexes}
        return dict(latent=self._image(prompt, 'final'), flow=None, states=states,
                    velocities={i: None for i in save_velocity})

    def resume(self, latent, flow, start_index):
        return self._image('resume', start_index)

    def renoise(self, latent, velocity, index, seed):
        return self._image(f'renoise{seed}', index)

    def decode(self, latent):
        return latent

    def flow(self, prompt):
        return None

    def t_at(self, index):
        return self.grid[index]


class FakeReader:
    """Answers the manifest question correctly with probability `accuracy`."""

    def __init__(self, accuracy=0.65):
        self.accuracy = accuracy

    def score(self, image, vqa):
        import hashlib
        answers = []
        tag = image.info.get('tag', '')
        for question, expected in vqa:
            value = int(hashlib.sha256((question + tag).encode()).hexdigest(), 16)
            if value % 100 < self.accuracy * 100:
                answers.append(expected)
            else:
                answers.append('no' if str(expected).strip().lower() != 'no' else 'yes')
        return answers


def build(records, config, output, device, model, dry_run=False, accuracy=0.65,
          backend='bagel', source=None):
    if dry_run:
        sampler = FakeSampler(config)
        reader = FakeReader(accuracy)
    elif backend == 'sensenova':
        from ..u15.runtime import Runtime as U15Runtime
        from .sensenova import SenseNovaSampler, SenseNovaReader
        if not source:
            raise SystemExit('--source is required for the sensenova backend')
        runtime = U15Runtime.load(model, source, config, device)
        sampler = SenseNovaSampler(runtime)
        reader = SenseNovaReader(runtime, template=config.get('vqa_template', '{question}'),
                                 max_answer_tokens=config.get('max_answer_tokens', 8))
        print(json.dumps(dict(event='loaded', backend='sensenova', device=str(runtime.device),
                              steps=sampler.num_steps,
                              t_rollback=sampler.t_at(int(config['rollback_index'])),
                              t_read=[sampler.t_at(int(i)) for i in config['read_indexes']])), flush=True)
    else:
        from ..bagel.backbone import load_native
        from ..correction.runtime import Runtime
        from .native import NativeSampler
        from .reader import UNDReader
        bundle = load_native(model, device)
        runtime = Runtime(bundle, config)
        sampler = NativeSampler(runtime)
        reader = UNDReader(bundle, template=config.get('vqa_template', '{question}\nAnswer:'),
                           max_answer_tokens=config.get('max_answer_tokens', 8))
        print(json.dumps(dict(event='loaded', device=str(runtime.device),
                              steps=len(sampler.grid) - 1,
                              t_rollback=sampler.t_at(int(config['rollback_index'])),
                              t_read=[sampler.t_at(int(i)) for i in config['read_indexes']])), flush=True)

    results = protocol.run(records, sampler, reader, config, output, log=print)
    report = evaluate(results, config['read_indexes'])
    (Path(output) / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    write_gallery(results, output)
    for line in summary_lines(report):
        print(line, flush=True)
    return results, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT')
    parser.add_argument('--manifest', default='data/geneval2_800.jsonl')
    parser.add_argument('--config', default='configs/upper_bound.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--backend', choices=['bagel', 'sensenova'], default='bagel')
    parser.add_argument('--source', help='pinned SenseNova-U1 source root')
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--read-index', nargs='*', type=int)
    parser.add_argument('--rollback-index', type=int)
    parser.add_argument('--full-resample', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--fake-accuracy', type=float, default=0.65)
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    if args.seed is not None:
        config['seed'] = args.seed
    if args.read_index:
        config['read_indexes'] = args.read_index
    if args.rollback_index is not None:
        config['rollback_index'] = args.rollback_index
    if args.full_resample:
        config['full_resample'] = True

    records = load_manifest(args.manifest, args.limit, args.ids)
    if args.shards > 1:
        if not 0 <= args.shard < args.shards:
            raise SystemExit('shard must be in [0, shards)')
        records = records[args.shard::args.shards]
    if not records:
        raise SystemExit('no prompts selected')
    return build(records, config, args.output, args.device, args.model,
                 dry_run=args.dry_run, accuracy=args.fake_accuracy,
                 backend=args.backend, source=args.source)


if __name__ == '__main__':
    main()
