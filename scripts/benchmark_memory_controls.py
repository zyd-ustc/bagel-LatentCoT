"""Run the same deployed benchmarks through all five inference controls.

Each arm is a normal memory_benchmark run, so official scorers and exports are
shared with existing single-arm runs. No prompt subsampling is introduced.
"""
from argparse import Namespace
import json
from pathlib import Path
import memory_benchmark as benchmark
from qwen_latent_cot.diagnostics.inference import ARMS, ALL_ARMS, INTERACTION_ARMS, resolve_checkpoint


DEFAULT_CHECKPOINT = '/root/outputs/feedback_nativeund_attn_full_20k'


def run(args):
    root = Path(args.run).resolve()
    if args.command == 'prepare':
        checkpoint = str(resolve_checkpoint(args.checkpoint))
        identity = dict(format='memory_controls_v1', arms=args.arms, checkpoint=checkpoint,
                        checkpoint_sha256=benchmark.digest(checkpoint), seeds=args.seeds,
                        sampling_config=benchmark.SAMPLING, early=args.early,
                        loop_layers=args.loop_layers, donor_seed_offset=args.donor_seed_offset)
        path = root/'controls.json'
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError('Existing controls differ; choose a new output directory')
        for arm in args.arms:
            options = Namespace(**vars(args))
            options.run, options.arm, options.checkpoint = str(root/arm), arm, checkpoint
            options.checkpoint_sha256 = identity['checkpoint_sha256']
            benchmark.prepare(options)
        benchmark.write_json(path, identity)
        return

    identity = json.loads((root/'controls.json').read_text())
    arms = args.arms if args.arms is not None else identity['arms']
    if not set(arms) <= set(identity['arms']):
        raise ValueError('Requested arms were not prepared')
    engine = None
    summaries = {}
    for arm in arms:
        options = Namespace(**vars(args))
        options.run = str(root/arm)
        if args.scores:
            options.scores = str(Path(args.scores)/arm)
        if args.command == 'generate':
            engine = benchmark.generate(options, engine)
        else:
            getattr(benchmark, args.command)(options)
        if args.command == 'summary':
            summaries[arm] = json.loads((Path(options.scores or options.run)/'summary.json').read_text())
    if summaries:
        benchmark.write_json(Path(args.scores or root)/'summary.json', summaries)


def main():
    parser = benchmark.build_parser()
    parser.description = __doc__
    parser.set_defaults(checkpoint=DEFAULT_CHECKPOINT)
    parser.add_argument('--arms', nargs='+', choices=ALL_ARMS,
                        help='default: controls for the selected checkpoint; prepared arms for other commands')
    args = parser.parse_args()
    if args.command == 'prepare' and args.arms is None:
        from qwen_latent_cot.interaction.checkpoint import read, KIND
        args.arms = list(INTERACTION_ARMS if read(args.checkpoint).get('kind') == KIND else ARMS)
    if len(args.seeds) != len(set(args.seeds)) or not 0 <= args.rank < args.world:
        parser.error('Seeds must be unique and rank must be in [0, world)')
    if args.command == 'score' and not args.scorer:
        parser.error('--scorer is required for score')
    if args.early <= 0 or args.loop_layers <= 0 or args.donor_seed_offset % (2**63-1) == 0:
        parser.error('early/layers must be positive and donor offset must produce a distinct seed')
    if args.arms and len(set(args.arms)) != len(args.arms):
        parser.error('Arms must be unique')
    run(args)


if __name__ == '__main__':
    main()
