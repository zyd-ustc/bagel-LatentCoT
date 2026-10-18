"""Observe parent -> interaction flow SFT -> fixed-writer Flow-GRPO."""
import argparse
import json
import os
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    for stage in ('sft', 'rl'):
        command = commands.add_parser(stage)
        command.add_argument('--model', required=True)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument('--parent' if stage == 'sft' else '--checkpoint', required=False)
        source.add_argument('--resume')
        command.add_argument('--config', default=f'configs/interaction_{stage}.json')
        command.add_argument('--train', required=True)
        command.add_argument('--val', required=True)
        command.add_argument('--output', required=True)
        command.add_argument('--device', default='npu:0')
        command.add_argument('--distributed', action='store_true')
        command.add_argument('--steps', type=int)
        command.add_argument('--global-batch', type=int)
        command.add_argument('--val-images', type=int)
        if stage == 'rl':
            command.add_argument('--reward-questions', required=True)
            command.add_argument('--judge-queue', required=True)
            command.add_argument('--prompts-per-batch', type=int)
        command.add_argument('--wandb-mode', default=os.getenv('WANDB_MODE', 'disabled'), choices=('online', 'offline', 'disabled'))
        command.add_argument('--wandb-project', default=os.getenv('WANDB_PROJECT', 'bagel-interaction'))
        command.add_argument('--wandb-entity', default=os.getenv('WANDB_ENTITY'))
        command.add_argument('--wandb-name', default=os.getenv('WANDB_NAME'))
    sample = commands.add_parser('sample')
    sample.add_argument('--model', required=True)
    sample.add_argument('--checkpoint', required=True)
    sample.add_argument('--caption', required=True)
    sample.add_argument('--output', required=True)
    sample.add_argument('--device', default='npu:0')
    sample.add_argument('--seed', type=int, default=0)
    sample.add_argument('--arm', default=None)
    audit = commands.add_parser('audit')
    audit.add_argument('--input', required=True)
    audit.add_argument('--output', required=True)
    audit.add_argument('--seed', type=int, default=17)
    audit.add_argument('--max-pairs', type=int)
    audit.add_argument('--val-fraction', type=float, default=.05)
    audit.add_argument('--exclude-hashes', help='One benchmark image SHA256 per line')
    export = commands.add_parser('export')
    export.add_argument('--corpus', required=True)
    export.add_argument('--output', required=True)
    export.add_argument('--count', type=int, default=16000)
    export.add_argument('--seed', type=int, default=17)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command in ("sft", "rl"):
        from ..anchored.legacy import refuse_legacy_memory_sft
        refuse_legacy_memory_sft(f"interaction {args.command}")
    if args.command == 'export':
        from .data import export_candidates
        print(json.dumps(export_candidates(args.corpus, args.output, args.count, args.seed)))
        return
    if args.command == 'audit':
        from .data import audit_pairs
        excluded = Path(args.exclude_hashes).read_text().splitlines() if args.exclude_hashes else []
        print(json.dumps(audit_pairs(args.input, args.output, seed=args.seed, val_fraction=args.val_fraction,
                                    max_pairs=args.max_pairs, excluded_images=excluded), ensure_ascii=False))
        return
    if args.command == 'sample':
        from ..diagnostics.inference import InferenceEngine
        from ..bagel.accelerator import set_device
        engine = InferenceEngine.load(args.model, args.checkpoint, set_device(args.device))
        image, nodes = engine.generate(args.caption, args.seed, arm=args.arm)
        image.save(args.output)
        Path(args.output+'.json').write_text(json.dumps(nodes, indent=2))
        return
    import random
    import torch
    from . import checkpoint
    from .data import load_pairs, validate_splits
    from .student import InteractionStudent
    from .training import Trainer, train_sft
    from ..bagel.accelerator import set_device
    from ..bagel.backbone import load_native
    from ..diagnostics.runtime import DiagnosticRuntime
    from ..diagnostics import distributed as parallel
    from ..diagnostics.data import file_hash
    from ..diagnostics.inference import resolve_checkpoint
    from ..diagnostics.training_logging import start_wandb

    cfg = json.loads(Path(args.config).read_text())
    for key in ('steps', 'global_batch', 'val_images', 'prompts_per_batch'):
        if getattr(args, key, None) is not None:
            cfg[key] = getattr(args, key)
    if cfg['early_steps'] != 20 or cfg['rank'] != 16 or cfg['alpha'] != 16:
        raise ValueError('First interaction protocol requires early20 and rank/alpha16')
    device = parallel.initialize(args.device) if args.distributed else set_device(args.device)
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    path = resolve_checkpoint(args.resume or getattr(args, 'parent', None) or getattr(args, 'checkpoint', None))
    saved = checkpoint.read(path)
    initialized = saved.get('kind') == checkpoint.KIND
    if args.command == 'sft' and not args.resume and initialized:
        raise ValueError('SFT --parent is a format-4 Observe/Feedback parent; continue SFT with --resume')
    if args.resume and (not initialized or saved['stage'] != args.command):
        raise ValueError('Resume requires the same interaction training stage')
    if args.command == 'rl' and not args.resume and (not initialized or saved['stage'] != 'sft'):
        raise ValueError('RL starts from a saved SFT interaction checkpoint')
    parent = saved['parent'] if initialized else saved
    runtime_cfg = dict(parent['config'])
    if (runtime_cfg['sampler_steps'], runtime_cfg['timestep_shift']) != (50, 3):
        raise ValueError('Interaction protocol requires the native 50-node shift=3 grid')
    runtime_cfg['source_layers'] = list(range(parent['depth']))
    runtime = DiagnosticRuntime(load_native(args.model, device, 3), runtime_cfg, 'C')
    runtime.native_files = parent['native_files']
    student = InteractionStudent(runtime, parent, cfg)
    if initialized:
        checkpoint.apply(saved, student)
    if args.command == 'rl' and not args.resume:
        student.begin_rl()
    train, val = load_pairs(args.train), load_pairs(args.val)
    validate_splits(train, val)
    parent_hash = parallel.gather_samples([file_hash(path)] if parallel.rank() == 0 and not args.resume else [])
    provenance = dict(model=str(Path(args.model).resolve()), train=str(Path(args.train).resolve()), train_sha256=file_hash(args.train),
                      val=str(Path(args.val).resolve()), val_sha256=file_hash(args.val),
                      parent_path=str(path) if not args.resume else saved['provenance']['parent_path'],
                      parent_sha256=parent_hash[0] if not args.resume else saved['provenance']['parent_sha256'])
    if args.command == 'rl':
        provenance['reward_questions_sha256'] = file_hash(args.reward_questions)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output/'train.jsonl').exists():
        raise FileExistsError('Use a fresh output directory or --resume')
    wandb = start_wandb(output, cfg, dict(stage='interaction_'+args.command),
                        dict(mode=args.wandb_mode, project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name))
    trainer = Trainer(student, train, val, output, args.command, provenance,
                      saved=saved if args.resume else None, wandb=wandb)
    try:
        if args.command == 'sft':
            train_sft(trainer)
        else:
            from .rl import train_rl
            train_rl(trainer, args.reward_questions, args.judge_queue)
    finally:
        if wandb:
            wandb.finish()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
