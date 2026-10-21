"""BAGEL full-depth dual-state loop: supervised training and sampling."""
import argparse
import json
import os
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    for stage in ('sft', 'stage_a'):
        command = commands.add_parser(stage)
        command.add_argument('--model', required=True)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument('--parent')
        source.add_argument('--resume')
        command.add_argument('--config', default=f'configs/interaction_{stage}.json')
        command.add_argument('--train', required=stage == 'sft')
        command.add_argument('--val', required=stage == 'sft')
        if stage == 'stage_a':
            command.add_argument('--corpus', help='Existing v1/v2 paired generation corpus; keeps inherited splits')
        command.add_argument('--output', required=True)
        command.add_argument('--device', default='npu:0')
        command.add_argument('--distributed', action='store_true')
        command.add_argument('--steps', type=int)
        command.add_argument('--global-batch', type=int)
        command.add_argument('--val-images', type=int)
        command.add_argument('--wandb-mode', default=os.getenv('WANDB_MODE', 'disabled'), choices=('online', 'offline', 'disabled'))
        command.add_argument('--wandb-project', default=os.getenv('WANDB_PROJECT', 'bagel-interaction'))
        command.add_argument('--wandb-entity', default=os.getenv('WANDB_ENTITY'))
        command.add_argument('--wandb-name', default=os.getenv('WANDB_NAME'))
    sample = commands.add_parser('sample')
    sample.add_argument('--model', required=True)
    source = sample.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', help='Trained full-loop v3 checkpoint')
    source.add_argument('--parent', help='C2 format-4 checkpoint: run the new loop without additional training')
    sample.add_argument('--config', default='configs/interaction_sft.json', help='Initialization config for --parent')
    sample.add_argument('--caption', required=True)
    sample.add_argument('--output', required=True)
    sample.add_argument('--device', default='npu:0')
    sample.add_argument('--seed', type=int, default=0)
    sample.add_argument('--native', action='store_true', help='Disable loop and adapters for a baseline')
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
        device = set_device(args.device)
        if args.parent:
            config = json.loads(Path(args.config).read_text())
            engine = InferenceEngine.from_parent(args.model, args.parent, config, device)
        else:
            engine = InferenceEngine.load(args.model, args.checkpoint, device)
        image, nodes = engine.generate(args.caption, args.seed, native=args.native)
        image.save(args.output)
        Path(args.output+'.json').write_text(json.dumps(nodes, indent=2))
        return
    import random
    import torch
    from . import checkpoint
    from .data import load_pairs, validate_splits, GenerationCorpus, CorpusPairs, corpus_validation
    from .student import InteractionStudent
    from .training import Trainer, train_sft, train_stage_a
    from ..bagel.accelerator import set_device
    from ..bagel.backbone import load_native
    from ..diagnostics.runtime import DiagnosticRuntime
    from ..diagnostics import distributed as parallel
    from ..diagnostics.data import file_hash
    from ..diagnostics.inference import resolve_checkpoint
    from ..diagnostics.training_logging import start_wandb

    cfg = json.loads(Path(args.config).read_text())
    for key in ('steps', 'global_batch', 'val_images'):
        if getattr(args, key, None) is not None:
            cfg[key] = getattr(args, key)
    if bool(cfg.get('stage_a')) != (args.command == 'stage_a'):
        raise ValueError('Stage A command requires a Stage A config; SFT keeps its original architecture')
    if args.command == 'stage_a' and (bool(args.corpus) == bool(args.train or args.val)):
        raise ValueError('Use --corpus OR both --train and --val')
    if not getattr(args, 'corpus', None) and not (args.train and args.val):
        raise ValueError('Paired training requires both --train and --val')
    device = parallel.initialize(args.device) if args.distributed else set_device(args.device)
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    path = resolve_checkpoint(args.resume or args.parent)
    saved = checkpoint.read(path)
    initialized = saved.get('kind') in checkpoint.KINDS
    if not args.resume and initialized:
        raise ValueError('--parent requires a format-4 Observe/Feedback parent; continue this training stage with --resume')
    if args.resume and (not initialized or saved['stage'] != args.command):
        raise ValueError('Resume requires the same interaction training stage')
    parent = saved['parent'] if initialized else saved
    runtime_cfg = dict(parent['config'])
    runtime_cfg['source_layers'] = list(range(parent['depth']))
    runtime = DiagnosticRuntime(load_native(args.model, device, runtime_cfg['timestep_shift']), runtime_cfg, 'C')
    runtime.native_files = parent['native_files']
    student = InteractionStudent(runtime, parent, cfg)
    if initialized:
        checkpoint.apply(saved, student)
    corpus = None
    if getattr(args, 'corpus', None):
        corpus = GenerationCorpus(args.corpus)
        train = CorpusPairs(corpus, dict(runtime_cfg, **cfg))
        val = corpus_validation(corpus, max(16, cfg['val_images']))
        if not len(train) or not val: raise ValueError('Corpus requires nonempty train/val generation views')
    else:
        train, val = load_pairs(args.train), load_pairs(args.val)
        validate_splits(train, val)
    parent_hash = parallel.gather_samples([file_hash(path)] if parallel.rank() == 0 and not args.resume else [])
    provenance = dict(model=str(Path(args.model).resolve()),
                      parent_path=str(path) if not args.resume else saved['provenance']['parent_path'],
                      parent_sha256=parent_hash[0] if not args.resume else saved['provenance']['parent_sha256'])
    if corpus:
        provenance.update(corpus=str(Path(args.corpus).resolve()), corpus_fingerprint=corpus.fingerprint)
    else:
        provenance.update(train=str(Path(args.train).resolve()), train_sha256=file_hash(args.train),
                          val=str(Path(args.val).resolve()), val_sha256=file_hash(args.val))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output/'train.jsonl').exists():
        raise FileExistsError('Use a fresh output directory or --resume')
    wandb = start_wandb(output, cfg, dict(stage='interaction_'+args.command),
                        dict(mode=args.wandb_mode, project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name))
    trainer = Trainer(student, train, val, output, args.command, provenance,
                      saved=saved if args.resume else None, wandb=wandb)
    try:
        (train_stage_a if args.command == 'stage_a' else train_sft)(trainer)
    finally:
        if corpus: corpus.close()
        if wandb:
            wandb.finish()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
