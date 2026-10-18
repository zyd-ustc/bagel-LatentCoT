"""GEN reading, executable feedback and same-state memory loop training."""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import torch
from . import distributed as parallel
from .data import load_dataset, file_hash
from .record import manifest, write_json
from .visual_record import initialize_student
from .training_logging import configure_logging


def load_input_rows(args, cfg):
    """Validate image contents once on rank 0, then share normalized rows."""
    result = [None]
    if parallel.rank() == 0:
        try:
            rows = load_dataset(args.data, visual=True,
                                allow_unreviewed_proxy=cfg.get('allow_unreviewed_proxy', False))
            if any(not re.fullmatch(r'[A-Za-z0-9_-]+', row['sample_id']) for row in rows):
                raise ValueError('sample_id must be a safe filename identifier')
            text_rows = []
            result[0] = dict(rows=rows, text_rows=text_rows, error=None)
        except Exception as error:
            if parallel.world_size() == 1:
                raise
            result[0] = dict(error=f'{type(error).__name__}: {error}')
    if parallel.world_size() > 1:
        import torch.distributed as dist
        dist.broadcast_object_list(result, src=0)
    if result[0]['error']:
        raise ValueError('rank 0 dataset validation failed: ' + result[0]['error'])
    return result[0]['rows'], result[0]['text_rows']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'train', 'eval', 'sample', 'label-states', 'source-review'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', help='small diagnostic JSONL only')
    parser.add_argument('--corpus', help='annotation task indexes for training')
    parser.add_argument('--config', default='configs/visual_memory_pilot.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='npu')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--topology', choices=['C'], default='C')
    parser.add_argument('--stage', choices=['C2-observe', 'C2-feedback'])
    parser.add_argument('--checkpoint', help='visual parent/checkpoint; C1 initialization uses --writer-init')
    parser.add_argument('--writer-init', help='optional historical C1 writer weights')
    parser.add_argument('--text-checkpoint', help='optional frozen text teacher; validated with E1')
    parser.add_argument('--resume', help='native UND format 4 optimizer/index checkpoint')
    parser.add_argument('--c0-pass')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--steps', type=int, help='override budget, e.g. 20 for resource check')
    parser.add_argument('--validation-max-families', type=int, help='positive bounded validation family budget')
    parser.add_argument('--validation-max-rows', type=int, help='bounded validation row budget')
    parser.add_argument('--skip-validation', action='store_true', help='resource checks only; no selected best checkpoint')
    parser.add_argument('--log-every', type=int, help='console interval; local JSONL and W&B record every step')
    parser.add_argument('--wandb-mode', choices=['disabled','offline','online'], default='disabled')
    parser.add_argument('--wandb-project', default=os.environ.get('WANDB_PROJECT','bagel-LatentCoT'))
    parser.add_argument('--wandb-entity', default=os.environ.get('WANDB_ENTITY'))
    parser.add_argument('--wandb-name', help='defaults to output directory name; each launch creates a new run')
    parser.add_argument('--eval-t', nargs='+', type=float)
    parser.add_argument('--noise-seeds', nargs='+', type=int)
    parser.add_argument('--review-t', nargs='+', type=float, default=[.25, .75, .9],
                        help='source-review states; leaves training evaluation policy unchanged')
    parser.add_argument('--rounds', choices=[1, 2], type=int)
    parser.add_argument('--images', action='store_true', help='save local x0/CFG prediction and blind review table')
    parser.add_argument('--continue-sampling', action='store_true', help='selected actual native node + complete native suffix')
    parser.add_argument('--layer-diagnostics', action='store_true')
    parser.add_argument('--max-samples', type=int)
    args = parser.parse_args(argv)
    if args.command == "train":
        from ..anchored.legacy import refuse_legacy_memory_sft
        refuse_legacy_memory_sft("visual-train")
    cfg = json.loads(Path(args.config).read_text())
    if args.steps is not None: cfg['steps'] = args.steps
    if args.log_every is not None:
        if args.log_every <= 0: parser.error('--log-every must be positive')
        cfg['log_every'] = args.log_every
    if args.command != 'train' and args.wandb_mode != 'disabled':
        parser.error('W&B tracking is configured for train')
    if args.validation_max_families is not None:
        if args.validation_max_families <= 0:
            parser.error('--validation-max-families must be positive')
        cfg['validation_max_families'] = args.validation_max_families
    if args.validation_max_rows is not None:
        if args.validation_max_rows <= 0: parser.error('--validation-max-rows must be positive')
        cfg['validation_max_rows'] = args.validation_max_rows
    if args.eval_t is not None: cfg['eval_t'] = args.eval_t
    if args.noise_seeds is not None: cfg['noise_seeds'] = args.noise_seeds
    stage = args.stage or cfg['stage']
    if args.command == 'train' and stage not in ('C2-observe', 'C2-feedback'):
        parser.error('Generation training uses qwen_latent_cot.interaction.cli sft|rl')
    cfg['stage'] = stage
    if args.command == 'check' and args.distributed:
        parser.error('E0 runs on one NPU before formal 16-NPU training')
    if not args.distributed and int(os.environ.get('WORLD_SIZE', '1')) > 1:
        parser.error('torchrun requires --distributed')
    if args.distributed:
        args.device = str(parallel.initialize(args.device))
    corpus=None; rows=[]
    if args.command=='train':
        if not args.corpus or args.data: parser.error('train requires --corpus; --data is diagnostic only')
        from .corpus import Corpus
        corpus=Corpus(args.corpus)
    else:
        if not args.data: parser.error('diagnostic commands require --data')
        rows,_=load_input_rows(args,cfg)
    if args.command in ('eval', 'sample', 'source-review') and not args.checkpoint:
        parser.error('evaluation requires --checkpoint')
    if args.command == 'source-review' and any(not 0 <= t < 1 for t in args.review_t):
        parser.error('source-review requires actual t in [0,1)')
    output = Path(args.output).resolve()
    if parallel.rank() == 0:
        if output.exists() and any(output.iterdir()):
            raise ValueError('use a fresh output directory')
        output.mkdir(parents=True, exist_ok=True)
    parallel.barrier()
    logger = configure_logging(output)
    logger.info('Loading model: command=%s stage=%s world_size=%d output=%s',
                args.command, stage, parallel.world_size(), output)
    torch.manual_seed(cfg['seed'])
    from ..bagel.backbone import load_native
    from .runtime import DiagnosticRuntime
    from .visual import validate_config
    runtime = DiagnosticRuntime(load_native(args.model, args.device, cfg['timestep_shift']), cfg, 'C')
    validate_config(runtime, stage)
    data_path = str(corpus.root/'manifest.json') if corpus else args.data
    run = manifest(runtime, args, data_path, rows=rows) if parallel.rank() == 0 else None
    if parallel.world_size() > 1:
        import torch.distributed as dist
        shared = [run]; dist.broadcast_object_list(shared, src=0); run = shared[0]
    runtime.native_files = run['model_files']
    runtime.data_hashes = dict(corpus=corpus.fingerprint) if corpus else dict(diagnostic=file_hash(args.data))
    if args.c0_pass:
        c0 = json.loads(Path(args.c0_pass).read_text())
        if c0.get('passed') is not True or c0.get('model_files') != runtime.native_files or c0.get('instruction_token_ids') != runtime.c0_instruction_ids:
            raise ValueError('C0 decoding calibration differs from frozen model/instruction')
    elif args.command == 'train':
        raise ValueError('training requires the existing approved --c0-pass calibration.json')
    student = initialize_student(runtime, args.writer_init, args.checkpoint, teacher_path=args.text_checkpoint)
    if parallel.rank() == 0:
        root = Path(__file__).resolve().parents[2]
        dirty = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root, text=True)
        (output/'source_dirty.diff').write_text(dirty)
        write_json(output/'config.json', cfg)
        write_json(output/'locked_data.json', dict(corpus=corpus.manifest if corpus else None, diagnostic_rows=rows, hashes=runtime.data_hashes))
        run.update(stage=stage, writer_architecture=cfg['writer_architecture'],
                   text_teacher_path=args.text_checkpoint, text_teacher_hash=student.text_teacher_hash,
                   data_hashes=runtime.data_hashes, global_effective_batch=cfg['effective_batch'],
                   trainable={name:p.numel() for name,p in student.native_parameters().items() if p.requires_grad},
                   memory_slots=student.writer.slots, loss_weights=cfg['loss_weights'],
                   source_policy=cfg['visual_source_policy'], cross_round_gradients=cfg['cross_round_gradients'],
                   git_status=subprocess.check_output(['git','status','--short'],cwd=root,text=True),
                   dirty_diff_hash=file_hash(output/'source_dirty.diff'),
                   world_size=parallel.world_size(), command_kind=args.command)
        write_json(output/'manifest.json', run)
    try:
        selected = [row for row in rows if row['split'] == args.split]
        if args.max_samples: selected = selected[:args.max_samples]
        if args.command == 'train':
            from .visual_training import train
            train(student, corpus, stage, output, args.checkpoint or args.writer_init,
                  resume=args.resume, skip_validation=args.skip_validation,
                  wandb_options=dict(mode=args.wandb_mode, project=args.wandb_project,
                                     entity=args.wandb_entity, name=args.wandb_name))
        elif args.command == 'check':
            from .flow_checks import check
            student.writer.requires_grad_(False)
            selected=[dict(r,task='generation',flow_supervision=True) for r in selected if r.get('image') and not r.get('feedback') and not r.get('state')]
            check(student, selected, output)
        elif args.command == 'source-review':
            from .source_review import source_review, summarize_source_review
            tasks = [(row, t, seed) for row in selected for t in args.review_t for seed in cfg['noise_seeds']]
            directory = output/f'rank_{parallel.rank():02d}' if args.distributed else output
            source_review(student, tasks[parallel.rank()::parallel.world_size()], rows, directory)
            parallel.barrier()
            if parallel.rank() == 0:
                summarize_source_review(output, args.checkpoint)
        else:
            # Family allocation keeps all variants of one scene on the same worker.
            families = sorted({row['family_id'] for row in selected})
            assigned = set(families[parallel.rank()::parallel.world_size()])
            local_rows = [row for row in selected if row['family_id'] in assigned]
            directory = output/f'rank_{parallel.rank():02d}' if args.distributed else output
            directory.mkdir(parents=True, exist_ok=True)
            if args.command == 'eval':
                from .visual_evaluation import evaluate, evaluate_feedback
                # Swap partners are available even when the donor family is on another rank.
                if stage == 'C2-observe':
                    evaluate(student, local_rows, directory, donor_rows=rows, rounds=1,
                             save_images=args.images, continue_sampling=args.continue_sampling,
                             layer_diagnostics=args.layer_diagnostics)
                else:
                    evaluate_feedback(student, local_rows, directory,
                                      updates=args.rounds or (2 if stage == 'C3-loop' else 1), donor_rows=selected)
            elif args.command == 'sample':
                from .visual_evaluation import complete_sampling
                for row in local_rows:
                    for seed in cfg['noise_seeds']:
                        for rounds, arm in ((1,'base'), (1,'new'), (2,'new')):
                            complete_sampling(student, row, seed, rounds, arm, output=directory)
            else:
                from .visual_evaluation import collect_label_states
                collect_label_states(runtime, local_rows, directory)
            parallel.barrier()
        run['status'] = 'completed'
    except Exception as error:
        logger.error('Run failed: %s: %s',type(error).__name__,error)
        run.update(status='failed', failure=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        if corpus: corpus.close()
        if parallel.rank() == 0: write_json(output/'manifest.json', run)
        if args.distributed:
            import torch.distributed as dist
            dist.destroy_process_group()


if __name__ == '__main__': main()
