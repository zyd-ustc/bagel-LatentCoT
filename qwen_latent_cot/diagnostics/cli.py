"""The single supported diagnostic entry point."""
import argparse
import json
import re
import os
from pathlib import Path
import torch
from .data import load_dataset
from .record import manifest, write_json, load_checkpoint


def main():
    import sys
    routes = {'visual-check': 'check', 'visual-train': 'train', 'visual-eval': 'eval',
              'loop-sample': 'sample', 'visual-label-states': 'label-states'}
    if len(sys.argv) > 1 and sys.argv[1] in routes:
        from .visual_cli import main as visual_main
        return visual_main([routes[sys.argv[1]], *sys.argv[2:]])
    parser = argparse.ArgumentParser(description='Frozen BAGEL independent Memory protocol (2026-10-11)')
    parser.add_argument('command', choices=['calibrate', 'train', 'caption', 'gen', 'trajectory', 'report', 'distill', *routes])
    parser.add_argument('--model')
    parser.add_argument('--query-data', help='independent verified train/val images for shared GEN queries')
    parser.add_argument('--data')
    parser.add_argument('--output', required=True)
    parser.add_argument('--runs', nargs='+', help='run directories to summarize for report')
    parser.add_argument('--config', default='configs/memory_topology.json')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--topology', choices=['L', 'C'], default='L')
    parser.add_argument('--stage', choices=['C1', 'C2'], default='C1')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--checkpoint')
    parser.add_argument('--text-checkpoint')
    parser.add_argument('--visual-checkpoint')
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--continue-sampling', action='store_true')
    parser.add_argument('--c0-pass', help='calibration.json with manually verified C0 freedom-to-decode pass')
    parser.add_argument('--distributed', action='store_true', help='synchronize writer gradients across torchrun workers; effective_batch is global')
    args = parser.parse_args()
    if args.distributed and args.command not in ('train', 'distill'):
        parser.error('--distributed is supported only for train/distill')
    if not args.distributed and int(os.environ.get('WORLD_SIZE', '1')) > 1:
        parser.error('torchrun training requires --distributed')
    if args.command == 'report':
        if not args.runs:
            parser.error('report requires --runs')
        from .report import summarize
        summarize(args.runs, Path(args.output).resolve())
        return
    if not args.model or not args.data:
        parser.error('this command requires --model and --data')
    config = json.loads(Path(args.config).read_text())
    if config['slots'] != 16 or config['resolution'] != 512:
        raise ValueError('first-round protocol fixes 16 slots and 512 resolution; use a separately named follow-up')
    if config['train_t'] != [0.05, 0.25, 0.5, 0.75, 0.9] or 1.0 not in config['eval_t']:
        raise ValueError('first-round training excludes pure noise; evaluation must include it')
    if config['steps'] <= 0 or config['effective_batch'] <= 0 or config['eval_every'] <= 0:
        raise ValueError('training budgets must be positive')
    rows = load_dataset(args.data)
    query_rows = None
    if args.command == 'distill':
        if not args.query_data or not args.checkpoint or args.topology != 'C':
            parser.error('distill requires --query-data, --checkpoint and --topology C')
        query_rows = load_dataset(args.query_data)
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', r['sample_id']) for r in rows):
        raise ValueError('sample_id must be a safe filename identifier')
    if args.command in ('train', 'gen', 'distill'):
        if not args.c0_pass:
            raise ValueError('first validate C0 free captions and provide --c0-pass calibration.json')
        c0 = json.loads(Path(args.c0_pass).read_text())
        if c0.get('passed') is not True:
            raise ValueError('C0 decoding has not passed; repair protocol before training/GEN interpretation')
    from . import distributed as parallel
    if args.distributed:
        args.device = str(parallel.initialize(args.device))
        if config['effective_batch'] % parallel.world_size():
            raise ValueError('global effective_batch must be divisible by WORLD_SIZE')
    output = Path(args.output).resolve()
    if parallel.rank() == 0 and output.exists() and any(output.iterdir()):
        raise ValueError('output must be empty; never mix runs or overwrite old records')
    if parallel.rank() == 0:
        output.mkdir(parents=True, exist_ok=True)
    parallel.barrier()
    torch.manual_seed(config['seed'])
    from ..bagel.backbone import load_native
    from .runtime import DiagnosticRuntime
    runtime = DiagnosticRuntime(load_native(args.model, args.device, config['timestep_shift']), config, args.topology)
    run_manifest = manifest(runtime, args, args.data) if parallel.rank() == 0 else None
    if parallel.world_size() > 1:
        import torch.distributed as dist
        shared = [run_manifest]
        dist.broadcast_object_list(shared, src=0)
        run_manifest = shared[0]
    if query_rows is not None:
        from .data import file_hash
        run_manifest['query_dataset_hash'] = file_hash(args.query_data)
        run_manifest['query_dataset_path'] = str(Path(args.query_data).resolve())
        run_manifest['query_image_hashes'] = {r['image']: file_hash(r['image']) for r in query_rows if r.get('image')}
    run_manifest['distributed'] = dict(world_size=parallel.world_size(), global_effective_batch=config['effective_batch'],
                                       gradient_reduction='SUM_after_global_batch_loss_scaling', backend=dist.get_backend() if args.distributed else None)
    if parallel.rank() == 0:
        write_json(output/'manifest.json', run_manifest)
    runtime.native_files = run_manifest['model_files']
    try:
        if args.command in ('train', 'gen', 'distill'):
            if c0.get('model_files') != run_manifest['model_files'] or c0.get('instruction') != config['c0_instruction'] or c0.get('instruction_token_ids') != runtime.c0_instruction_ids:
                raise ValueError('C0 calibration does not match native model/tokenizer or decode instruction')
        if args.command == 'caption' and args.checkpoint:
            load_checkpoint(args.checkpoint, runtime)  # Explicit init controls are allowed.
        if args.command == 'distill':
            from .response import train
            train(runtime, rows, query_rows, output, args.checkpoint)
        elif args.command == 'train':
            from .caption import train
            train(runtime, rows, args.stage, output, args.checkpoint, args.scratch)
        else:
            selected = [r for r in rows if r['split'] == args.split]
            if not selected:
                raise ValueError('requested evaluation split is empty')
            if args.command == 'trajectory':
                from .trajectory import collect
                collect(runtime, selected, output)
            elif args.command in ('calibrate', 'caption'):
                from .caption import evaluate
                evaluate(runtime, selected, 'C0' if args.command == 'calibrate' else args.stage, output)
                if args.command == 'calibrate':
                    write_json(output/'calibration.json', dict(passed=False, model_files=run_manifest['model_files'],
                        config_hash=run_manifest['config_hash'], instruction=runtime.config['c0_instruction'], instruction_token_ids=runtime.c0_instruction_ids,
                        status='inspect captions.jsonl free text against references, then record passed=true with reviewer'))
            else:
                from .gen import diagnose_gen
                diagnose_gen(runtime, selected, output, args.text_checkpoint, args.visual_checkpoint, args.continue_sampling)
        run_manifest['status'] = 'executed; see outputs; quality requires independent evaluation'
    except Exception as error:
        run_manifest['status'] = 'failed'
        run_manifest['failure'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        if parallel.rank() == 0:
            write_json(output/'manifest.json', run_manifest)
        if args.distributed:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
