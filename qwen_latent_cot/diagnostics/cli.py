"""The single supported diagnostic entry point."""
import argparse
import json
import re
from pathlib import Path
import torch
from .data import load_dataset
from .record import manifest, write_json, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Frozen BAGEL independent Memory protocol (2026-10-11)')
    parser.add_argument('command', choices=['calibrate', 'train', 'caption', 'gen', 'trajectory', 'report'])
    parser.add_argument('--model')
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
    args = parser.parse_args()
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
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', r['sample_id']) for r in rows):
        raise ValueError('sample_id must be a safe filename identifier')
    if args.command in ('train', 'gen'):
        if not args.c0_pass:
            raise ValueError('first validate C0 free captions and provide --c0-pass calibration.json')
        c0 = json.loads(Path(args.c0_pass).read_text())
        if c0.get('passed') is not True:
            raise ValueError('C0 decoding has not passed; repair protocol before training/GEN interpretation')
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be empty; never mix runs or overwrite old records')
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed'])
    from ..bagel.backbone import load_native
    from .runtime import DiagnosticRuntime
    runtime = DiagnosticRuntime(load_native(args.model, args.device, config['timestep_shift']), config, args.topology)
    run_manifest = manifest(runtime, args, args.data)
    write_json(output/'manifest.json', run_manifest)
    runtime.native_files = run_manifest['model_files']
    try:
        if args.command in ('train', 'gen'):
            if c0.get('model_files') != run_manifest['model_files'] or c0.get('instruction') != config['caption_instruction']:
                raise ValueError('C0 calibration does not match native model/tokenizer or decode instruction')
        if args.command == 'caption' and args.checkpoint:
            load_checkpoint(args.checkpoint, runtime)  # Explicit init controls are allowed.
        if args.command == 'train':
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
                        config_hash=run_manifest['config_hash'], instruction=runtime.config['caption_instruction'],
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
        write_json(output/'manifest.json', run_manifest)


if __name__ == '__main__':
    main()
