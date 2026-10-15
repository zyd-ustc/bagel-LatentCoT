"""Generate a complete image, reflect and regenerate with fixed text memory."""
import argparse
import json
from pathlib import Path
import re
import torch
from .data import file_hash, tensor_hash
from .record import manifest, write_json, load_checkpoint
from .explicit_feedback import (Feedback, reflect, PROTOCOL, validate_windows, generate_base,
                                compare, save_comparison, render_review)

SAMPLER_KEYS = ('resolution', 'sampler_steps', 'timestep_shift', 'cfg_text_scale', 'cfg_renorm_type', 'cfg_renorm_min')


def load_rows(path):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = set()
    for row in rows:
        name = row.get('sample_id', '')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in ids:
            raise ValueError('sample_id must be unique and safe for filenames')
        ids.add(name)
        if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
            raise ValueError('each sample requires the original prompt')
        if type(row.get('noise_seed')) is not int or row['noise_seed'] < 0:
            raise ValueError('each sample requires a nonnegative integer noise_seed')
        if row.get('initial_state'):
            row['initial_state'] = str((path.parent/row['initial_state']).resolve())
        if row.get('feedback'):
            if row['feedback'].get('origin', 'manual_checked') != 'manual_checked':
                raise ValueError('supplied feedback must be manually checked')
            Feedback(**row['feedback'])
    if not rows:
        raise ValueError('feedback dataset is empty')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True, help='JSONL sample_id, original prompt, noise_seed; optional checked feedback')
    parser.add_argument('--text-checkpoint', required=True, help='frozen continuous C1/C1-read checkpoint')
    parser.add_argument('--config', default='configs/explicit_feedback.json')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', required=True)
    parser.add_argument('--early', nargs='+', type=int, help='number of initial Euler updates with fixed memory; default 10 20')
    parser.add_argument('--auto-reflect', action='store_true')
    parser.add_argument('--text-control', action='store_true', help='also rerun with native reflection text for the same early windows')
    parser.add_argument('--prepare-only', action='store_true', help='save full native image, exact initial noise and a checked-feedback template')
    args = parser.parse_args(argv)
    rows = load_rows(args.data)
    settings = json.loads(Path(args.config).read_text())
    if settings.get('protocol') != PROTOCOL:
        raise ValueError('configuration must select explicit_feedback')
    early = args.early if args.early is not None else settings['early']
    settings = dict(settings, early=early)
    if not args.prepare_only and not args.auto_reflect and any(not row.get('feedback') for row in rows):
        parser.error('provide checked feedback per sample or select --auto-reflect')
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be empty; never overwrite earlier results')
    saved = torch.load(args.text_checkpoint, map_location='cpu', weights_only=True)
    config = dict(saved['config'])
    if saved['topology'] != 'C' or saved['slots'] != 16:
        raise ValueError('restart feedback requires continuous C1 memory with 16 slots')
    validate_windows(config, early)
    config.update(protocol=PROTOCOL, feedback_protocol=settings)
    from ..bagel.backbone import load_native
    from .runtime import DiagnosticRuntime
    runtime = DiagnosticRuntime(load_native(args.model, args.device, config['timestep_shift']), config, 'C')
    run = manifest(runtime, args, args.data, rows=rows)
    runtime.native_files = run['model_files']
    load_checkpoint(args.text_checkpoint, runtime, ('C1', 'C1-read'))
    runtime.writer.eval().requires_grad_(False)
    sampler = {key:config[key] for key in SAMPLER_KEYS}
    run.update(protocol=PROTOCOL, inference_settings=settings, sampler_config=sampler,
        text_checkpoint_sha256=file_hash(args.text_checkpoint),
        initial_asset_hashes={row['initial_state']:file_hash(row['initial_state']) for row in rows if row.get('initial_state')},
        source_policy='original prompt plus native reflection-text KV at all writer layers',
        feedback_scope='complete native image; fixed text memory; restart identical initial noise; early windows',
        writer_updates_per_feedback=1, shallow_loop=False, gen_passes_per_step=1,
        parameters_updated=False, quality_status='unverified')
    transform = None
    if args.auto_reflect and not args.prepare_only:
        from ..bagel.modeling.native_image_transform import ImageTransform
        transform = ImageTransform(settings['vit_max_image_size'], settings['vit_min_image_size'], runtime.model.vit_patch_size)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'manifest.json', run)
    records = []; templates = []
    try:
        for row in rows:
            initial = None
            if row.get('initial_state'):
                locked = torch.load(row['initial_state'], map_location=runtime.device, weights_only=True)
                if (locked.get('protocol'), locked.get('prompt'), locked.get('noise_seed')) != (PROTOCOL,row['prompt'],row['noise_seed']):
                    raise ValueError('saved initial state differs from declared prompt/seed/protocol')
                if locked.get('model_files') != runtime.native_files or locked.get('sampler_config') != sampler:
                    raise ValueError('saved initial state has different model/sampler provenance')
                initial = locked['x_initial']
                if locked.get('initial_noise_hash') != tensor_hash(initial):
                    raise ValueError('saved initial noise hash differs')
            prepared = generate_base(runtime, row['prompt'], row['noise_seed'], initial)
            initial_file = row['sample_id']+'_initial.pt'
            torch.save(dict(protocol=PROTOCOL, x_initial=prepared['initial'].cpu(), prompt=row['prompt'],
                noise_seed=row['noise_seed'], initial_noise_hash=tensor_hash(prepared['initial']),
                model_files=runtime.native_files, sampler_config=sampler), output/initial_file)
            image_file = row['sample_id']+'_Base.png'
            prepared['image'].save(output/image_file)
            metadata = dict(sample_id=row['sample_id'], prompt=row['prompt'], noise_seed=row['noise_seed'],
                initial_state=initial_file, sampler_config=sampler, checkpoint=args.text_checkpoint,
                source_index=row.get('source_index'))
            if args.prepare_only:
                write_json(output/(row['sample_id']+'_base.json'), dict(metadata,
                    initial_noise_hash=tensor_hash(prepared['initial']), nodes=prepared['base']['nodes'], base_image=image_file))
                templates.append(dict(sample_id=row['sample_id'], prompt=row['prompt'], noise_seed=row['noise_seed'],
                    initial_state=initial_file, base_image=image_file, feedback=dict(observation='', gap='', correction='',
                    satisfied=False, origin='manual_checked')))
                continue
            versions = [('manual_checked', Feedback(**row['feedback']))] if row.get('feedback') else []
            if args.auto_reflect:
                versions.append(('bagel_auto', None))
            for origin, feedback in versions:
                name = row['sample_id']+'_'+origin
                reflector = (lambda image: reflect(runtime, image, row['prompt'], transform,
                    settings['max_reflection_tokens'])) if feedback is None else None
                try:
                    result = compare(runtime, prepared, early=early, feedback=feedback,
                                             reflector=reflector, text_control=args.text_control)
                except Exception as error:
                    if hasattr(error, 'raw_reflection'):
                        write_json(output/(name+'_reflection_failure.json'), dict(metadata, raw=error.raw_reflection, error=str(error)))
                    raise
                records.append(save_comparison(output, name, result, dict(metadata, feedback_origin=origin)))
                render_review(output, records)
                print(f'Saved {name}: full restart, early={early}, one GEN per sampling update', flush=True)
        if args.prepare_only:
            (output/'feedback_template.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in templates))
        write_json(output/'summary.json', dict(protocol=PROTOCOL, comparisons=len(records),
            prepared_images=len(templates), early=early, gen_passes_per_step=1,
            quality='pending paired final-image annotation; no demonstrated improvement'))
        (output/'annotations.csv').write_text('sample_id,feedback_origin,arm,target_fact_fix,correct_fact_preservation,quality,notes\n'+
            ''.join(f"{r['sample_id']},{r['feedback_origin']},{arm},,,,\n" for r in records for arm in r['images']))
        run['status'] = 'prepared complete images; checked feedback pending' if args.prepare_only else 'executed; final-image quality unverified'
    except Exception as error:
        run.update(status='failed', failure=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        write_json(output/'manifest.json', run)


if __name__ == '__main__':
    main()
