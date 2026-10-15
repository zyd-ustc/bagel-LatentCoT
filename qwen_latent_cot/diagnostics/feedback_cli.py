"""Opt-in explicit feedback inference; existing commands retain their behavior."""
import argparse
import json
from pathlib import Path
import re
import torch
from .explicit_feedback import (Feedback, reflect, intervention_state, compare,
                                save_comparison, render_review)
from .record import manifest, write_json, load_checkpoint
from .data import file_hash


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
        if not isinstance(row.get('t'), (int, float)) or not 0 < row['t'] <= 1:
            raise ValueError('each sample requires requested t in (0, 1]')
        if row.get('state'):
            row['state'] = str((path.parent/row['state']).resolve())
        if row.get('feedback'):
            if row['feedback'].get('origin', 'manual_checked') != 'manual_checked':
                raise ValueError('supplied feedback must be explicitly manually checked')
            Feedback(**row['feedback'])
    if not rows:
        raise ValueError('feedback dataset is empty')
    return rows


def main(argv=None):
    # v2 is the explicit inference default. Keep the previous single-node
    # experiment available only through its separately named v1 config.
    route = argparse.ArgumentParser(add_help=False)
    route.add_argument('--config', default='configs/explicit_feedback.json')
    selected, _ = route.parse_known_args(argv)
    if json.loads(Path(selected.config).read_text()).get('protocol') == 'explicit_feedback_restart_v2':
        from .feedback_restart_cli import main as restart_main
        return restart_main(argv)
    parser = argparse.ArgumentParser(description='Single-node explicit feedback and formal P+G inference')
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True, help='JSONL original prompt, noise_seed, t, checked feedback or automatic reflection')
    parser.add_argument('--text-checkpoint', required=True, help='frozen continuous C1/C1-read checkpoint')
    parser.add_argument('--visual-checkpoint', help='optional frozen C2/C3 student for P+G implicit arm')
    parser.add_argument('--config', default='configs/explicit_feedback.json', help='inference settings only; writer interface comes from checkpoint')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', required=True)
    parser.add_argument('--auto-reflect', action='store_true', help='also evaluate BAGEL feedback separately from checked feedback')
    parser.add_argument('--prepare-only', action='store_true', help='save exact states/full previews and a template for checked feedback')
    args = parser.parse_args(argv)
    rows = load_rows(args.data)
    settings = json.loads(Path(args.config).read_text())
    if settings.get('protocol') != 'explicit_feedback_v1':
        raise ValueError('unsupported explicit feedback inference protocol')
    if not args.prepare_only and not args.auto_reflect and any(not row.get('feedback') for row in rows):
        parser.error('provide checked feedback per row or select --auto-reflect')
    if settings['preview_kind'] not in ('complete', 'local_cfg'):
        raise ValueError('unsupported teacher preview kind')
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be empty; do not overwrite earlier experiments')
    saved = torch.load(args.visual_checkpoint or args.text_checkpoint, map_location='cpu', weights_only=True)
    config = dict(saved['config'])
    if saved['topology'] != 'C' or saved['slots'] != 16:
        raise ValueError('feedback requires a continuous C1 checkpoint with 16 slots')
    config.update(feedback_protocol=settings, protocol=settings['protocol'])
    from ..bagel.backbone import load_native
    from .runtime import DiagnosticRuntime
    runtime = DiagnosticRuntime(load_native(args.model, args.device, config['timestep_shift']), config, 'C')
    layers = settings['source_layers']
    if not layers or len(set(layers)) != len(layers) or any(type(i) is not int or not 0 <= i < len(runtime.decoder.layers) for i in layers):
        raise ValueError('invalid formal GEN source window')
    run = manifest(runtime, args, args.data, rows=rows)
    runtime.native_files = run['model_files']
    student = None
    if args.visual_checkpoint:
        from .visual_record import initialize_student
        from .memory import MemoryWriter
        student = initialize_student(runtime, args.text_checkpoint, args.visual_checkpoint).eval().requires_grad_(False)
        # Preserve the visual student's trained writer independently of fixed C1.
        runtime.writer = MemoryWriter(runtime.decoder, 'C', config['slots'], config['rank'])
    load_checkpoint(args.text_checkpoint, runtime, ('C1', 'C1-read'))
    runtime.writer.eval().requires_grad_(False)
    run.update(inference_settings=settings, text_checkpoint_sha256=file_hash(args.text_checkpoint),
               source_policy='formal P+G in GEN window; P at every other writer layer',
               feedback_scope='single native node; R1/R2; native original-prompt suffix',
               quality_status='unverified', parameters_updated=False)
    transform = None
    if args.auto_reflect and not args.prepare_only:
        from ..bagel.modeling.native_image_transform import ImageTransform
        transform = ImageTransform(settings['vit_max_image_size'], settings['vit_min_image_size'], runtime.model.vit_patch_size)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'manifest.json', run)
    records = []; templates = []; states = {}; by_id = {row['sample_id']: row for row in rows}
    def state_for(row):
        name = row['sample_id']
        if name not in states:
            if row.get('state'):
                x, flow, lengths, positions, meta = runtime.state(dict(row, label_type='observed'), row['t'], row['noise_seed'])
                states[name] = (x, row['t'], flow, lengths, positions, dict(meta, requested_t=row['t']))
            else:
                states[name] = intervention_state(runtime, row['prompt'], row['noise_seed'], row['t'])
        return states[name]
    try:
        for row in rows:
            if row.get('preview_kind', settings['preview_kind']) != settings['preview_kind']:
                raise ValueError('checked feedback preview kind differs from this run')
            x, t, flow, lengths, positions, meta = state_for(row)
            state_file = row['sample_id']+'_state.pt'
            torch.save(dict(meta, x_t=x.cpu(), state_source='trajectory'), output/state_file)
            if args.prepare_only:
                from .schedule import native_suffix
                velocity = runtime.velocity(x, t, flow)[1]
                latent = (runtime.continue_once(x, t, velocity, flow, native_suffix(config, t))
                          if settings['preview_kind'] == 'complete' else x-t*velocity)
                preview = runtime.image(latent)
                preview_file = row['sample_id']+'_preview.png'
                preview.save(output/preview_file)
                write_json(output/(row['sample_id']+'_preview.json'), dict(meta, preview=preview_file,
                                                                        preview_kind=settings['preview_kind']))
                templates.append(dict(sample_id=row['sample_id'], prompt=row['prompt'], noise_seed=row['noise_seed'],
                    t=t, state=state_file, preview=preview_file, preview_kind=settings['preview_kind'],
                    feedback=dict(observation='', gap='', correction='',
                    satisfied=False, origin='manual_checked'), **({'swap_id':row['swap_id']} if row.get('swap_id') else {})))
                continue
            donor = None
            if row.get('swap_id'):
                other = by_id.get(row['swap_id'])
                if other is None or other['prompt'] != row['prompt'] or not other.get('feedback'):
                    raise ValueError('swap partner requires the same prompt and manually checked feedback')
                _, _, _, _, _, ometa = state_for(other)
                donor = dict(feedback=Feedback(**other['feedback']), metadata=ometa)
            versions = [('manual_checked', Feedback(**row['feedback']))] if row.get('feedback') else []
            if args.auto_reflect:
                versions.append(('bagel_auto', None))
            for origin, feedback in versions:
                name = row['sample_id']+'_'+origin
                reflector = (lambda preview: reflect(runtime, preview, row['prompt'], transform,
                             settings['max_reflection_tokens'])) if feedback is None else None
                try:
                    result = compare(runtime, x, t, flow, lengths, positions, feedback=feedback,
                                     reflector=reflector, source_layers=layers, student=student,
                                     donor=donor, original_prompt=row['prompt'], preview_kind=settings['preview_kind'])
                except Exception as error:
                    if hasattr(error, 'raw_reflection'):
                        write_json(output/(name+'_reflection_failure.json'), dict(raw=error.raw_reflection, error=str(error), **meta))
                    raise
                records.append(save_comparison(output, name, result, dict(meta, sample_id=row['sample_id'],
                    feedback_origin=origin, checkpoint=args.text_checkpoint, swap_id=row.get('swap_id'),
                    sampler_config={key:config[key] for key in ('resolution', 'sampler_steps', 'timestep_shift',
                        'cfg_text_scale', 'cfg_renorm_type', 'cfg_renorm_min')})))
                render_review(output, records)
                print(f"Saved {name}: t={t:.8f}, {len(result['images'])} arms", flush=True)
        if args.prepare_only:
            (output/'feedback_template.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in templates))
        write_json(output/'summary.json', dict(comparisons=len(records), prepared_previews=len(templates),
            feedback_origins=sorted({r['feedback_origin'] for r in records}),
            quality='pending paired human annotation; no demonstrated improvement'))
        (output/'annotations.csv').write_text('sample_id,feedback_origin,arm,target_fact_fix,correct_fact_preservation,quality,notes\n'+
            ''.join(f"{r['sample_id']},{r['feedback_origin']},{arm},,,,\n" for r in records for arm in r['images']))
        run['status'] = 'prepared previews; checked feedback pending' if args.prepare_only else 'executed; final-image quality unverified'
    except Exception as error:
        run.update(status='failed', failure=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        write_json(output/'manifest.json', run)


if __name__ == '__main__':
    main()
