"""Generate and score individual frozen T2I or editing benchmark cases."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from .cases import load_case, restore_geometry

DEFAULT_SCORER = '/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/scoring_sources'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--kind', choices=('t2i', 'edit'), required=True)
    gen.add_argument('--manifest', required=True)
    gen.add_argument('--record-id', required=True)
    gen.add_argument('--output', required=True)
    gen.add_argument('--model', required=True)
    gen.add_argument('--source', required=True)
    gen.add_argument('--config', required=True)
    gen.add_argument('--device', default='npu:0')
    gen.add_argument('--checkpoint-mode', choices=('trained', 'native', 'both'), required=True)
    gen.add_argument('--checkpoint')
    gen.add_argument('--source-image', help='Local copy of the frozen edit source')
    score = sub.add_parser('score')
    score.add_argument('--kind', choices=('t2i', 'edit'), required=True)
    score.add_argument('--manifest', required=True)
    score.add_argument('--record-id', required=True)
    score.add_argument('--generation', required=True)
    score.add_argument('--arm', choices=('baseline', 'native_result', 'result'), default='result')
    score.add_argument('--output', required=True)
    score.add_argument('--source-root', default=DEFAULT_SCORER)
    score.add_argument('--source-image', help='Verified local copy of the frozen edit source')
    score.add_argument('--metrics-assets', default='/mnt/sfs/250010019/unimoe_r3/outputs/0_editing7_execution_20261030/tool_assets')
    score.add_argument('--captions', help='Frozen MagicBrush local_descriptions.json')
    score.add_argument('--metrics-device', default='cpu')
    score.add_argument('--rubric-root', help='Frozen GEdit v2 rubric repository')
    score.add_argument('--reference-arm', choices=('native_result', 'result'), default='native_result')
    score.add_argument('--env-file', help='Private Metto environment file')
    return p


def generate(args):
    from qwen_latent_cot.u15_target_correction.accelerator import set_device
    from qwen_latent_cot.u15_target_correction.cli import _load_editor
    from qwen_latent_cot.u15_target_correction.config import load_config
    from qwen_latent_cot.u15_target_correction.images import save_image
    from qwen_latent_cot.u15_target_correction.inference import edit_image
    from qwen_latent_cot.u15_target_correction.loading import load_native
    from .generate import letterbox, t2i_pair

    if (args.checkpoint_mode in ('trained', 'both')) != bool(args.checkpoint):
        raise ValueError('trained and both modes require --checkpoint; native mode must omit it')
    case = load_case(args.kind, args.manifest, args.record_id, source_override=args.source_image)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, set_device(args.device), False)
    mode = 'native' if args.checkpoint_mode == 'both' else args.checkpoint_mode
    editor, meta = _load_editor(rt, cfg, mode, args.checkpoint if mode == 'trained' else None)
    def load_trained():
        from qwen_latent_cot.u15_target_correction.checkpoint import editor_identity, load_stage_weights
        return load_stage_weights(args.checkpoint, editor, editor_identity(rt, cfg))
    if case.kind == 't2i':
        loaded = []
        def load():
            loaded.append(load_trained())
        images, info = t2i_pair(rt, editor, cfg, case.prompt, case.seed,
                                 load_trained=load if args.checkpoint_mode == 'both' else None)
        if loaded:
            meta = loaded[0]
    else:
        with Image.open(case.source_image) as source:
            original_size = source.size
            letterbox(source, rt.width, rt.height).save(output / 'source_canvas.png')
        if args.checkpoint_mode == 'both':
            native_images, native_state, _ = edit_image(rt, editor, cfg, output / 'source_canvas.png', case.prompt, case.seed)
            meta = load_trained()
        images, state, info = edit_image(rt, editor, cfg, output / 'source_canvas.png', case.prompt, case.seed)
        if args.checkpoint_mode == 'both':
            images['native_result'] = native_images['result']
        import torch
        torch.save(state.detach().float().cpu(), output / 'edited_state.pt')
        if args.checkpoint_mode == 'both':
            torch.save(native_state.detach().float().cpu(), output / 'native_edited_state.pt')
        info['source_size'] = list(original_size)
    for name, image in images.items():
        save_image(output / (name + ( '_canvas.png' if case.kind == 'edit' and name in ('result', 'native_result') else '.png')), image)
    if case.kind == 'edit':
        for name in ('result', 'native_result'):
            if (output / (name + '_canvas.png')).exists():
                with Image.open(output / (name + '_canvas.png')) as canvas:
                    restore_geometry(canvas, original_size).save(output / (name + '.png'))
    image_hashes = {name: hashlib.sha256((output / (name + '.png')).read_bytes()).hexdigest()
                    for name in ('baseline', 'native_result', 'result') if (output / (name + '.png')).exists()}
    record = dict(image_sha256=image_hashes, kind=case.kind, benchmark=case.benchmark, record_id=case.record_id,
                  original_prompt=case.prompt, seed=case.seed, source_image=str(case.source_image) if case.source_image else None,
                  checkpoint_mode=args.checkpoint_mode, checkpoint=args.checkpoint,
                  checkpoint_step=meta.get('step') if meta else None,
                  model=str(Path(args.model).resolve()), source=str(Path(args.source).resolve()),
                  config=cfg, generation=info)
    (output / 'case.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(record, ensure_ascii=False))


def score(args):
    from .scoring import MODEL, load_env_file, score_abacus, score_edit_metrics, score_gedit, score_gedit2, score_imgedit, score_t2i
    if getattr(args, "env_file", None):
        load_env_file(args.env_file)
    case = load_case(args.kind, args.manifest, args.record_id, source_override=args.source_image)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    generation = Path(args.generation).resolve()
    record = json.loads((generation / 'case.json').read_text())
    expected = dict(kind=case.kind, benchmark=case.benchmark, record_id=case.record_id,
                    original_prompt=case.prompt, seed=case.seed)
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('generation record does not match the frozen case')
    if case.kind == 'edit' and args.arm == 'baseline':
        raise ValueError('The editing benchmark has no unedited generation arm')
    image = generation / (args.arm + '.png')
    if not image.is_file():
        raise FileNotFoundError(image)
    if hashlib.sha256(image.read_bytes()).hexdigest() != record.get('image_sha256', {}).get(args.arm):
        raise ValueError('generated image differs from its generation record')
    if case.kind == 't2i':
        result = score_t2i(case.row, image, args.source_root)
    elif case.benchmark == 'gedit_v2':
        if not args.rubric_root or args.reference_arm == args.arm:
            raise ValueError('GEdit v2 needs --rubric-root and a different --reference-arm')
        reference = generation / (args.reference_arm + '.png')
        if not reference.is_file() or hashlib.sha256(reference.read_bytes()).hexdigest() != record.get('image_sha256', {}).get(args.reference_arm):
            raise ValueError('reference image differs from its generation record')
        result = score_gedit2(case.row, case.source_image, reference, image, args.rubric_root)
        result['reference_arm'] = args.reference_arm
    elif case.benchmark == 'gedit':
        result = score_gedit(case.row, case.source_image, image)
    elif case.benchmark == 'imgedit':
        result = score_imgedit(case.row, case.source_image, image)
    elif case.benchmark == 'abacus':
        result = score_abacus(case.row, case.source_image, image)
    elif case.benchmark in ('magicbrush', 'emu_edit'):
        result = score_edit_metrics(case.row, image, args.metrics_assets, args.captions, args.metrics_device)
    else:
        raise ValueError('No migrated editing scorer for ' + case.benchmark)
    judge = None if case.benchmark in ('magicbrush', 'emu_edit') else MODEL
    result.update(kind=case.kind, benchmark=case.benchmark, record_id=case.record_id,
                  image=str(image), arm=args.arm, judge=judge, official_equivalent=False,
                  checkpoint_mode=record.get('checkpoint_mode'), checkpoint=record.get('checkpoint'),
                  checkpoint_step=record.get('checkpoint_step'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))


def main():
    args = parser().parse_args()
    {'generate': generate, 'score': score}[args.action](args)


if __name__ == '__main__':
    main()
