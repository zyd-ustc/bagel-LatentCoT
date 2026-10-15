"""Convert terminal CoRT36k chains into C1 text-feedback compression pairs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


def relocated(path, root):
    old = '/data/bagel-LatentCoT/datasets/'
    return root / path[len(old):] if path.startswith(old) else Path(path)


def chain_rows(chain, split, root, tokenize, max_tokens=48):
    """Use image i for fix i; terminal image for the explicit keep instruction."""
    if (chain.get('chain_status') != 'terminal' or
            chain.get('adapter_role') != 'stage11_stage12_stage2_gold'):
        return [], 'not_terminal_gold'
    steps, images = chain.get('steps', []), chain.get('image_paths', [])
    if not steps or len(steps) != len(images) or len(steps) != chain.get('num_turns', -1) + 1:
        return [], 'chain_alignment'
    if steps[-1].get('fix') or steps[-1].get('analysis', '').strip().lower() != 'the image matches the prompt.':
        return [], 'terminal_label'
    prompt = chain.get('prompt', '').strip()
    if not prompt:
        return [], 'missing_prompt'
    sid = chain['sample_id']; rows = []
    for index, step in enumerate(steps):
        satisfied = index == len(steps) - 1
        correction = '' if satisfied else step.get('fix', '').strip()
        caption = 'No change is needed.' if satisfied else correction
        if not satisfied and (not correction or not step.get('analysis', '').strip()):
            return [], 'missing_feedback'
        if step.get('output_image') and (satisfied or Path(step['output_image']).name != Path(images[index+1]).name):
            return [], 'output_alignment'
        # Keep the full instruction; exclude rather than truncate long corrections.
        if len(tokenize(caption)) > max_tokens or len(tokenize(prompt)) + len(tokenize(caption)) > 2000:
            continue
        image = relocated(images[index], root).resolve()
        rows.append(dict(sample_id=f'historical_{sid}_step{index}', scene_id=sid, family_id=sid,
            split=split, prompt=prompt, image=str(image), caption=caption, facts=[],
            label_kind='feedback', label_type='clean', verified=True, human_reviewed=False,
            feedback=dict(observation=step.get('analysis', ''), gap='' if satisfied else step['analysis'],
                          correction=correction, satisfied=satisfied, origin='source_annotation'),
            verification_scope='source_terminal_consistency_annotation; not_manual_review_or_BAGEL_trajectory',
            training_usage='C1_prompt_correction_compression_and_native_text_velocity_only',
            source_sample_id=sid, source_image_index=index, source_generator=chain.get('generator_model'),
            source_terminal_image=str(relocated(images[-1], root).resolve()),
            source_terminal_analysis=steps[-1]['analysis']))
    return rows, None


def prepare(root, output, tokenize, *, max_tokens=48, keep_ratio=.25, seed=20261014):
    from PIL import Image
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or output.with_suffix('.summary.json').exists():
        raise ValueError('use a fresh historical feedback output')
    manifests = root/'qwen_latent_cot_v3_adapted_20260816/manifests'
    excluded = set()
    legacy = root/'memory_gen_sample16.jsonl'
    if legacy.exists():
        for line in legacy.read_text().splitlines():
            row = json.loads(line)
            excluded.add(Path(row['image']).parent.name)
    corrections, keeps, rejected = [], [], Counter()
    sources = {}; asset_splits = {}; image_cache = {}; family_splits = {}
    for split in ('test', 'val', 'train'):
        manifest = manifests/f'cort36k_{split}.jsonl'
        sources[split] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        with manifest.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                chain = json.loads(line)
                sid = chain['sample_id']
                if sid in excluded:
                    rejected['legacy_review_family'] += 1; continue
                if family_splits.setdefault(sid, split) != split:
                    raise ValueError('source family leaks across splits')
                rows, reason = chain_rows(chain, split, root, tokenize, max_tokens)
                if reason:
                    rejected[reason] += 1; continue
                for row in rows:
                    image = row['image']
                    if image not in image_cache:
                        try:
                            with Image.open(image) as view:
                                view.verify()
                            image_cache[image] = hashlib.sha256(Path(image).read_bytes()).hexdigest()
                        except (OSError, ValueError, SyntaxError):
                            image_cache[image] = None
                    digest = image_cache[image]
                    if digest is None:
                        rejected['missing_or_unreadable_image'] += 1; continue
                    if asset_splits.setdefault(digest, split) != split:
                        rejected['image_content_split_collision'] += 1; continue
                    row['image_sha256'] = digest
                    row['source_manifest'] = str(manifest)
                    (keeps if row['feedback']['satisfied'] else corrections).append(row)
    selected = list(corrections)
    for split in ('train', 'val', 'test'):
        pool = sorted((r for r in keeps if r['split'] == split), key=lambda r:r['sample_id'])
        random.Random(f'{seed}:{split}').shuffle(pool)
        count = int(sum(r['split'] == split for r in corrections)*keep_ratio)
        selected.extend(pool[:count])
    selected.sort(key=lambda r:(r['split'], r['sample_id']))
    if not any(r['split']=='train' for r in corrections) or not any(r['split']=='val' for r in corrections):
        raise ValueError('historical feedback needs train and val corrections')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = dict(rows=len(selected), splits=dict(Counter(r['split'] for r in selected)),
        corrections=dict(Counter(r['split'] for r in corrections)),
        keeps=dict(Counter(r['split'] for r in selected if r['feedback']['satisfied'])),
        rejected=dict(rejected), max_correction_tokens=max_tokens, keep_to_correction_ratio=keep_ratio,
        seed=seed, source_manifest_sha256=sources, sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        feedback_origin='source_annotation', human_reviewed=False,
        state_policy='controlled noise from historical source image; not original generator trajectory',
        supervision='correction text and frozen native text conditional velocity; no paired correction flow target')
    output.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return summary


def pilot(data, output, seed=20261014):
    """Bound the first run and its readout validation, preserving source splits."""
    data, output = Path(data), Path(output)
    if output.exists():
        raise ValueError('use a fresh pilot output')
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    selected = []
    for split, limit in (('train', 512), ('val', 32), ('test', 32)):
        for satisfied, count in ((False, limit), (True, limit//4)):
            pool = [r for r in rows if r['split'] == split and r['feedback']['satisfied'] == satisfied]
            random.Random(f'{seed}:{split}:{satisfied}').shuffle(pool)
            selected.extend(pool[:count])
    selected.sort(key=lambda r:(r['split'], r['sample_id']))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    summary = dict(rows=len(selected), splits=dict(Counter(r['split'] for r in selected)),
                   source=str(data.resolve()), source_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                   seed=seed, selection='512/32/32 corrections plus up to 25% keep rows',
                   sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return summary


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default='/data/zyd_workspace/bagel-LatentCoT/datasets')
    parser.add_argument('--model', default='/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT')
    parser.add_argument('--output', required=True)
    parser.add_argument('--pilot-output', help='also write a bounded 640 train / 40 val / 40 test pilot')
    parser.add_argument('--max-tokens', type=int, default=48)
    parser.add_argument('--keep-ratio', type=float, default=.25)
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.keep_ratio < 0:
        parser.error('max-tokens must be positive and keep-ratio nonnegative')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(json.dumps(prepare(args.dataset_root, args.output,
        lambda text:tokenizer.encode(text, add_special_tokens=False), max_tokens=args.max_tokens,
        keep_ratio=args.keep_ratio), ensure_ascii=False, indent=2))
    if args.pilot_output:
        print(json.dumps(pilot(args.output, args.pilot_output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
