"""Build all terminal CoRT consistency pairs; retain source label provenance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image
from transformers import AutoTokenizer


def records(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def relocated(path, root):
    old = '/data/bagel-LatentCoT/datasets/'
    return Path(str(root) + '/' + path[len(old):]) if path.startswith(old) else Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default='/data/zyd_workspace/bagel-LatentCoT/datasets')
    parser.add_argument('--model', default='/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT')
    parser.add_argument('--output', help='fresh JSONL path; defaults to visual_memory_full.jsonl')
    args = parser.parse_args()
    root = Path(args.dataset_root).resolve()
    output = Path(args.output).resolve() if args.output else root / 'visual_memory_full.jsonl'
    summary_path = output.with_suffix('.summary.json')
    if output.exists() or summary_path.exists():
        raise ValueError('output already exists; keep the locked dataset or use a fresh path')
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    manifests = root / 'qwen_latent_cot_v3_adapted_20260816/manifests'
    held_out = set()
    legacy = root / 'memory_gen_sample16.jsonl'
    if legacy.exists():
        for row in records(legacy):
            held_out.add(row.get('source_sample_id', row['sample_id']))
            held_out.add(Path(row['image']).parent.name)
    counts = Counter(); lengths = Counter(); seen_ids = {}; seen_images = {}; sources = {}
    rejected = Counter(); total = 0
    temporary = output.with_suffix('.jsonl.partial')
    if temporary.exists():
        raise ValueError('partial output exists; inspect it before using a fresh output path')
    with temporary.open('x') as stream:
        # Preserve held-out content if source manifests contain exact duplicates.
        for split in ('test', 'val', 'train'):
            source = manifests / f'cort36k_{split}.jsonl'
            sources[split] = dict(path=str(source), sha256=digest(source))
            for source_row in records(source):
                total += 1
                sid = source_row['sample_id']
                if sid in held_out:
                    rejected['legacy_review_image'] += 1
                    continue
                steps = source_row.get('steps', [])
                terminal = steps[-1] if steps else {}
                if (source_row.get('chain_status') != 'terminal'
                    or source_row.get('adapter_role') != 'stage11_stage12_stage2_gold'
                    or terminal.get('fix')
                    or terminal.get('analysis', '').strip().lower() != 'the image matches the prompt.'):
                    rejected['no_terminal_consistency_annotation'] += 1
                    continue
                caption = source_row.get('prompt', '').strip()
                images = source_row.get('image_paths', [])
                if not caption or not images:
                    rejected['missing_caption_or_image_reference'] += 1
                    continue
                # The generation prompt describes the terminal image, not img0 before edits.
                image = relocated(images[-1], root).resolve()
                if not image.is_file():
                    rejected['missing_image'] += 1
                    continue
                tokens = len(tokenizer.encode(caption, add_special_tokens=False))
                # Leave room for the native prompt wrapper below writer position 2048.
                if tokens > 2040:
                    rejected['caption_exceeds_native_source_positions'] += 1
                    continue
                try:
                    with Image.open(image) as view:
                        width, height = view.size
                        view.verify()
                except (OSError, ValueError, SyntaxError):
                    rejected['unreadable_image'] += 1
                    continue
                if sid in seen_ids:
                    rejected['duplicate_source_id'] += 1
                    continue
                image_hash = digest(image)
                if image_hash in seen_images:
                    rejected['duplicate_image_content'] += 1
                    continue
                seen_ids[sid] = split; seen_images[image_hash] = split
                row = dict(sample_id='visual_' + sid, scene_id=sid, family_id=sid,
                    split=split, image=str(image), image_sha256=image_hash,
                    image_width=width, image_height=height, prompt='An image.',
                    label_kind='target_clean', label_type='clean', verified=True,
                    target_caption=caption, target_facts=[], caption_token_count=tokens,
                    label_origin='cort36k_terminal_consistency_annotation', human_reviewed=False,
                    verification_scope='source_dataset_terminal_image_prompt_consistency; not_manual_caption_review',
                    source_sample_id=sid, source_manifest=str(source), source_split=split,
                    source_image_index=len(images)-1, source_num_turns=source_row.get('num_turns'),
                    source_generator=source_row.get('generator_model'),
                    source_terminal_analysis=terminal['analysis'])
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                counts[split] += 1
                bucket = '<=48' if tokens <= 48 else '49-128' if tokens <= 128 else '>128'
                lengths[bucket] += 1
                if sum(counts.values()) % 2000 == 0:
                    print(f'Accepted {sum(counts.values())} pairs; current split={split}', flush=True)
    temporary.replace(output)
    summary = dict(dataset=str(output), sha256=digest(output), input_rows=total,
        accepted_rows=sum(counts.values()), splits=dict(counts), rejected=dict(rejected),
        caption_token_buckets=dict(lengths), source_manifests=sources,
        split_policy='source split unchanged; duplicate content priority test > val > train',
        image_policy='terminal image only; byte SHA256 deduplication',
        label_policy='source consistency labels; no new manual review; no invented fact annotations',
        caption_policy='complete source prompt, no truncation; neutral student GEN prompt',
        excluded_legacy_manifest=str(legacy) if legacy.exists() else None,
        fact_metrics='unavailable: target_facts is empty; use captions/NLL and image review',
        training_budget='all training pairs in pool; number of updates controlled by --steps')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
