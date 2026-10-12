"""Sample short text-oracle descriptions and independent GEN query images."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from transformers import AutoTokenizer
from prepare_response_data import write_once


def reservoir(stream, count, rng):
    chosen = []; seen = 0
    for row in stream:
        seen += 1
        if len(chosen) < count:
            chosen.append(row)
        else:
            index = rng.randrange(seen)
            if index < count:
                chosen[index] = row
    if len(chosen) != count:
        raise ValueError(f'only {seen} eligible samples, need {count}')
    return chosen


def source_rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default='/data/zyd_workspace/bagel-LatentCoT/datasets')
    parser.add_argument('--model', default='/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT')
    parser.add_argument('--seed', type=int, default=20261012)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    manifests = root/'qwen_latent_cot_v3_adapted_20260816/manifests'
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    reviewed = list(source_rows(root/'memory_gen_sample16.jsonl'))
    held_out = list(source_rows(root/'memory_response_queries16.jsonl'))
    held_out = [r for r in held_out if r['split'] == 'val']
    excluded = {r.get('source_sample_id', r['sample_id']) for r in reviewed}
    # Original sample IDs can have a diagnostic prefix. Image directory is the source ID.
    excluded.update(Path(r['image']).parent.name for r in reviewed)
    used_ids = set(excluded); used_text = set()
    captions = list(source_rows(root/'c1_text80.jsonl'))
    for row in captions:
        row['dataset_group'] = 'legacy_template'
        used_text.add(' '.join(row['caption'].lower().split()))
    for split, count in [('train', 2048), ('val', 32), ('test', 32)]:
        def eligible():
            for row in source_rows(manifests/f'cort36k_{split}.jsonl'):
                text = ' '.join(row.get('prompt', '').split())
                normalized = text.lower()
                sid = row['sample_id']
                if sid in used_ids or normalized in used_text or row.get('chain_status') != 'terminal':
                    continue
                if not 6 <= len(text.split()) <= 30 or '?' in text or not re.match(r'^(A|An|The|One|Two|Three|Four|Five)\b', text):
                    continue
                if len(tokenizer.encode(text, add_special_tokens=False)) > 40:
                    continue
                used_ids.add(sid); used_text.add(normalized)
                yield dict(sample_id='natural_'+sid, scene_id='natural_'+sid, family_id='natural_'+sid,
                    split=split, caption=text, label_type='clean', verified=True, facts=[],
                    dataset_group='natural_text', source_sample_id=sid, source_manifest=f'cort36k_{split}.jsonl',
                    verification_scope='text_oracle_only; source generation prompt is not a verified image caption')
        selected = reservoir(eligible(), count, random.Random(args.seed+len(captions)))
        captions.extend(sorted(selected, key=lambda r: r['sample_id']))
        print(f'{split}: added {count} natural text descriptions', flush=True)
    caption_ids = {r.get('source_sample_id') for r in captions}
    image_hashes = {hashlib.sha256(Path(r['image']).read_bytes()).hexdigest() for r in reviewed}
    def query_candidates():
        seen_ids = set()
        for row in source_rows(manifests/'cort36k_train.jsonl'):
            sid = row['sample_id']
            if sid in caption_ids or sid in excluded or sid in seen_ids or not row.get('image_paths'):
                continue
            seen_ids.add(sid)
            image = Path(row['image_paths'][-1].replace('/data/bagel-LatentCoT/datasets/', str(root)+'/'))
            if image.is_file():
                yield dict(sample_id='query_'+sid, scene_id='query_'+sid, family_id='query_'+sid,
                    split='train', caption='An image.', label_type='clean', verified=True, facts=[], image=str(image),
                    source_sample_id=sid, query_role='frozen_GEN_queries_independent_of_training_target_caption',
                    verification_scope='existing_image_asset_only; no semantic image caption asserted')
    # Reservoir selects paths without decoding the image library. Hash only selected candidates.
    candidates = reservoir(query_candidates(), 160, random.Random(args.seed+71))
    queries = []
    for row in candidates:
        digest = hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()
        if digest not in image_hashes:
            image_hashes.add(digest); queries.append(row)
        if len(queries) == 128:
            break
    if len(queries) != 128:
        raise ValueError('insufficient distinct sampled query images')
    queries.sort(key=lambda r: r['sample_id'])
    queries.extend(held_out)
    text_path = root/'c1_response_text2192.jsonl'
    query_path = root/'memory_response_queries132.jsonl'
    write_once(text_path, ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in captions))
    write_once(query_path, ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in queries))
    print(f'Captions: {text_path} (2112 train / 40 val / 40 test)')
    print(f'Queries: {query_path} (128 train / original 4 val)')


if __name__ == '__main__':
    main()
