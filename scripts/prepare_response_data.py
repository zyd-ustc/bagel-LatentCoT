"""Reuse C1 captions and split existing GEN images into independent query pools."""
import argparse
import json
from pathlib import Path


def write_once(path, text):
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'different existing data file: {path}')
        return
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default='/data/zyd_workspace/bagel-LatentCoT/datasets')
    args = parser.parse_args()
    root = Path(args.dataset_root)
    source = root/'memory_gen_sample16.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if len(rows) != 16:
        raise ValueError('expected the existing 16-source GEN manifest')
    for row in rows:
        row['split'] = 'val' if row['pair_id'] in ('sample_pair_01', 'sample_pair_04') else 'train'
        row['query_role'] = 'frozen_GEN_queries_independent_of_training_target_caption'
    if sum(row['split'] == 'val' for row in rows) != 4:
        raise ValueError('the four reviewed high-noise images must remain validation queries')
    write_once(root/'memory_response_queries16.jsonl', ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    captions = Path(__file__).resolve().parents[1]/'data/c1_text80.jsonl'
    write_once(root/'c1_text80.jsonl', captions.read_text())
    print(f'Caption manifest: {root}/c1_text80.jsonl (original train/val/test split retained)')
    print(f'Query manifest: {root}/memory_response_queries16.jsonl (12 train images / 4 validation images)')


if __name__ == '__main__':
    main()
