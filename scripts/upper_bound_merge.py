"""Merge sharded upper-bound runs into one metrics/gallery directory."""
import argparse
import json
import os
import shutil
from pathlib import Path

from qwen_latent_cot.upper_bound.gallery import write_gallery
from qwen_latent_cot.upper_bound.metrics import evaluate, summary_lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    rows, indexes = [], None
    for source in args.inputs:
        source = Path(source)
        records = json.loads((source / 'records.json').read_text())
        run_info = json.loads((source / 'run.json').read_text())
        current = [int(i) for i in run_info['config']['read_indexes']]
        indexes = current if indexes is None else indexes
        if current != indexes:
            raise ValueError(f'read_indexes differ in {source}')
        for record in records:
            record['_source'] = str(source)
            rows.append(record)

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    images = output / 'images'
    images.mkdir(parents=True)
    for row in rows:
        source = Path(row.pop('_source'))
        for name in [row['base']['image']] + [p['image'] for p in row['previews']]:
            os.symlink((source / 'images' / name).resolve(), images / name)
        for retry in row['retries']:
            for arm in ('loop', 'resample', 'full_resample'):
                if arm in retry:
                    os.symlink((source / 'images' / retry[arm]['image']).resolve(),
                               images / retry[arm]['image'])

    report = evaluate(rows, indexes)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'records.json').write_text(json.dumps(rows, ensure_ascii=False) + '\n')
    write_gallery(rows, output)
    for line in summary_lines(report):
        print(line)
    print(f'merged {len(rows)} prompts from {len(args.inputs)} shards')


if __name__ == '__main__':
    main()
