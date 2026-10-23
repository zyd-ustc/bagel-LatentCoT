"""Attach full-resample verdicts to an upper-bound run and re-evaluate."""
import argparse
import json
from pathlib import Path

from qwen_latent_cot.upper_bound.metrics import evaluate, summary_lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True)
    parser.add_argument('--full', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    full = {}
    for path in args.full:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                full[row['prompt_id']] = row

    records = json.loads(Path(args.records).read_text())
    indexes = None
    attached = 0
    for record in records:
        row = full.get(record['prompt_id'])
        if row is None:
            continue
        indexes = [int(r['index']) for r in record['retries']]
        for retry in record['retries']:
            retry['full_resample'] = dict(seed=row['seed'], image=row['image'],
                                          atoms=row['atoms'], passed=row['passed'],
                                          steps=50)
            attached += 1
    if indexes is None:
        raise SystemExit('no retries found in records')

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = evaluate(records, sorted(set(indexes)))
    (output / 'metrics_full.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'attached {attached} full-resample verdicts')
    for line in summary_lines(report):
        print(line)
    for point in report['read_points']:
        retry = point['retry']
        if retry.get('full_rate') is not None:
            print(f"t={point['t']:.3f} full_resample={retry['full_rate']:.3f} "
                  f"(loop={retry['loop_rate']:.3f} partial={retry['resample_rate']:.3f})")
        atoms = point['atoms']
        compact = {k: (round(v.get('loop_rate') or 0, 3), round(v.get('resample_rate') or 0, 3),
                       round(v.get('full_rate') or 0, 3), v['atoms']) for k, v in atoms.items()}
        print(f"   atoms (loop, partial, full, n) {compact}")


if __name__ == '__main__':
    main()
