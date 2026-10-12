"""Render local GEN predictions and optional before/after rows for human review."""
import argparse
import csv
import html
import json
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageChops, ImageStat

ARMS = ['G-base', 'G-layout', 'G-fixed', 'G-neutral', 'G-text', 'G-TC', 'G-swap-TC']


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def image_path(run, row):
    return run / row['predicted_image']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--baseline')
    args = parser.parse_args()
    run = Path(args.run).resolve()
    data_path = Path(args.data).resolve()
    data = {r['sample_id']: r for r in read_rows(data_path)}
    plan = json.loads((run/'plan.json').read_text())
    groups = {}
    for row in read_rows(run/'gen.jsonl'):
        key = row['sample_id'], row['noise_seed'], row['t']
        if row['arm'] in groups.setdefault(key, {}):
            raise ValueError('duplicate sample/seed/t/arm')
        groups[key][row['arm']] = row
    baseline = Path(args.baseline).resolve() if args.baseline else None
    old = {}
    if baseline:
        old_plan = json.loads((baseline/'plan.json').read_text())
        if any(old_plan['common_config'][key] != plan['common_config'][key]
               for key in ('resolution', 'posterior_seed', 'posterior_seed_policy', 'cfg_text_scale', 'cfg_renorm_min', 'cfg_renorm_type')):
            raise ValueError('before/after image/noise or CFG policies differ')
        for row in read_rows(baseline/'gen.jsonl'):
            key = row['sample_id'], row['noise_seed'], row['t']
            old.setdefault(key, {})[row['arm']] = row
        for key, group in groups.items():
            if key not in old or set(group) != set(ARMS) or set(old[key]) != set(ARMS):
                raise ValueError('before/after groups must contain the same complete seven arms')
            for arm in ARMS:
                if any(group[arm][field] != old[key][arm][field] for field in ('x_t_hash', 'prompt', 'supplementary_caption')):
                    raise ValueError('before/after must use the exact same state, prompt and caption')
    review = run/'review'; review.mkdir(exist_ok=True)
    pairs = sorted({(sample, seed) for sample, seed, t in groups})
    pages = []; stats = {}; annotations = []
    for index, (sample, seed) in enumerate(pairs):
        times = sorted(t for sid, s, t in groups if (sid, s) == (sample, seed))
        sheet = Image.new('RGB', (1600, 250*len(times)*(2 if baseline else 1)), 'white')
        draw = ImageDraw.Draw(sheet); row_index = 0
        for t in times:
            key = sample, seed, t
            variants = [('before', baseline, old[key]), ('current', run, groups[key])] if baseline else [('current', run, groups[key])]
            for label, base, group in variants:
                for column, arm in enumerate(['Source']+ARMS):
                    path = (data_path.parent/data[sample]['image']).resolve() if arm == 'Source' else image_path(base, group[arm])
                    with Image.open(path) as image:
                        tile = ImageOps.contain(image.convert('RGB'), (196, 196))
                    x = column*200; y = row_index*250
                    sheet.paste(tile, (x+(200-tile.width)//2, y+20))
                    draw.text((x+4, y+221), f'{label} t={t} {arm}', fill='black')
                row_index += 1
            for a, b in [('G-text', 'G-base'), ('G-TC', 'G-swap-TC'), ('G-TC', 'G-neutral')]:
                with Image.open(image_path(run, groups[key][a])) as aa, Image.open(image_path(run, groups[key][b])) as bb:
                    diff = ImageChops.difference(aa.convert('RGB'), bb.convert('RGB'))
                    mae = sum(ImageStat.Stat(diff).mean)/3
                stats.setdefault((t, a, b), []).append(dict(sample_id=sample, seed=seed, mae=mae))
            annotations.append([sample, seed, t, '', '', '', '', ''])
        name = f'{index:02d}_{sample}_seed{seed}.jpg'; sheet.save(review/name, quality=92)
        pages.append(f'<section><h2>{html.escape(sample)} / seed {seed}</h2><p>{html.escape(data[sample]["caption"])}</p><a href="{name}"><img src="{name}" width="1600"></a></section>')
    description = f'Current checkpoint: {plan["checkpoint"]}; SHA256: {plan["checkpoint_hash"]}. '
    if baseline:
        description += f'Before: {old_plan["checkpoint"]}; SHA256: {old_plan["checkpoint_hash"]}. '
    description += 'Same source and noise across conditions. Local x0_hat predictions, not final generated images. Click sheets for full size.'
    (review/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>GEN Memory review</title><style>body{font:16px sans-serif;max-width:1640px;margin:24px auto;background:#eee}section{background:white;padding:16px;margin:24px 0}img{max-width:100%}</style><h1>GEN Memory: before / current</h1><p>'+html.escape(description)+'</p>'+''.join(pages))
    with (review/'annotations.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'seed', 't', 'teacher_semantic_effect', 'TC_semantic_effect', 'TC_swap_difference', 'artifacts', 'notes'])
        writer.writerows(annotations)
    numeric = [dict(t=t, comparison=f'{a} vs {b}', states=len(values), mean_MAE_0_255=sum(v['mae'] for v in values)/len(values), details=values)
               for (t, a, b), values in sorted(stats.items())]
    (run/'pixel_comparison.json').write_text(json.dumps(numeric, indent=2)+'\n')
    print(review/'index.html')


if __name__ == '__main__':
    main()
