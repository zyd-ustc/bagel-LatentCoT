"""Shared random prompt selection and a portable generation comparison page."""
import html
import json
from pathlib import Path
import random
import secrets


def prepare(prompts, output, count, passes, early, selection_seed=None):
    pool = [json.loads(line) for line in Path(prompts).read_text().splitlines() if line.strip()]
    if not 0 < count <= len(pool):
        raise ValueError('prompt count must fit the dataset')
    seed = secrets.randbits(63) if selection_seed is None else selection_seed
    indexes = random.Random(seed).sample(range(len(pool)), count)
    rows = [dict(pool[i], source_index=i) for i in indexes]
    plan = dict(selection_seed=seed, source=str(Path(prompts).resolve()),
                population=len(pool), count=count, passes=passes, early=early, rows=rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'selection.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    return plan


def render(output):
    output = Path(output)
    plan = json.loads((output/'selection.json').read_text())
    arms = [('Base', 'base')] + [(f'early{e} · R{r}', f'early{e}_R{r}')
                                for e in plan['early'] for r in plan['passes']]
    parts = ['''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>16槽 Memory 文生图对照</title>
<style>
body{font-family:system-ui;margin:24px;background:#f4f4f4;color:#222}
section{margin-bottom:36px}.scroll{overflow-x:auto}
.grid{display:grid;gap:10px;min-width:1200px}
figure{margin:0;background:white;padding:8px;border-radius:8px}
figcaption{margin-bottom:8px;text-align:center}img{width:100%;display:block}
small{color:#666}h3{line-height:1.5}
</style><h1>Base 与连续 Memory 文生图对照</h1>''']
    parts.append(f'<p>{plan["count"]}条随机prompt · 抽样种子 {plan["selection_seed"]} · 点击图片查看原图</p>')
    for index, row in enumerate(plan['rows']):
        parts.append(f'<section><h3>{index+1}. {html.escape(row["prompt"])}</h3>')
        parts.append(f'<small>来源索引 {row["source_index"]} · {html.escape(str(row.get("prompt_id", "")))}</small>')
        parts.append(f'<div class="scroll"><div class="grid" style="grid-template-columns:repeat({len(arms)},minmax(0,1fr))">')
        for label, suffix in arms:
            name = f'p{index:02d}_{suffix}.png'
            if not (output/name).is_file():
                raise FileNotFoundError(output/name)
            parts.append(f'<figure><figcaption>{label}</figcaption><a href="{name}" target="_blank"><img loading="lazy" src="{name}"></a></figure>')
        parts.append('</div></div></section>')
    parts.append('</html>')
    (output/'index.html').write_text('\n'.join(parts), encoding='utf-8')
    print(output/'index.html', flush=True)


if __name__ == '__main__':
    import sys
    render(sys.argv[1])
