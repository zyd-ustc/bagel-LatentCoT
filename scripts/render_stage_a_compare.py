"""Validate all sampling outputs and build an offline HTML comparison."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
from PIL import Image
from sample_stage_a_compare import load_cases,arms_for,ARM_LABELS


def render(output):
    output=Path(output)
    cases=load_cases(output/'prompts.jsonl')
    records=[]
    for i,case in enumerate(cases):
        record=json.loads((output/f'case_{i:03d}.json').read_text())
        if record['id']!=case['id'] or set(record['arms'])!=set(arms_for(case)):
            raise ValueError('Incomplete or mismatched case: '+case['id'])
        for arm in record['arms'].values():
            with Image.open(output/arm['image']) as image:
                image.verify()
        records.append(record)
    checkpoints={r['checkpoint'] for r in records};steps={r['checkpoint_step'] for r in records};seeds={r['seed'] for r in records}
    if len(checkpoints)!=1 or len(steps)!=1 or len(seeds)!=1:raise ValueError('Mixed checkpoint/seed comparison')
    workers=[json.loads(p.read_text()) for p in sorted(output.glob('worker_*.json'))]
    if not workers or any(w['status']!='complete' for w in workers):raise ValueError('Sampling workers did not all complete')
    summary=dict(cases=len(records),images=sum(len(r['arms']) for r in records),groups=dict(Counter(r['group'] for r in records)),
                 checkpoint=next(iter(checkpoints)),checkpoint_step=next(iter(steps)),seed=next(iter(seeds)),workers=len(workers),
                 sampling=workers[0]['sampling'],blocks=workers[0]['blocks'],active_updates=workers[0]['active_updates'],
                 max_peak_memory_gib=max(a['peak_memory_gib'] for r in records for a in r['arms'].values()),
                 semantic_evaluation='Manual inspection against per-case checks; no automatic semantic win rate assigned')
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    esc=lambda value:html.escape(str(value),quote=True)
    sections=[]
    for index,r in enumerate(records):
        figures={}
        for arm,a in r['arms'].items():
            figures[arm]=f'<figure><a href="{esc(a["image"])}" target="_blank"><img src="{esc(a["image"])}" width="512" height="512" loading="lazy" alt="{esc(r["id"]+" "+ARM_LABELS[arm])}"></a><figcaption><b>{esc(ARM_LABELS[arm])}</b><span>{a["seconds"]:.1f}s · {a["peak_memory_gib"]:.1f} GiB</span></figcaption></figure>'
        full='<div class="image-grid full-view">'+figures['native_full']+figures['loop_full']+'</div>'
        gap=('<div class="image-grid gap-view">'+''.join(figures[a] for a in ('native_partial','loop_partial','wrong_reader'))+'</div>') if r.get('removed_clauses') else '<p class="gap-view empty">短 Prompt 对照：不执行句子删除。</p>'
        checks=''.join('<li>'+esc(c)+'</li>' for c in r.get('checks',[]))
        details=f'<details><summary>查看 Prompt、删除句子和错配 UND 文本</summary><div class="prompts"><div><h4>完整 Prompt → UND / 完整组 GEN</h4><p>{esc(r["full_prompt"])}</p></div><div><h4>删减 Prompt → 信息差组 GEN</h4><p>{esc(r["partial_prompt"])}</p></div><div><h4>删除句子 → 观察是否被补回</h4><p>{esc(r.get("removed_clauses") or "无")}</p></div><div><h4>错配 UND：{esc(r["donor_id"])}</h4><p>{esc(r["donor_prompt"])}</p></div></div></details>'
        if not r.get('removed_clauses'):
            details=f'<details><summary>查看完整 Prompt 与评测标签</summary><div class="prompts"><div><h4>完整 Prompt</h4><p>{esc(r["full_prompt"])}</p></div><div><h4>原始评测标签</h4><p>{esc(json.dumps(r.get("benchmark_metadata",{}),ensure_ascii=False))}</p></div></div></details>'
        sections.append(f'<section class="case" data-group="{esc(r["group"])}"><header><div class="case-number">{index+1:02d}</div><div><p class="eyebrow">{esc(r["group"])} / {esc(r["category"])}</p><h2>{esc(r["id"])}</h2></div></header><ul class="checks">{checks}</ul>{full}{gap}{details}</section>')
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage A · Semantic comparison</title>
<style>
:root{--paper:#f6f4ee;--ink:#172c3b;--muted:#617077;--line:#d7dcd7;--accent:#b34f21}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 Georgia,"Songti SC",serif}.wrap{max-width:1480px;margin:auto;padding:32px}.masthead{border-top:5px solid var(--ink);padding:30px 0}.eyebrow{font:12px/1.5 Menlo,monospace;letter-spacing:.08em;color:var(--muted);margin:0 0 5px}h1{font-size:clamp(36px,5vw,68px);line-height:1.05;letter-spacing:-.04em;margin:15px 0 20px}h1 span{color:var(--accent)}.lede{max-width:860px}.meta{font:13px/1.7 Menlo,monospace;overflow-wrap:anywhere;color:var(--muted)}.toolbar{position:sticky;top:0;z-index:3;background:var(--paper);padding:16px 0;border-block:1px solid var(--ink);display:flex;gap:20px;flex-wrap:wrap;align-items:center}button,select{font-family:inherit;font-size:15px;line-height:1.5;color:var(--ink);padding:9px 14px;border:1px solid var(--line);background:transparent;cursor:pointer}button.active{background:var(--ink);color:white;border-color:var(--ink)}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid var(--accent);outline-offset:3px}.case{padding:42px 0;border-bottom:1px solid var(--ink)}.case>header{display:flex;gap:20px;align-items:center}.case-number{font:42px/1 Menlo,monospace;color:var(--accent)}h2{font-size:25px;margin:0;overflow-wrap:anywhere}.checks{padding-left:22px;margin:18px 0}.image-grid{display:grid;gap:18px}.full-view{grid-template-columns:repeat(2,minmax(0,1fr))}.gap-view{grid-template-columns:repeat(3,minmax(0,1fr));display:none}body.gap .full-view{display:none}body.gap .gap-view{display:grid}figure{margin:0;background:#fff}img{display:block;width:100%;height:auto}figcaption{padding:12px 14px;border-bottom:2px solid var(--line);font-size:15px}figcaption span{display:block;color:var(--muted);font:12px/1.6 Menlo,monospace}details{margin-top:20px;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;color:var(--accent)}.prompts{display:grid;grid-template-columns:1fr 1fr;gap:24px;overflow-wrap:anywhere}.prompts h4{font-size:15px;margin-bottom:6px}.prompts p{font:15px/1.65 Georgia,serif;margin-top:0}.empty{padding:20px;background:#e9e7de}footer{padding:30px 0;color:var(--muted);font-size:14px}[hidden]{display:none!important}@media(max-width:760px){.wrap{padding:18px}.gap-view,.full-view,.prompts{grid-template-columns:1fr}h1{font-size:42px}.toolbar{gap:10px}}
</style><main class="wrap"><div class="masthead"><p class="eyebrow">BAGEL / FULL-LAYER LOOP / STAGE A</p><h1>先看语义，<span>再看变化。</span></h1><p class="lede">完整组比较普通生图；信息差组检查 UND 能否把 GEN 没看见的句子带回图中。所有组使用同一初始噪声种子。点击图片查看原尺寸。</p><p class="meta">__META__</p><p class="meta">__CHECKPOINT__</p></div><nav class="toolbar"><div><button class="active" id="full" type="button">完整 Prompt · 2 组</button><button id="gap" type="button">信息差 · 3 组</button></div><label>样本 <select id="filter"><option value="all">全部</option><option value="controlled_long">可核验长 Prompt</option><option value="heldout_long">v1 留出长 Prompt</option><option value="short_control">短 Prompt 对照</option></select></label><span id="count" class="meta"></span></nav>__SECTIONS__<footer>错配组只把 UND 文本换为另一条 Prompt，GEN 仍看本条删减文本；它不是另一张图的 Memory 交换。按每条检查项人工判断语义符合情况，不能把图片变化或 RGB 距离当作改进。</footer></main><script>
const full=document.querySelector('#full'),gap=document.querySelector('#gap'),filter=document.querySelector('#filter');function apply(){let n=0;document.querySelectorAll('.case').forEach(s=>{s.hidden=filter.value!=='all'&&s.dataset.group!==filter.value;if(!s.hidden)n++});document.querySelector('#count').textContent=n+' cases'}full.onclick=()=>{document.body.classList.remove('gap');full.classList.add('active');gap.classList.remove('active')};gap.onclick=()=>{document.body.classList.add('gap');gap.classList.add('active');full.classList.remove('active')};filter.onchange=apply;apply();
</script></html>'''
    meta=f'STEP {summary["checkpoint_step"]:,} · SEED {summary["seed"]} · {summary["cases"]} CASES / {summary["images"]} IMAGES · {summary["sampling"]["resolution"]}px · {summary["active_updates"]} LOOP UPDATES / {summary["blocks"]} BLOCKS'
    labels=dict(controlled_long='可核验长 Prompt',heldout_long='v1 留出长 Prompt',short_control='短 Prompt 对照',geneval='GenEval 原始 Prompt')
    options='<option value="all">全部</option>'+''.join(f'<option value="{esc(g)}">{esc(labels.get(g,g))}</option>' for g in summary['groups'])
    import re
    template=re.sub(r'<select id="filter">.*?</select>',lambda _: '<select id="filter">'+options+'</select>',template)
    if not any(r.get('removed_clauses') for r in records):
        template=template.replace('先看语义，<span>再看变化。</span>','原生 BAGEL，<span>对比 Stage A。</span>')
        template=template.replace('完整组比较普通生图；信息差组检查 UND 能否把 GEN 没看见的句子带回图中。所有组使用同一初始噪声种子。点击图片查看原尺寸。','两组使用相同的原始 Prompt、种子和采样设置。按原始评测标签检查成图；点击图片查看原尺寸。')
        template=template.replace('id="gap" type="button"','id="gap" type="button" hidden')
        template=re.sub(r'<footer>.*?</footer>','<footer>原生 BAGEL 与训练后 Loop 的配对生图。正式评分及覆盖率见 benchmark_summary.json；小样本检查不是完整榜单分数。</footer>',template)
    page=template.replace('__META__',esc(meta)).replace('__CHECKPOINT__',esc(summary['checkpoint'])).replace('__SECTIONS__',''.join(sections))
    (output/'index.html').write_text(page)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    render(parser.parse_args().output)
