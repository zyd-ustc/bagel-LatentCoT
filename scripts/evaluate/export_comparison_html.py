#!/usr/bin/env python
"""Portable gallery from existing scored images; never performs inference."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path


def export(root,output):
    rows=[json.loads(x) for x in (root/'quality_report/scores.jsonl').read_text().splitlines() if x.strip()]
    summary=json.loads((root/'quality_report/summary.json').read_text())
    arms=list(summary['arms'])
    arms=['BASE']+[a for a in arms if a!='BASE']
    groups={}
    for row in rows:
        group=groups.setdefault((row['prompt_id'],row['seed']),{})
        if row['arm'] in group:raise ValueError('duplicate scored image')
        group[row['arm']]=row
    esc=lambda value:html.escape(str(value),quote=True)
    parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BAGEL · 配对图片</title><style>
    :root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#141819;color:#ecf0ea;font:16px/1.5 system-ui,-apple-system,sans-serif}main{max-width:1720px;margin:auto;padding:28px 20px}h1{font-size:30px}h2{font-size:18px;font-weight:500}p,small,details{color:#b8c4bc}.meta{font:12px ui-monospace,monospace;overflow-wrap:anywhere}.grid{display:grid;gap:12px}.label{font:13px ui-monospace,monospace;color:#d4e6b4;padding:8px 0;overflow-wrap:anywhere}img{width:100%;height:auto;display:block;cursor:zoom-in}.pair{border-top:1px solid #435047;margin:30px 0;padding:14px 0}table{border-collapse:collapse;margin:20px 0;font-size:14px}th,td{padding:8px 14px;text-align:left;border-bottom:1px solid #435047}li{margin:8px 0}details{margin-top:12px}dialog{background:#141819;border:1px solid #687864;padding:10px;max-width:98vw;max-height:98vh}dialog img{width:auto;max-width:92vw;max-height:84vh;cursor:default}button{margin-bottom:8px}@media(max-width:850px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))!important}main{padding:20px 10px}}
    </style><main><h1>BAGEL：循环深度配对比较</h1><p>同 prompt、同 noise seed。图片嵌入文件，可离线查看。点击图片查看原尺寸。评分是模型代理；Repair / Damage 未经人工确认。生成时间为工程日志。</p>''']
    parts.append(f'<p class="meta">RUN {esc(root.name)}<br>SOURCE {esc(summary["source_sha256"])}</p>')
    parts.append('<table><tr><th>Arm</th><th>Semantic GM</th><th>Quality proxy</th><th>Repair / Damage vs Base</th><th>Median seconds</th></tr>')
    for arm in arms:
        value=summary['arms'][arm];delta=value['vs_BASE']
        parts.append(f'<tr><td>{esc(arm)}</td><td>{value["semantic_gm"]:.4f}</td><td>{value["quality_proxy"]:.4f}</td><td>{delta["repair_count"]} / {delta["damage_count"]}</td><td>{value["latency_median_seconds"]:.2f}</td></tr>')
    parts.append('</table>')
    for group in sorted(groups.values(),key=lambda g:(g['BASE']['index'],g['BASE']['seed'])):
        if set(group)!=set(arms):raise ValueError('incomplete arm coverage in HTML')
        base=group['BASE']
        parts.append(f'<section class="pair"><small>#{base["index"]:02d} · seed {base["seed"]}</small><h2>{esc(base["prompt"])}</h2><div class="grid" style="grid-template-columns:repeat({len(arms)},minmax(0,1fr))">')
        for arm in arms:
            row=group[arm]
            if any(row[k]!=base[k] for k in ('prompt','noise_sha256','height','width')):
                raise ValueError('paired inputs differ')
            parts.append(f'<div><div class="label">{esc(arm)} · quality {row["quality_proxy"]:.2f}</div>')
            if row['valid_file']:
                raw=Path(row['path']).read_bytes()
                if hashlib.sha256(raw).hexdigest()!=row['image_sha256']:raise ValueError('image hash changed')
                encoded=base64.b64encode(raw).decode('ascii')
                parts.append(f'<img loading="lazy" src="data:image/png;base64,{encoded}" alt="{esc(arm)}" onclick="showImage(this)">')
            else:parts.append('<p>Invalid generated image</p>')
            if arm!='BASE':
                changes=[];questions=row.get('semantic_questions') or []
                for i,(a,b) in enumerate(zip(base['semantic_atoms'],row['semantic_atoms'])):
                    if (a>=.5)==(b>=.5):continue
                    question=questions[i] if i<len(questions) else f'atom {i}'
                    if not isinstance(question,str):question=json.dumps(question,ensure_ascii=False)
                    kind='Repair' if b>=.5 else 'Damage'
                    changes.append(f'<li>{kind} · {esc(question)} · {a:.4f} → {b:.4f}</li>')
                parts.append('<details><summary>vs Base 约束评分变化</summary><ul>'+(''.join(changes) or '<li>没有阈值翻转。</li>')+'</ul></details>')
            parts.append('</div>')
        parts.append('</div></section>')
    parts.append('''</main><dialog id="viewer"><button onclick="document.getElementById('viewer').close()">关闭</button><img id="large"></dialog><script>function showImage(source){document.getElementById('large').src=source.src;document.getElementById('viewer').showModal()}document.getElementById('viewer').addEventListener('click',function(e){if(e.target===this)this.close()})</script></html>''')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(''.join(parts))
    return {'html':str(output),'prompt_seed_groups':len(groups),'images':len(rows),'bytes':output.stat().st_size}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True);parser.add_argument('--output')
    args=parser.parse_args();root=Path(args.run_dir).resolve()
    print(json.dumps(export(root,Path(args.output).resolve() if args.output else root/'comparison.html')))


if __name__=='__main__':main()
