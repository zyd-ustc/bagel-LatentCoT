"""Build a readable local gallery from downloaded native/one-round/two-round records."""
import argparse
from html import escape
import json
from pathlib import Path
import os

p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--step',type=int,required=True);a=p.parse_args()
root=Path(a.run).resolve();directory=root/f'semantic_{a.step:06d}';records=json.loads((directory/'records.json').read_text())
rows=[]
for row in records:
 images=[]
 for arm,label in [('native','BAGEL 原生'),('one_round','UND反馈 · 一轮编辑'),('two_round','UND反馈 · 两轮编辑')]:
  path=root/row['arms'][arm]
  if not path.is_file():raise FileNotFoundError(path)
  relative=os.path.relpath(path,directory)
  images.append(f'<figure><a href="{escape(relative)}"><img loading="lazy" src="{escape(relative)}"></a><figcaption>{label}</figcaption></figure>')
 rows.append(f'<section><h2>{row["index"]+1:02d} · {escape(row.get("prompt_id",""))}</h2><p>{escape(row["prompt"])}</p><div class="images">'+''.join(images)+'</div></section>')
html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>MoT 内部编辑对照</title>
<style>body{margin:0;background:#f5f3ee;color:#262a28;font:16px/1.6 system-ui}main{max-width:1500px;margin:auto;padding:36px 24px}header{padding-bottom:28px;border-bottom:2px solid #66796b}h1{font-size:32px;margin:0}h2{font-size:19px}p{max-width:1100px}section{padding:26px 0;border-bottom:1px solid #d4d7cf}.images{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}figure{margin:0}img{width:100%;display:block;border-radius:4px}figcaption{padding:10px 0;color:#46564a}@media(max-width:700px){.images{grid-template-columns:1fr}main{padding:20px}}</style><main><header><h1>MoT · 单次文生图内的理解与编辑</h1>'''
html+=f'<p>Step {a.step} · {len(records)} 条固定 prompt/seed · 原生 / 一轮 / 两轮。图像用于逐题查看语义与画风；loss 不表示语义成功率。</p></header>'+''.join(rows)+'</main></html>'
(directory/'index.html').write_text(html);print(directory/'index.html')
