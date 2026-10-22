"""Render fixed-order native/global/corrected previews and six score summaries."""
import argparse
import html
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True,help='label=/local/run/root');p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=32);a=p.parse_args()
    roots={label:Path(path) for label,path in (s.split('=',1) for s in a.runs)}
    a.output.mkdir(parents=True,exist_ok=True);e=html.escape
    body=['<h1>U1.5 · 同题生成与纠错</h1><p>本地27B内部评分。每个arm单独显示完整分母；图像按manifest固定顺序。</p>']
    score_rows={};records={}
    for label,root in roots.items():
        plan=json.loads((root/'PLAN.json').read_text());rows=[json.loads(x) for x in (root/'manifests/combined.jsonl').read_text().splitlines() if x.strip()]
        records[label]={(x['benchmark'],str(x['record_id'])):x for x in rows};score_root=root/'scoring_qwen27b_v1/scores'/plan['arm']
        for bench,expected in plan['counts'].items():
            summary=score_root/bench/'summary.json';value=json.loads(summary.read_text()) if summary.exists() else {}
            result=score_root/bench/('result_scores.jsonl' if bench=='wise_verified' else 'results.jsonl')
            key='index' if bench.endswith('_pp') else 'prompt_id' if bench=='wise_verified' else 'record_id'
            raw=[json.loads(x) for x in result.read_text().splitlines() if x.strip()] if result.exists() else []
            latest={str(x[key]):x for x in raw}
            def valid(x):
                if bench=='wise_verified':return type(x.get('score')) in (int,float) and x['score'] in (0,1) and not x.get('error') and x.get('status') in (None,'ok','scored')
                return x.get('status')==('ok' if bench.endswith('_pp') else 'scored')
            successful=sum(valid(x) for x in latest.values());complete=successful==expected and len(latest)==expected
            score_rows.setdefault(bench,[]).append(f'<td><b>{e(label)}</b><br>{successful}/{expected} · {"完整" if complete else "未完成"}<details><summary>原评分明细</summary><pre>{e(json.dumps(value,ensure_ascii=False,indent=2))}</pre></details></td>')
    body.append('<table>'+''.join('<tr><th>'+e(b)+'</th>'+''.join(cells)+'</tr>' for b,cells in score_rows.items())+'</table>')
    first=next(iter(records.values()))
    for record_id,row in list(first.items())[:a.limit]:
        body.append('<h2>'+e(row['instruction'])+'</h2><div class="row">')
        for label,root in roots.items():
            other=records[label][record_id]
            if other['instruction']!=row['instruction'] or other['seed']!=row['seed']:raise ValueError('Comparison prompt/seed differs')
            image=root/'images'/other['output_rel'];target=a.output/'assets'/label/other['output_rel'];target.parent.mkdir(parents=True,exist_ok=True)
            if image.exists():
                shutil.copyfile(image,target);body.append(f'<figure><img src="{e(str(target.relative_to(a.output)))}"><figcaption>{e(label)} · seed {other["seed"]}</figcaption></figure>')
            else:body.append(f'<figure>{e(label)}：图片待生成</figure>')
        body.append('</div>')
    css='body{margin:32px;font:16px/1.6 sans-serif;background:#f4f4ed;color:#202822}h1{font:36px Georgia}h2{font-size:17px;margin-top:36px}.row{display:flex;gap:20px}figure{margin:0;flex:1}img{width:100%}table{width:100%;border-collapse:collapse}td,th{padding:14px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap;font-size:12px}'
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>U1.5 comparison</title><style>'+css+'</style>'+''.join(body))
    print(a.output/'index.html')


if __name__=='__main__':main()
