"""Render local manual review of Memory captions; keep group mapping revealable."""
import argparse
import csv
import html
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    report = json.loads((root/'source_review.json').read_text())
    groups = {}
    for row in report['records']:
        groups.setdefault(row['key'], {})[row['arm']] = row
    sections = []; mapping = []; annotations = []
    rng = random.Random(20261013)
    labels = {'Correct':'正确来源', 'Source-off':'关闭来源', 'Source-swap':'交换来源'}
    for number, (key, group) in enumerate(sorted(groups.items()), 1):
        if set(group) != set(labels):
            raise ValueError('each state must contain exactly three source arms')
        row = group['Correct']
        if any(item['state'] != row['state'] for item in group.values()):
            raise ValueError('recipient state differs across caption arms')
        arms = list(labels); rng.shuffle(arms)
        pictures = []
        for role, title in [('recipient','本图'), ('donor','交换图')]:
            asset = row['images'][role]
            for field, detail in [('image','原图，按训练尺寸缩放'), ('preview','当前状态的原生预测预览')]:
                path = asset[field]
                if not (root/path).is_file():
                    raise ValueError('missing review image: '+path)
                pictures.append(f'<figure><figcaption>{title} · {detail}</figcaption><a href="{html.escape(path)}"><img src="{html.escape(path)}" loading="lazy"></a></figure>')
        captions = []
        for index, arm in enumerate(arms):
            alias = chr(65+index)
            text = group[arm]['caption']['text']
            mapping.append(dict(key=key, column=alias, arm=arm))
            captions.append(f'<article class="caption"><h3>读出 {alias}<span class="arm"> · {labels[arm]}</span></h3><p lang="en">{html.escape(text)}</p><label>与图片的关系<select data-column="{alias}"><option value="">未标注</option><option>对应本图</option><option>对应交换图</option><option>泛化描述</option><option>明显错误</option><option>不确定</option></select></label></article>')
        sections.append(f'''<section data-key="{html.escape(key)}"><div class="state-head"><h2>{number:02d} · t={row['t']}</h2><span>{html.escape(row['sample_id'])} · seed={row['noise_seed']}</span></div>
<div class="pictures">{''.join(pictures)}</div>
<details><summary>查看来源数据中的配对描述</summary><p><b>本图：</b>{html.escape(row['target_caption'])}</p><p><b>交换图：</b>{html.escape(row['donor_caption'])}</p><p class="muted">配对依据是数据集一致性标注；请以实际图片为准。</p></details>
<div class="captions">{''.join(captions)}</div>
<div class="annotation"><label>来源依赖判断<select data-column="overall"><option value="">未标注</option><option>读出随来源图像变化，内容对应</option><option>读出变化，但内容不对应</option><option>三组基本相同</option><option>证据不足</option></select></label><label>审查备注<textarea data-column="notes" rows="2" placeholder="对象、属性、数量或关系的具体差异"></textarea></label></div></section>''')
        annotations.append(dict(key=key, sample_id=row['sample_id'], donor_id=row['donor_id'],
                                t=row['t'], noise_seed=row['noise_seed'], A='',B='',C='',overall='',notes=''))
    (root/'review_mapping.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n')
    with (root/'manual_annotations.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(annotations[0]));writer.writeheader();writer.writerows(annotations)
    payload=json.dumps(dict(checkpoint=report['checkpoint'],checkpoint_sha256=report['checkpoint_sha256'],
                           mapping=mapping,annotations=annotations),ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Memory 来源对照审查</title>
<style>
:root{color-scheme:light;font-family:system-ui,-apple-system,sans-serif;color:#20282c;background:#f1f3f3}*{box-sizing:border-box}body{margin:0}main{max-width:1320px;margin:auto;padding:28px 24px 64px}header{padding:0 0 22px}h1{font-size:30px;margin:10px 0}h2{font-size:21px;margin:0}h3{font-size:17px;margin:0 0 12px}p{line-height:1.65;margin:8px 0}header p{max-width:1000px}.muted{color:#617079;font-size:14px}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:18px 0}button{border:1px solid #a6b4bb;border-radius:6px;padding:10px 14px;font:inherit;background:white;cursor:pointer}button.primary{background:#204d4a;color:white;border-color:#204d4a}.status{font-size:14px;color:#52646b}section{background:white;border:1px solid #d9e0e2;border-radius:10px;margin:18px 0 30px;padding:20px}.state-head{display:flex;justify-content:space-between;gap:16px;align-items:baseline;flex-wrap:wrap;margin-bottom:18px}.state-head span{color:#617079;font-size:13px}.pictures{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}figure{margin:0}figcaption{font-size:13px;margin-bottom:7px;color:#46585d}img{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#f1f3f3;border-radius:4px}details{background:#f5f7f7;padding:12px;margin:16px 0;font-size:14px}summary{cursor:pointer}.captions{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.caption{border:1px solid #dce3e4;border-top:3px solid #829a9b;border-radius:5px;padding:16px;display:flex;flex-direction:column}.caption p{margin:0 0 18px;flex:1;font-size:17px;line-height:1.6}.arm{display:none;color:#285a54;font-weight:500;font-size:14px}.revealed .arm{display:inline}label{display:block;font-size:13px;color:#4c6066}select,textarea{display:block;width:100%;margin-top:7px;padding:9px;border:1px solid #bccacc;border-radius:4px;font:inherit;color:#26383e;background:white}textarea{resize:vertical}.annotation{display:grid;grid-template-columns:1fr 2fr;gap:14px;margin-top:18px;border-top:1px solid #e1e7e7;padding-top:16px}.top{position:fixed;bottom:18px;right:18px;text-decoration:none;color:#244d49;background:#fff;padding:9px 13px;border:1px solid #bdcdcb;border-radius:5px}@media(max-width:800px){main{padding:18px 12px}.pictures{grid-template-columns:repeat(2,1fr)}.captions,.annotation{grid-template-columns:1fr}section{padding:14px}}
</style><main id="top"><header><p class="muted">C2-observe · 最佳 checkpoint：1000步</p><h1>Memory 是否读取了图像内容</h1><p>4张固定验证图，t=0.25 / 0.75 / 0.9，噪声种子0。三组只改变 Memory 的视觉来源，使用相同中性 prompt 和固定描述指令。caption 由冻结 UND 只读取 Memory 后自由生成。</p><p class="muted">请先比较图片与三列读出，再揭示组别。预测预览是原生 GEN 的 x₀估计，只作当前带噪状态的观察代理。本页没有将 Memory 注入 GEN，不用于判断生成图像质量。</p><div class="toolbar"><button id="reveal">揭示组别</button><button class="primary" id="export">导出审查记录</button><a href="manual_annotations.csv">下载空白 CSV</a><span class="status" id="status">标注保存在当前浏览器</span></div><details><summary>运行信息</summary><p id="checkpoint"></p><p class="muted" id="hash"></p></details></header>'''+''.join(sections)+'''</main><a class="top" href="#top">回到顶部</a><script id="payload" type="application/json">'''+payload+'''</script><script>
const meta=JSON.parse(document.getElementById('payload').textContent);const storageKey='memory-source-review:'+meta.checkpoint_sha256;let saved={};try{saved=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){}document.getElementById('checkpoint').textContent=meta.checkpoint;document.getElementById('hash').textContent='SHA256: '+meta.checkpoint_sha256;document.querySelectorAll('section[data-key]').forEach(section=>{const key=section.dataset.key;section.querySelectorAll('[data-column]').forEach(input=>{input.value=saved[key]?.[input.dataset.column]||'';input.addEventListener('input',()=>{saved[key]??={};saved[key][input.dataset.column]=input.value;try{localStorage.setItem(storageKey,JSON.stringify(saved));document.getElementById('status').textContent='已保存到当前浏览器'}catch(e){document.getElementById('status').textContent='浏览器未保存，请导出记录'}})})});document.getElementById('reveal').onclick=()=>{document.body.classList.toggle('revealed');document.getElementById('reveal').textContent=document.body.classList.contains('revealed')?'隐藏组别':'揭示组别'};document.getElementById('export').onclick=()=>{const result={checkpoint:meta.checkpoint,checkpoint_sha256:meta.checkpoint_sha256,reviewed_at:new Date().toISOString(),mapping:meta.mapping,annotations:meta.annotations.map(row=>({...row,...saved[row.key]}))};const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='memory_source_manual_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script></html>'''
    (root/'index.html').write_text(page)
    print(root/'index.html')


if __name__=='__main__':main()
