"""GenEval2 high-complexity prompts: base versus 1/2/3/4 feedback updates."""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import random
import time
import traceback

ARMS=('base','loop1','loop2','loop3','loop4')
LABELS=dict(base='Base · 原生 BAGEL',loop1='1 次反馈',loop2='2 次反馈 · 训练轮数',loop3='3 次反馈 · 外推',loop4='4 次反馈 · 外推')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def select_hard(rows,count=32,seed=17):
    maximum=max(r['atom_count'] for r in rows)
    pool=[r for r in rows if r['atom_count']==maximum]
    if not 1<=count<=len(pool):raise ValueError('Requested count exceeds maximum-atom pool')
    chosen=random.Random(seed).sample(pool,count)
    chosen.sort(key=lambda r:r['source_index'])
    return chosen,dict(source_prompts=len(rows),pool_size=len(pool),atom_count=maximum,
                       selected=count,selection_seed=seed,
                       rule='fixed-seed sample from the maximum atom_count bin; original prompt/questions unchanged; not selected by model performance')


def prepare(args):
    chosen,selection=select_hard(read_rows(args.source),args.count,args.selection_seed)
    out=Path(args.output)
    if out.exists():raise FileExistsError('Use a fresh output directory: '+str(out))
    from qwen_latent_cot.diagnostics.inference import resolve_checkpoint
    pinned=str(resolve_checkpoint(args.checkpoint))
    out.mkdir(parents=True)
    (out/'prompts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen))
    plan=dict(benchmark='GenEval2',checkpoint=pinned,seed=args.seed,arms=list(ARMS),selection=selection,
              expected_images=len(chosen)*5,gate_extension='extra feedback rounds reuse the last trained velocity gate',
              base='original BAGEL, original UND weights, no Loop or adapters',
              input='full original prompt in every arm; no clause dropout',evaluation='manual comparison; no automatic GenEval2 score')
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(plan,ensure_ascii=False,indent=2))


def worker(args):
    from qwen_latent_cot.bagel.accelerator import set_device,synchronize,reset_peak_memory_stats,max_memory_allocated
    from qwen_latent_cot.diagnostics.inference import InferenceEngine,load_runtime
    from qwen_latent_cot.interaction.checkpoint import read,STAGE_A_KIND
    from qwen_latent_cot.interaction.loop import active_updates
    from qwen_latent_cot.diagnostics.record import write_json
    out=Path(args.output);plan=json.loads((out/'plan.json').read_text());rows=read_rows(out/'prompts.jsonl')
    if not 0<=args.rank<args.world:raise ValueError('Invalid worker rank/world')
    assigned=list(range(args.rank,len(rows),args.world))
    status=out/f'worker_{args.rank:02d}.json'
    meta=dict(rank=args.rank,world=args.world,pid=os.getpid(),checkpoint=plan['checkpoint'],seed=plan['seed'],assigned=assigned)
    try:
        write_json(status,dict(meta,status='loading'))
        device=set_device('npu:0')
        saved=read(plan['checkpoint'])
        if saved.get('kind')!=STAGE_A_KIND:raise ValueError('Requires a trained Stage A v4 checkpoint')
        runtime=load_runtime(args.model,saved['parent'],device)
        engine=InferenceEngine(runtime,saved['parent'],saved['config'],saved=saved)
        meta.update(checkpoint_step=saved['step'],trained_feedback_updates=saved['config']['feedback_updates'],
                    trained_velocity_gates=engine.student.bridge.vel_scale.detach().float().cpu().tolist(),
                    bridge_strength=float(engine.student.bridge.strength),blocks=len(runtime.decoder.layers),
                    active_updates=active_updates(runtime.config),sampling={k:runtime.config[k] for k in ('resolution','sampler_steps','timestep_shift','cfg_text_scale','cfg_renorm_type','cfg_renorm_min')})
        del saved
        write_json(status,dict(meta,status='sampling'))
        for index in assigned:
            row=rows[index]
            record=dict(row,index=index,checkpoint=plan['checkpoint'],checkpoint_step=meta['checkpoint_step'],seed=plan['seed'],rank=args.rank,arms={})
            for arm in ARMS:
                filename=f'case_{index:03d}_{arm}.png'
                if (out/filename).exists():raise FileExistsError(out/filename)
                n=0 if arm=='base' else int(arm[-1])
                reset_peak_memory_stats(device);began=time.perf_counter()
                image,nodes=engine.generate(row['prompt'],plan['seed'],native=n==0,feedback_updates=n if n else None)
                synchronize(device);image.save(out/filename)
                write_json(out/f'case_{index:03d}_{arm}.json',nodes)
                record['arms'][arm]=dict(image=filename,feedback_updates=n,seconds=time.perf_counter()-began,
                      peak_memory_gib=max_memory_allocated(device)/1024**3,loop_updates=sum(x['loop_active'] for x in nodes),
                      velocity_gate_indices=[min(i,len(meta['trained_velocity_gates'])-1) for i in range(n)],
                      conditional_gen_layer_calls=sum(x['conditional_gen_layer_calls'] for x in nodes))
                write_json(out/f'case_{index:03d}.partial.json',record)
                print(f'ARM_COMPLETE case={index} arm={arm}',flush=True)
            write_json(out/f'case_{index:03d}.json',record);(out/f'case_{index:03d}.partial.json').unlink()
        write_json(status,dict(meta,status='complete'))
    except Exception:
        write_json(status,dict(meta,status='failed',error=traceback.format_exc()));raise


def render(output):
    from PIL import Image
    out=Path(output);plan=json.loads((out/'plan.json').read_text());prompts=read_rows(out/'prompts.jsonl')
    records=[]
    for i,row in enumerate(prompts):
        r=json.loads((out/f'case_{i:03d}.json').read_text())
        if r['prompt']!=row['prompt'] or r['prompt_id']!=row['prompt_id'] or set(r['arms'])!=set(ARMS):raise ValueError('Case identity/arms mismatch')
        if r['checkpoint']!=plan['checkpoint'] or r['seed']!=plan['seed']:raise ValueError('Mixed checkpoint/seed')
        for arm,a in r['arms'].items():
            with Image.open(out/a['image']) as im:im.verify()
            nodes=json.loads((out/f'case_{i:03d}_{arm}.json').read_text())
            n=0 if arm=='base' else int(arm[-1])
            for node in nodes:
                if node['passes']!=(1+n if node['loop_active'] else 1):raise ValueError('Incorrect GEN pass count')
            if arm=='base' and any(node['loop_active'] for node in nodes):raise ValueError('Base contains Loop')
        records.append(r)
    workers=[json.loads(f.read_text()) for f in sorted(out.glob('worker_*.json'))]
    if not workers or any(w['status']!='complete' for w in workers):raise ValueError('Incomplete workers')
    summary=dict(plan,cases=len(records),images=len(records)*5,checkpoint_step=workers[0]['checkpoint_step'],
                 workers=len(workers),sampling=workers[0]['sampling'],active_updates=workers[0]['active_updates'],
                 blocks=workers[0]['blocks'],trained_velocity_gates=workers[0]['trained_velocity_gates'],
                 max_peak_memory_gib=max(a['peak_memory_gib'] for r in records for a in r['arms'].values()))
    esc=lambda x:html.escape(str(x),quote=True)
    sections=[]
    for i,r in enumerate(records):
        figures=''.join(f'<figure data-arm="{a}"><a href="{esc(r["arms"][a]["image"])}" target="_blank"><img width="512" height="512" loading="lazy" src="{esc(r["arms"][a]["image"])}" alt="{esc(r["prompt_id"]+" "+a)}"></a><figcaption><b>{esc(LABELS[a])}</b><span>{r["arms"][a]["seconds"]:.1f}s · {r["arms"][a]["peak_memory_gib"]:.1f} GiB</span></figcaption></figure>' for a in ARMS)
        questions=''.join(f'<li>{esc(q)} <strong>{esc(a)}</strong></li>' for q,a in r['vqa_list'])
        sections.append(f'<section><header><span class="number">{i+1:02d}</span><div><p class="eyebrow">{esc(r["prompt_id"])} / atom_count {r["atom_count"]}</p><h2>{esc(r["prompt"])}</h2></div></header><div class="grid">{figures}</div><details><summary>原始核验问题与期望答案 · {len(r["vqa_list"])} 项</summary><ol>{questions}</ol></details></section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GenEval2 · Feedback sweep</title><style>
:root{--bg:#f3f1eb;--ink:#183346;--muted:#67737a;--line:#ccd3d5;--accent:#af4e27}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 Georgia,"Songti SC",serif}main{max-width:1900px;margin:auto;padding:32px}.mast{border-top:5px solid var(--ink);padding:28px 0}h1{font-size:clamp(38px,5vw,72px);line-height:1.1;letter-spacing:-.04em;margin:12px 0 20px}h1 span{color:var(--accent)}.eyebrow,.meta{font:12px/1.7 Menlo,monospace;color:var(--muted);overflow-wrap:anywhere}.lede{max-width:1000px}.toolbar{position:sticky;top:0;z-index:2;background:var(--bg);border-block:1px solid var(--ink);padding:15px 0;display:flex;gap:20px;align-items:center;flex-wrap:wrap}select{font:16px Georgia,"Songti SC",serif;padding:9px 16px;color:var(--ink);border:1px solid var(--line);background:transparent}a:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid var(--accent);outline-offset:3px}section{padding:38px 0;border-bottom:1px solid var(--ink)}section header{display:flex;align-items:start;gap:20px;margin-bottom:20px}.number{font:36px/1.3 Menlo,monospace;color:var(--accent)}h2{font-size:22px;line-height:1.4;margin:4px 0;max-width:1300px}.grid{display:grid;grid-template-columns:repeat(5,minmax(280px,1fr));gap:14px;overflow-x:auto;padding-bottom:8px}body.pair .grid{grid-template-columns:repeat(2,minmax(280px,1fr));max-width:1240px}figure{margin:0;background:#fff}img{width:100%;height:auto;display:block}figcaption{padding:10px 12px;border-bottom:2px solid var(--line);font-size:14px}figcaption span{display:block;font:12px/1.6 Menlo,monospace;color:var(--muted)}details{padding-top:16px}summary{cursor:pointer;color:var(--accent)}ol{columns:2;column-gap:50px;padding-left:22px}li{break-inside:avoid;margin-bottom:8px}li strong{color:var(--accent);padding-left:12px}footer{padding:28px 0;color:var(--muted);font-size:14px}[hidden]{display:none!important}@media(max-width:700px){main{padding:18px}ol{columns:1}h2{font-size:20px}body.pair .grid{grid-template-columns:repeat(2,minmax(260px,1fr))}}
</style><main><div class="mast"><p class="eyebrow">BAGEL / STAGE A / GENEVAL2 / SAME SEED</p><h1>增加轮数，<span>语义更准吗？</span></h1><p class="lede">Base 是原版 BAGEL；1–4 指每个启用循环的时间点，在原生 GEN 后执行的反馈次数。所有组输入同一条完整 Prompt、同一初始噪声。3、4 次反馈复用最后一个已训练速度门控；不额外放大修正。</p><p class="meta">__META__</p><p class="meta">__CHECKPOINT__</p><p class="meta">__SELECTION__</p></div><nav class="toolbar"><label>比较 <select id="focus"><option value="all">全部五组</option><option value="loop1">Base vs 1 次反馈</option><option value="loop2">Base vs 2 次反馈</option><option value="loop3">Base vs 3 次反馈</option><option value="loop4">Base vs 4 次反馈</option></select></label><span class="meta">可横向滚动；点击图片查看原尺寸；展开原始问题逐项核验。</span></nav>__SECTIONS__<footer>本页用于人工比较，不包含自动 GenEval2 分数。额外反馈轮次未单独接受训练；成图变化不代表语义更准确。</footer></main><script>document.querySelector('#focus').onchange=e=>{const v=e.target.value;document.body.classList.toggle('pair',v!=='all');document.querySelectorAll('figure[data-arm]').forEach(f=>f.hidden=v!=='all'&&f.dataset.arm!=='base'&&f.dataset.arm!==v)};</script></html>'''
    meta=f'STEP {summary["checkpoint_step"]} · SEED {plan["seed"]} · {len(records)} CASES / {len(records)*5} IMAGES · {summary["sampling"]["resolution"]}px · FIRST {summary["active_updates"]} UPDATES / ALL {summary["blocks"]} BLOCKS'
    selection=plan['selection']
    selection_text=f'选样：原始 {selection["source_prompts"]} 条中，atom_count={selection["atom_count"]} 的最大复杂度组共 {selection["pool_size"]} 条；固定 seed {selection["selection_seed"]} 抽取 {selection["selected"]} 条。未按模型成绩挑选。'
    page=page.replace('__META__',esc(meta)).replace('__CHECKPOINT__',esc(plan['checkpoint'])).replace('__SELECTION__',esc(selection_text)).replace('__SECTIONS__',''.join(sections))
    (out/'index.html').write_text(page)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','worker','render'));p.add_argument('--output',required=True)
    p.add_argument('--source',default='data/geneval2_800.jsonl');p.add_argument('--checkpoint');p.add_argument('--count',type=int,default=32)
    p.add_argument('--selection-seed',type=int,default=17);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=16)
    p.add_argument('--model',default='/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT')
    args=p.parse_args()
    if args.command=='prepare':
        if not args.checkpoint:p.error('prepare requires --checkpoint')
        prepare(args)
    elif args.command=='worker':worker(args)
    else:render(args.output)


if __name__=='__main__':main()
