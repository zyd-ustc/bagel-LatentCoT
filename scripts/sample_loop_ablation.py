"""Paired Stage A ablations on eight original prompts per benchmark."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import html
import json
import os
from pathlib import Path
import random
import time
import traceback

from sample_feedback_sweep import read_rows, select_hard

ARMS = {
    'base4': dict(label='原版 BAGEL · CFG 4', native=True, loop_enabled=False, cfg=4., updates=0),
    'parent4': dict(label='Parent UND · 无 Loop · CFG 4', native=False, loop_enabled=False, cfg=4., updates=0),
    'loop1': dict(label='正常 1 次反馈 · CFG 4', native=False, loop_enabled=True, cfg=4., updates=1),
    'masked1': dict(label='屏蔽 Memory · 1 次 · CFG 4', native=False, loop_enabled=True, cfg=4., updates=1, memory_masked=True),
    'noentry1': dict(label='关闭入口回灌 · 1 次 · CFG 4', native=False, loop_enabled=True, cfg=4., updates=1, reentry=False),
    'parent1': dict(label='Parent UND · 无 Loop · CFG 1', native=False, loop_enabled=False, cfg=1., updates=0),
    'loop1cfg1': dict(label='正常 1 次反馈 · CFG 1', native=False, loop_enabled=True, cfg=1., updates=1),
    'loop2': dict(label='正常 2 次反馈 · CFG 4', native=False, loop_enabled=True, cfg=4., updates=2),
}


def csv_cases(path, benchmark):
    grouped = defaultdict(list)
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f): grouped[row['item_id']].append(row)
    return [dict(prompt_id=f'{benchmark}_{key}', benchmark=benchmark, prompt=rows[0]['text'],
                 vqa_list=[[r['question_natural_language'], 'yes'] for r in rows],
                 annotations=rows, source_path=str(path)) for key,rows in sorted(grouped.items())]


def prepare(args):
    from qwen_latent_cot.diagnostics.inference import resolve_checkpoint
    hard, selection = select_hard(read_rows('data/geneval2_800.jsonl'),8,17)
    cases = [dict(r, benchmark='geneval2') for r in hard]
    tools = Path(args.tools)
    source = tools/'geneval/src/geneval/prompts/evaluation_metadata.jsonl'
    rows = read_rows(source)
    # Cover all six categories; add one extra counting and one extra position case.
    rng = random.Random(17)
    tags = ('single_object','two_object','counting','colors','position','color_attr','counting','position')
    chosen = set()
    for tag in tags:
        i = rng.choice([i for i,r in enumerate(rows) if r['tag']==tag and i not in chosen]); chosen.add(i)
        r = rows[i]
        cases.append(dict(prompt_id=f'geneval_{i:05d}',benchmark='geneval',prompt=r['prompt'],
                          annotations=r,category=tag,source_path=str(source),vqa_list=[]))
    counts = dict(geneval2=dict(source=selection['source_prompts'],pool=selection['pool_size'],selected=8,atom_count=10),
                  geneval=dict(source=len(rows),selected=8,categories=dict(Counter(tags))))
    for bench, path in (
        ('dpg',tools/'dpg/src/dpg_bench/dpg_bench.csv'),
        ('dsg',tools/'dsg/src/dsg/data/dsg-1k-anns.csv')):
        pool = csv_cases(path,bench)
        picked = random.Random(17).sample(pool,8)
        cases.extend(picked); counts[bench] = dict(source=len(pool),selected=8)
    out=Path(args.output)
    if out.exists(): raise FileExistsError('Use a fresh output directory')
    out.mkdir(parents=True)
    plan=dict(checkpoint=str(resolve_checkpoint(args.checkpoint)),seed=17,selection_seed=17,
              arms=ARMS,benchmarks=counts,cases=len(cases),expected_images=len(cases)*len(ARMS),
              evaluation='paired diagnostic images and conditional velocity measurements; no official benchmark scores',
              controls='all arms keep full original prompt, seed, 512px, 50 nodes, shift 3; first 25 updates/all 28 blocks',
              caveats=['Memory masking retains the +16 query position shift and consumer adapters.',
                       'CFG effect compares parent4→loop1 against parent1→loop1cfg1, not loop1 against loop1cfg1 alone.',
                       'Closing reentry keeps Memory and consumer adapters. No fine-tuning or gate amplification.'])
    (out/'prompts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases))
    write(out/'plan.json',plan)
    print(json.dumps(plan,ensure_ascii=False,indent=2))


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


@contextmanager
def conditional_probe(student):
    import torch
    import qwen_latent_cot.interaction.loop as loop
    original=loop.conditional_gen
    measurements=[]
    def call(runtime,state,bank=None,**kwargs):
        heads=[]
        def hook(module,args,value):
            if not heads: heads.append(value[state.flow['packed_vae_token_indexes']].detach().clone())
        handle=runtime.model.llm2vae.register_forward_hook(hook)
        try: result=original(runtime,state,bank,**kwargs)
        finally: handle.remove()
        if bank is not None:
            base=kwargs['velocity_baseline'].detach().float()
            corrected=(base+kwargs['velocity_scale'].detach()*(heads[0].float()-base)).to(heads[0].dtype).float()
            delta=corrected-base
            rms=lambda v: float(v.square().mean().sqrt())
            measurements.append(dict(t=float(state.t),round=kwargs['round_index'],
                native_rms=rms(base),conditional_delta_rms=rms(delta),
                conditional_delta_ratio=rms(delta)/max(rms(base),1e-8),
                conditional_cosine=float(torch.nn.functional.cosine_similarity(base.flatten(),corrected.flatten(),dim=0)),
                final_velocity_rms=rms(result.velocity.float())))
        return result
    loop.conditional_gen=call
    try: yield measurements
    finally: loop.conditional_gen=original


def worker(args):
    from qwen_latent_cot.bagel.accelerator import set_device,synchronize,reset_peak_memory_stats,max_memory_allocated
    from qwen_latent_cot.diagnostics.inference import InferenceEngine
    from qwen_latent_cot.interaction.sampling import sample
    out=Path(args.output);plan=json.loads((out/'plan.json').read_text());rows=read_rows(out/'prompts.jsonl')
    status=out/f'worker_{args.rank:02d}.json'
    meta=dict(rank=args.rank,world=args.world,pid=os.getpid(),checkpoint=plan['checkpoint'],seed=plan['seed'])
    try:
        write(status,dict(meta,status='loading'))
        device=set_device('npu:0');engine=InferenceEngine.load(args.model,plan['checkpoint'],device)
        student=engine.student;runtime=engine.runtime
        meta.update(bridge_strength=float(student.bridge.strength),velocity_gates=student.bridge.vel_scale.detach().float().cpu().tolist())
        write(status,dict(meta,status='sampling'))
        for i in range(args.rank,len(rows),args.world):
            record=dict(rows[i],index=i,checkpoint=plan['checkpoint'],seed=plan['seed'],arms={})
            for arm,control in plan['arms'].items():
                filename=f'case_{i:03d}_{arm}.png'
                if (out/filename).exists(): raise FileExistsError(out/filename)
                old_cfg=runtime.config['cfg_text_scale'];runtime.config['cfg_text_scale']=control['cfg']
                reset_peak_memory_stats(device);began=time.perf_counter()
                try:
                    with conditional_probe(student) as measurements:
                        image,nodes=sample(student,record['prompt'],plan['seed'],native=control['native'],
                            loop_enabled=control['loop_enabled'],feedback_updates=control['updates'] if control['loop_enabled'] else None,
                            memory_masked=control.get('memory_masked',False),reentry=control.get('reentry',True))
                finally: runtime.config['cfg_text_scale']=old_cfg
                synchronize(device);image.save(out/filename)
                write(out/f'case_{i:03d}_{arm}.json',dict(nodes=nodes,measurements=measurements,control=control))
                record['arms'][arm]=dict(image=filename,seconds=time.perf_counter()-began,
                    peak_memory_gib=max_memory_allocated(device)/1024**3,control=control,
                    mean_conditional_delta_ratio=(sum(m['conditional_delta_ratio'] for m in measurements)/len(measurements) if measurements else 0.))
                write(out/f'case_{i:03d}.partial.json',record)
                print(f'ARM_COMPLETE case={i} arm={arm}',flush=True)
            write(out/f'case_{i:03d}.json',record);(out/f'case_{i:03d}.partial.json').unlink()
        write(status,dict(meta,status='complete'))
    except Exception:
        write(status,dict(meta,status='failed',error=traceback.format_exc()));raise


def render(output):
    from PIL import Image
    out=Path(output);plan=json.loads((out/'plan.json').read_text());rows=read_rows(out/'prompts.jsonl')
    esc=lambda x:html.escape(str(x),quote=True)
    sections=[];metrics=defaultdict(list)
    for i,row in enumerate(rows):
        r=json.loads((out/f'case_{i:03d}.json').read_text())
        if any(r[k]!=row[k] for k in ('prompt_id','prompt','benchmark')) or r['checkpoint']!=plan['checkpoint'] or r['seed']!=plan['seed']:raise ValueError('Mixed case identity')
        if set(r['arms'])!=set(plan['arms']): raise ValueError('Missing arms')
        figures=[]
        for arm,control in plan['arms'].items():
            a=r['arms'][arm]
            with Image.open(out/a['image']) as im: im.verify()
            trace=json.loads((out/f'case_{i:03d}_{arm}.json').read_text())
            if trace['control']!=control or a['control']!=control:raise ValueError('Control mismatch')
            if len(trace['nodes'])!=49:raise ValueError('Sampler node count changed')
            for j,n in enumerate(trace['nodes']):
                active=control['loop_enabled'] and j<25
                if n['loop_active']!=active or n['passes']!=(1+control['updates'] if active else 1): raise ValueError('Incorrect execution counts')
            if len(trace['measurements'])!=25*control['updates']:raise ValueError('Missing conditional measurements')
            metrics[(row['benchmark'],arm)].append(a['mean_conditional_delta_ratio'])
            figures.append(f'<figure data-arm="{arm}"><a href="{a["image"]}" target="_blank"><img loading="lazy" src="{a["image"]}" width="512" height="512" alt="{esc(row["prompt_id"]+" "+arm)}"></a><figcaption>{esc(control["label"])}<small>{a["seconds"]:.1f}s · Δv/native RMS {a["mean_conditional_delta_ratio"]:.3f}</small></figcaption></figure>')
        questions=''.join(f'<li>{esc(q)} <b>{esc(a)}</b></li>' for q,a in row['vqa_list'])
        annotations=esc(json.dumps(row.get('annotations',{}),ensure_ascii=False,indent=2))
        sections.append(f'<section data-bench="{row["benchmark"]}"><p class="eyebrow">{i+1:02d} / {esc(row["benchmark"])} / {esc(row["prompt_id"])}</p><h2>{esc(row["prompt"])}</h2><div class="grid">{"".join(figures)}</div><details><summary>原始核验问题与标注</summary><ol>{questions}</ol><pre>{annotations}</pre></details></section>')
    workers=[json.loads(f.read_text()) for f in out.glob('worker_*.json')]
    if len(workers)!=16 or any(w['status']!='complete' for w in workers):raise ValueError('Workers incomplete')
    summary=dict(plan,completed_cases=len(rows),completed_images=len(rows)*len(plan['arms']),workers_complete=len(workers),
        measurements=[dict(benchmark=b,arm=a,mean_conditional_delta_ratio=sum(v)/len(v)) for (b,a),v in metrics.items()])
    options=''.join(f'<option value="{a}">{esc(c["label"])}</option>' for a,c in plan['arms'].items())
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Loop · 小规模消融</title><style>
:root{--paper:#f3f1eb;--ink:#183346;--accent:#af4e27}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.65 Georgia,"Songti SC",serif}main{max-width:1800px;margin:auto;padding:32px}h1{font-size:clamp(38px,5vw,68px);line-height:1.1;letter-spacing:-.04em;margin:24px 0}h1 span{color:var(--accent)}.mast{border-top:5px solid var(--ink);padding-bottom:22px}.eyebrow,small{font:12px/1.7 Menlo,monospace;color:#67737a}.lede{max-width:1050px}nav{position:sticky;top:0;background:var(--paper);border-block:1px solid var(--ink);padding:12px 0;display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:2}select,button{font:15px Georgia,"Songti SC",serif;color:var(--ink);background:transparent;padding:9px;border:1px solid #aab7bd}section{padding:30px 0;border-bottom:1px solid var(--ink)}h2{font-size:22px;max-width:1300px;line-height:1.45}.grid{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:14px;max-width:1250px;overflow-x:auto}.all .grid{grid-template-columns:repeat(8,minmax(280px,1fr));max-width:none}figure{margin:0;background:white}img{display:block;width:100%;height:auto}figcaption{padding:12px;font-size:15px;border-bottom:2px solid #ccd3d5}small{display:block}summary{cursor:pointer;color:var(--accent);padding-top:15px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 Menlo,monospace}li{margin:6px 0}a:focus-visible,select:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid var(--accent)}[hidden]{display:none!important}footer{padding:24px 0;font-size:14px;color:#67737a}@media(max-width:700px){main{padding:16px}.grid{grid-template-columns:repeat(2,minmax(260px,1fr))}}</style><main><header class="mast"><p class="eyebrow">BAGEL / STAGE A / STEP 4000 / SEED 17</p><h1>画风从哪一步<span>开始偏移？</span></h1><p class="lede">GenEval2 最大复杂度组、GenEval、DPG、DSG 各 8 条原始 Prompt，共 32 条、8 组、256 张图。固定 checkpoint 与初始噪声，前 25 次更新、全部 28 层参与循环。默认比较 Parent 与正常一次反馈，可独立选择两列或展开全部。</p><p class="lede">Memory 屏蔽组保留 consumer、入口回灌及 +16 位置偏移；关闭回灌组保留 Memory 和 consumer。判断 CFG 的作用时，请比较 CFG 4 与 CFG 1 各自的 Parent→Loop 差异。</p><p class="eyebrow">__CHECKPOINT__</p></header><nav><label>题库 <select id="bench"><option value="all">全部</option><option>geneval2</option><option>geneval</option><option>dpg</option><option>dsg</option></select></label><label>左列 <select id="left">__OPTIONS__</select></label><label>右列 <select id="right">__OPTIONS__</select></label><label><input type="checkbox" id="all"> 全部八组</label></nav>__SECTIONS__<footer>人工消融页，不包含官方 benchmark 分数。Δv/native RMS 是每个时间点、每轮条件速度差异与其 baseline RMS 之比的平均值，在 CFG 之前计算；不是语义准确率，也不是画质分数。</footer></main><script>
const $=s=>document.querySelector(s);$('#left').value='parent4';$('#right').value='loop1';function refresh(){let all=$('#all').checked;document.body.classList.toggle('all',all);document.querySelectorAll('section[data-bench]').forEach(s=>{s.hidden=$('#bench').value!=='all'&&s.dataset.bench!==$('#bench').value;const grid=s.querySelector('.grid');[...grid.children].forEach(f=>{f.hidden=!all&&f.dataset.arm!==$('#left').value&&f.dataset.arm!==$('#right').value;f.style.order=all?'':f.dataset.arm===$('#left').value?'0':'1'})});}document.querySelectorAll('nav input,nav select').forEach(e=>e.onchange=refresh);refresh();</script></html>'''
    page=page.replace('__CHECKPOINT__',esc(plan['checkpoint'])).replace('__OPTIONS__',options).replace('__SECTIONS__',''.join(sections))
    (out/'index.html').write_text(page);write(out/'summary.json',summary)
    print(json.dumps(dict(cases=len(rows),images=summary['completed_images'],workers=16)))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','worker','render'));p.add_argument('--output',required=True)
    p.add_argument('--checkpoint');p.add_argument('--tools',default='/root/npu-eval-tools');p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=16)
    p.add_argument('--model',default='/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT');args=p.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command=='worker':worker(args)
    else:render(args.output)

if __name__=='__main__': main()
