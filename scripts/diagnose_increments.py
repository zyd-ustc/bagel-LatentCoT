#!/usr/bin/env python
"""Prepare, run and report user-run fixed-state Memory feedback diagnostics."""
import argparse
import csv
import html
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qwen_latent_cot.evaluation.io import read_jsonl,sha256,source_hash


def config(path):
    c=json.loads(Path(path).read_text())
    if c['schema']!=1 or c['max_rounds']!=4 or c['trajectory']!='Early20_R4' or c['memory_update']!='full_depth_restart':
        raise ValueError('diagnostic requires restarted Early20 R4, with counterfactual R0..4')
    steps=c['steps']
    if not steps or steps!=sorted(set(steps)) or any(type(i) is not int or not 0<=i<20 for i in steps):
        raise ValueError('steps must be distinct increasing Early20 denoiser indexes')
    if c['max_prompts']<1 or not c['seeds'] or any(type(s) is not int for s in c['seeds']) or len(set(c['seeds']))!=len(c['seeds']):
        raise ValueError('require positive prompt count and distinct integer seeds')
    if any(type(i) is not int or not 0<=i<28 for i in c['tensor_layers']):raise ValueError('tensor dump layers outside native depth')
    return c


def prepare(a):
    c=config(a.config);data=read_jsonl(a.prompts)
    if len(data)<c['max_prompts']:raise ValueError('insufficient prompts for requested coverage')
    ids=[str(r.get('prompt_id',r.get('id',i))) for i,r in enumerate(data[:c['max_prompts']])]
    if len(set(ids))!=len(ids) or any(not r['prompt'].strip() for r in data[:c['max_prompts']]):raise ValueError('invalid prompts or duplicate IDs')
    model=Path(a.model_path).resolve()
    if json.loads((model/'llm_config.json').read_text())['num_hidden_layers']!=28:raise ValueError('require native 28-layer model')
    files=[model/n for n in ('ema.safetensors','ae.safetensors','llm_config.json','vit_config.json','tokenizer_config.json','vocab.json','merges.txt')]
    for f in files:
        if not f.is_file():raise ValueError('missing native model file: '+str(f))
    print('Binding source, prompts and native weights once...',flush=True)
    plan=dict(schema=1,source_sha256=source_hash(ROOT),config=c,config_sha256=sha256(a.config),
        model_path=str(model),model_sha256={f.name:sha256(f) for f in files},
        model_stat={f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in files},
        prompts=str(Path(a.prompts).resolve()),prompts_sha256=sha256(a.prompts),prompt_ids=ids,
        sampling=dict(image_size=512,num_timesteps=50,timestep_shift=3.,cfg_text_scale=4.,cfg_renorm_type='global'),
        semantic_gain_verified=False,quality_retention_verified=False,
        comparison_scope='R0..4 at the identical x_t/t sampled on Early20_R4; not independent image arms')
    path=Path(a.plan)
    if path.exists():raise ValueError('use a fresh diagnostic directory')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(plan,indent=2)+'\n')
    print(f"Prepared {len(ids)*len(c['seeds'])} trajectories, {len(c['steps'])} fixed-state probes each",flush=True)


def read_plan(path):
    p=json.loads(Path(path).read_text())
    if p['source_sha256']!=source_hash(ROOT) or p['prompts_sha256']!=sha256(p['prompts']):raise ValueError('source/prompts changed after binding')
    for name,stat in p['model_stat'].items():
        f=Path(p['model_path'])/name
        if [f.stat().st_size,f.stat().st_mtime_ns]!=stat:raise ValueError('model files changed')
    return p


def run(a):
    import torch
    from qwen_latent_cot.bagel.accelerator import set_device,device_info
    from qwen_latent_cot.bagel.backbone import load_native
    from qwen_latent_cot.bagel.internal_loop import LoopConfig,InternalLoopRuntime
    from qwen_latent_cot.bagel.inferencer import T2IGenerator
    from qwen_latent_cot.evaluation.increments import sample_with_diagnostics
    p=read_plan(a.plan);c=p['config']
    if not 0<=a.shard_index<a.num_shards:raise ValueError('invalid shard')
    device=set_device(a.device);output=Path(a.output_dir);output.mkdir(parents=True,exist_ok=True)
    if (output/'run.json').exists():raise ValueError('diagnostics do not append or silently resume')
    (output/'run.json').write_text(json.dumps(dict(plan_sha256=sha256(a.plan),accelerator=device_info(device),
        source_sha256=p['source_sha256'],shard_index=a.shard_index,num_shards=a.num_shards),indent=2)+'\n')
    bundle=load_native(p['model_path'],str(device),3.)
    runtime=InternalLoopRuntime(bundle.model,LoopConfig(extra_rounds=4,start_layer=0,end_layer=8,
        memory_update='full_depth_restart',progress_start=0.,progress_end=19/48))
    generator=T2IGenerator(bundle,runtime);data=read_jsonl(p['prompts'])[:len(p['prompt_ids'])]
    jobs=[(i,s) for i in range(len(data)) for s in c['seeds']]
    completed=[]
    try:
        for ordinal,(i,seed) in enumerate(jobs):
            if ordinal%a.num_shards!=a.shard_index:continue
            directory=output/f'prompt_{i:05d}_s{seed}';directory.mkdir()
            row=data[i];shape=(int(row.get('height',512)),int(row.get('width',512)))
            print(f"Start prompt={p['prompt_ids'][i]} seed={seed} shape={shape}",flush=True)
            result=sample_with_diagnostics(generator,directory,row['prompt'],shape,seed,c['steps'],c['max_rounds'],
                c['tensor_layers'],c['save_previews'])
            result.update(prompt_id=p['prompt_ids'][i],index=i,seed=seed,prompt=row['prompt'],shape=list(shape),path=str(directory))
            (directory/'trajectory.json').write_text(json.dumps(result,indent=2)+'\n')
            completed.append(result)
            print(f"Completed prompt={p['prompt_ids'][i]} seed={seed}",flush=True)
    finally:runtime.close()
    (output/'completed.json').write_text(json.dumps(completed,indent=2)+'\n')


def aggregate(rows,fields):
    groups=defaultdict(list)
    for r in rows:
        if 'relative_l2' in r:groups[tuple(r[k] for k in fields)].append(r)
    out=[]
    for key,items in sorted(groups.items()):
        out.append(dict(zip(fields,key),count=len(items),
            median_relative_l2=statistics.median(r['relative_l2'] for r in items),
            max_relative_l2=max(r['relative_l2'] for r in items),
            exact_equal_fraction=sum(r['equal'] for r in items)/len(items),
            median_changed_fraction=statistics.median(r['changed_fraction'] for r in items),
            median_delta_norm=statistics.median(r['delta_norm'] for r in items),
            median_reference_norm=statistics.median(r['reference_norm'] for r in items),
            median_euler_delta_relative_xt=(statistics.median(r['euler_delta_relative_xt'] for r in items) if 'euler_delta_relative_xt' in items[0] else None),
            max_abs=max(r['max_abs'] for r in items)))
    return out


def report(a):
    p=read_plan(a.plan);root=Path(a.output_dir);c=p['config'];layers=[];velocity=[];seen=set();trajectories=[];locations=[]
    for done in sorted(root.glob('workers/worker_*/completed.json')):
        run=json.loads((done.parent/'run.json').read_text())
        if run['plan_sha256']!=sha256(a.plan):raise ValueError('worker provenance differs')
        for tr in json.loads(done.read_text()):
            key=(tr['prompt_id'],tr['seed'])
            if key in seen:raise ValueError('duplicate diagnostic trajectory')
            seen.add(key);trajectories.append(tr)
            if tr['denoiser_steps']!=49 or [case['step_index'] for case in tr['cases']]!=c['steps']:raise ValueError('missing diagnostic steps')
            for case in tr['cases']:
                if not all(case['contracts'].values()):raise ValueError('failed read-only contract')
                path=Path(tr['path'])/f"step_{case['step_index']:02d}"
                labels=dict(prompt_id=tr['prompt_id'],seed=tr['seed'],step_index=case['step_index'])
                lr=read_jsonl(path/'layers.jsonl');vr=read_jsonl(path/'velocity.jsonl')
                if len(lr)!=case['layer_rows'] or len(vr)!=case['velocity_rows']:raise ValueError('truncated diagnostic rows')
                # Each final candidate must expose all 28 layer outputs.
                for depth in range(c['max_rounds']+1):
                    actual={r['layer'] for r in lr if r['series']=='final_depth' and r['component']=='gen_output_hidden' and r['round']==depth and r['subset']=='image' and r['comparison'] in ('first_snapshot','adjacent_round')}
                    if actual!=set(range(28)):raise ValueError('missing final GEN layers')
                if not all(r['finite'] for r in lr+vr):raise ValueError('nonfinite diagnostic data')
                layers.extend(dict(r,**labels) for r in lr);velocity.extend(dict(r,**labels) for r in vr)
                for depth in range(2,c['max_rounds']+1):
                    adjacent=[r for r in lr if r['comparison']=='adjacent_round' and r['round']==depth]
                    def frozen(component,subset,series):
                        grouped=defaultdict(list)
                        for r in adjacent:
                            if r['component'] in component and r['subset']==subset and r['series']==series:grouped[r['layer']].append(r)
                        return [i for i,items in sorted(grouped.items()) if len(items)==len(component) and all(r['equal'] for r in items)]
                    def find_layer(layer):
                        return next(r for r in adjacent if r['series']=='final_depth' and r['component']=='gen_output_hidden' and r['subset']=='image' and r['layer']==layer)
                    body=find_layer(7);suffix=find_layer(27)
                    cv=next(r for r in vr if r['component']=='conditional' and r['comparison']=='adjacent_round' and r['round']==depth)
                    gv=next(r for r in vr if r['component']=='cfg_post_renorm' and r['comparison']=='adjacent_round' and r['round']==depth)
                    locations.append(dict(labels,from_round=depth-1,round=depth,
                        exactly_frozen_memory_KV_layers=frozen(('memory_read_K','memory_read_V'),'content','within_deepest'),
                        exactly_frozen_gen_input_KV_layers=frozen(('gen_input_K','gen_input_V'),'image','within_deepest'),
                        body_exit_relative_l2=body['relative_l2'],suffix_exit_relative_l2=suffix['relative_l2'],
                        conditional_velocity_relative_l2=cv['relative_l2'],cfg_velocity_relative_l2=gv['relative_l2'],
                        cfg_to_cond_delta_norm_ratio=gv['delta_norm']/max(cv['delta_norm'],1e-12),
                        euler_delta_relative_xt=gv['euler_delta_relative_xt'],
                        interpretation='descriptive measurements; not semantic gain or a causal intervention'))
    expected={(i,s) for i in p['prompt_ids'] for s in c['seeds']}
    if seen!=expected:raise ValueError(f'incomplete trajectories: {len(seen)}/{len(expected)}')
    (root/'localisation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in locations))
    lsummary=aggregate(layers,['step_index','series','component','phase','layer','subset','comparison','round'])
    vsummary=aggregate(velocity,['step_index','component','comparison','round'])
    for name,rows in [('layers',lsummary),('velocity',vsummary)]:
        with (root/f'{name}_summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(status='complete',trajectories=len(seen),fixed_state_probes=len(seen)*len(c['steps']),
        source_sha256=p['source_sha256'],scope=p['comparison_scope'],read_only_contracts_passed=True,localisation_rows=len(locations),
        layers=lsummary,velocity=vsummary,semantic_gain_verified=False,quality_retention_verified=False)
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Memory 增量定位','',p['comparison_scope'],'',
        f"完整轨迹 {len(seen)}；固定状态探针 {summary['fixed_state_probes']}。全部只读合同通过。",
        '数值变化不代表语义提升。下表为固定 x_t/t 下相邻轮的相对 L2 中位数。',
        '|step|R变化|conditional velocity|CFG 后 velocity|', '|---|---|---:|---:|']
    vmap={(r['step_index'],r['component'],r['round']):r for r in vsummary if r['comparison']=='adjacent_round'}
    for step in c['steps']:
        for depth in range(1,c['max_rounds']+1):
            cond=vmap[step,'conditional',depth];cfg=vmap[step,'cfg_post_renorm',depth]
            lines.append(f"|{step}|{depth-1}→{depth}|{cond['median_relative_l2']:.6g}|{cfg['median_relative_l2']:.6g}|")
    lines+=['','逐样本定位见 localisation.jsonl；逐层数值见 layers_summary.csv；velocity 见 velocity_summary.csv。',
        'memory_block_update 是一次 UND block 内的变化，不是轮间增量。',
        'final_depth 比较完整独立 R 的终轮 GEN；within_deepest 比较一次 R4 调用内部各轮。',
        'GEN KV 指原生 attention 的层输入 KV；Memory read KV 指实际供 GEN 读取的 KV。',
        '层0先固定、随后更多浅层固定，支持逐层锁定；Memory仍变化而GEN输出不变，指向读取不敏感；',
        'body变化而suffix/final velocity减弱，指向后半网络；conditional变化而CFG减弱，指向guidance/readout。',
        'raw reconstructed CFG 使用 float32 重建，仅诊断；cfg_post_renorm 是 BAGEL 实际返回值。',
        '层、步和prompt须分别查看，聚合中位数不能确认单个样本的因果机制。']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    esc=lambda x:html.escape(str(x),quote=True)
    page=['<!doctype html><meta charset="utf-8"><title>Memory 增量定位</title><style>body{font:16px system-ui;margin:32px;background:#faf9f6;color:#202020}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ddd}img{max-width:100%;width:180px}section{margin:28px 0}.grid{display:flex;gap:10px;flex-wrap:wrap}pre{white-space:pre-wrap}</style><h1>Memory 增量定位</h1>',
        '<p>固定状态比较，不是独立最终图片评测。x0 预览由 x_t−t·v 推导；它是早期预测，不是已完成图像。</p>',
        '<p><a href="layers_summary.csv">逐层 CSV</a> · <a href="velocity_summary.csv">速度 CSV</a> · <a href="summary.json">完整汇总</a></p>',
        '<pre>'+esc('\n'.join(lines))+'</pre>']
    for step in c['steps']:
        page.append('<h2>逐层变化 · step '+str(step)+'</h2><p>单元格：相邻轮 relative L2 中位数 / 完全相同的样本比例。浅色表示更小的变化。— 表示此层或轮次没有该记录。</p>')
        for series,component,subset in [('within_deepest','gen_input_K','image'),
                ('within_deepest','gen_input_V','image'),('within_deepest','gen_output_hidden','image'),
                ('within_deepest','memory_output_hidden','content'),('within_deepest','memory_read_K','content'),
                ('within_deepest','memory_read_V','content'),('final_depth','gen_output_hidden','image'),
                ('final_depth','gen_input_K','image'),('final_depth','gen_input_V','image'),
                ('final_depth','gen_read_memory_K','content'),('final_depth','gen_read_memory_V','content'),('final_depth','gen_normalized_hidden','image')]:
            found={(r['layer'],r['round']):r for r in lsummary if r['step_index']==step and r['series']==series
                and r['component']==component and r['subset']==subset and r['comparison']=='adjacent_round'}
            page.append('<h3>'+esc(series+' / '+component+' / '+subset)+'</h3><table><tr><th>层</th>'+''.join('<th>R'+str(r-1)+'→R'+str(r)+'</th>' for r in range(1,c['max_rounds']+1))+'</tr>')
            for layer in range(28):
                cells=[]
                for depth in range(1,c['max_rounds']+1):
                    item=found.get((layer,depth))
                    if item is None:cells.append('<td>—</td>');continue
                    value=item['median_relative_l2'];strength=min(1.,max(0.,value/.05))
                    cells.append(f'<td style="background:rgba(233,139,50,{strength:.3f})">{value:.3g} / {item["exact_equal_fraction"]:.0%}</td>')
                page.append('<tr><td>'+str(layer)+'</td>'+''.join(cells)+'</tr>')
            page.append('</table>')
    for tr in sorted(trajectories,key=lambda r:(r['index'],r['seed'])):
        page+=['<section><h2>'+esc(tr['prompt_id'])+' · seed '+str(tr['seed'])+'</h2><p>'+esc(tr['prompt'])+'</p>']
        directory=Path(tr['path'])
        for step in c['steps']:
            page+=['<h3>step '+str(step)+'</h3><div class="grid">']
            for depth in range(c['max_rounds']+1):
                image=directory/f'step_{step:02d}'/f'x0_R{depth}.png'
                if image.exists():page.append('<div>R'+str(depth)+'<br><img loading="lazy" src="'+esc(image.relative_to(root))+'"></div>')
            page.append('</div>')
        page+=['<p>Early20 R4 最终图</p><img loading="lazy" src="'+esc((directory/'trajectory_final.png').relative_to(root))+'"></section>']
    (root/'diagnostics.html').write_text('\n'.join(page))
    print(root/'diagnostics.html',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--model-path',required=True);prep.add_argument('--prompts',required=True)
    prep.add_argument('--config',default=str(ROOT/'configs/increment_diagnostics.json'));prep.add_argument('--plan',required=True)
    worker=sub.add_parser('run');worker.add_argument('--plan',required=True);worker.add_argument('--output-dir',required=True)
    worker.add_argument('--device',default='npu:0');worker.add_argument('--shard-index',type=int,default=0);worker.add_argument('--num-shards',type=int,default=1)
    merge=sub.add_parser('report');merge.add_argument('--plan',required=True);merge.add_argument('--output-dir',required=True)
    a=parser.parse_args();{'prepare':prepare,'run':run,'report':report}[a.command](a)

if __name__=='__main__':main()
