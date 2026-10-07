#!/usr/bin/env python
"""User-run six-window comparison: prepare, validate, generate, score, report."""
import argparse
import base64
import hashlib
import html
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen_latent_cot.evaluation.io import read_jsonl, sha256, source_hash, load_manifests, identity
from qwen_latent_cot.evaluation.windows import load_comparison, validate_config, arm_configs


def prepare_main():
    p = argparse.ArgumentParser(description='Bind six windows, prompts, weights and source before workers')
    p.add_argument('--model-path', required=True)
    p.add_argument('--prompts', default=str(ROOT/'data/prompts32.jsonl'))
    p.add_argument('--config', default=str(ROOT/'configs/window_comparison.json'))
    p.add_argument('--plan', required=True)
    a = p.parse_args()
    weights = Path(a.model_path).resolve()
    depth = json.loads((weights/'llm_config.json').read_text())['num_hidden_layers']
    config = load_comparison(a.config, depth)
    data = read_jsonl(a.prompts)[:config['max_prompts']]
    ids = [str(r.get('prompt_id', r.get('id', i))) for i,r in enumerate(data)]
    if not data or len(set(ids)) != len(ids) or any(not r['prompt'].strip() for r in data):
        raise ValueError('require nonempty, uniquely identified prompts')
    model_files = [weights/n for n in ('ema.safetensors','ae.safetensors')]
    if not all(f.is_file() for f in model_files):
        raise ValueError('require native ema.safetensors and ae.safetensors')
    model_files += [weights/n for n in ('llm_config.json','vit_config.json','tokenizer.json',
        'tokenizer_config.json','vocab.json','merges.txt') if (weights/n).is_file()]
    print('Hashing native weights once for all workers...', flush=True)
    plan = {'schema':10, 'architecture':'persistent_und_full_prompt_kv_v1',
        'source_sha256':source_hash(ROOT), 'config_sha256':sha256(a.config), 'config':config,
        'model_path':str(weights), 'model_sha256':{f.name:sha256(f) for f in model_files},
        'model_stat':{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in model_files},
        'benchmark':str(Path(a.prompts).resolve()),'benchmark_sha256':sha256(a.prompts),
        'prompt_ids':ids, 'seeds':config['seeds'], 'arms':list(arm_configs(config)),
        'arm_configs':arm_configs(config), 'sampling':dict(config['sampling'],
            actual_denoiser_calls=config['sampling']['num_timesteps']-1),
        'native_depth':depth, 'stage':'evaluation', 'diagnostics':False,
        'memory_topologies':{'mode':'persistent_und_full_prompt_dynamic_kv',
            'suffix':'dynamic_final_writer_continuation', 'capacity':'full_prompt',
            'special_hidden_and_kv':'pinned_native', 'null_cfg':'native_bypass'},
        'loop':{'extra_rounds':2,'progress_start':0.,'progress_end':1.}}
    output=Path(a.plan)
    if output.exists():raise ValueError('use a fresh plan file')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(plan,indent=2)+'\n')
    print(f"Prepared {len(ids)*len(plan['seeds'])*len(plan['arms'])} images, Base + six R2 windows",flush=True)


def read_plan(path):
    plan=json.loads(Path(path).read_text())
    if plan['source_sha256'] != source_hash(ROOT):raise ValueError('source changed after plan binding')
    if plan['benchmark_sha256'] != sha256(plan['benchmark']):raise ValueError('benchmark changed')
    for name,stat in plan['model_stat'].items():
        f=Path(plan['model_path'])/name
        if [f.stat().st_size,f.stat().st_mtime_ns] != stat:raise ValueError('native model files changed')
    depth=json.loads((Path(plan['model_path'])/'llm_config.json').read_text())['num_hidden_layers']
    if arm_configs(validate_config(plan['config'],depth)) != plan['arm_configs']:
        raise ValueError('inconsistent arm configuration')
    return plan


def generate_main():
    p=argparse.ArgumentParser(description='Generate Base + six fixed R2 windows on a prompt shard')
    p.add_argument('--plan',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1)
    args=p.parse_args()
    import torch
    from qwen_latent_cot.bagel.backbone import load_native
    from qwen_latent_cot.bagel.inferencer import T2IGenerator, InvalidGeneratedImage
    from qwen_latent_cot.bagel.internal_loop import LoopConfig, InternalLoopRuntime
    plan=read_plan(args.plan)
    if not 0<=args.shard_index<args.num_shards:raise ValueError('invalid generation shard')
    torch.cuda.set_device(torch.device(args.device))
    data=read_jsonl(plan['benchmark'])[:len(plan['prompt_ids'])];ids=plan['prompt_ids'];seeds=plan['seeds']
    sampling=plan['sampling']
    args.model_path=plan['model_path'];args.timestep_shift=sampling['timestep_shift']
    args.num_timesteps=sampling['num_timesteps'];args.image_size=sampling['image_size']
    args.cfg_text_scale=sampling['cfg_text_scale'];args.cfg_renorm_type=sampling['cfg_renorm_type']
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    provenance=dict(plan,plan_sha256=sha256(args.plan),shard=[args.shard_index,args.num_shards],
        gpu=torch.cuda.get_device_name(),precision='bfloat16',kernel='native_flash_attention',torch=torch.__version__,training=False)
    runfile=output/'run.json'
    if runfile.exists():raise ValueError('generation worker requires a fresh output directory')
    runfile.write_text(json.dumps(provenance,indent=2)+'\n')
    manifest=output/'manifest.jsonl';manifest.touch()
    bundle = load_native(args.model_path,args.device,args.timestep_shift)
    jobs = [(i,s) for i in range(len(data)) for s in seeds]
    if len(bundle.model.language_model.model.layers)!=plan['native_depth']:
        raise ValueError('loaded decoder depth differs from plan')
    for arm, config in plan['arm_configs'].items():
        cfg=LoopConfig(**config);mode=cfg.mode;rounds=cfg.extra_rounds
        runtime = InternalLoopRuntime(bundle.model,cfg)
        generator = T2IGenerator(bundle,runtime)
        try:
            for ordinal,(i,seed) in enumerate(jobs):
                if ordinal%args.num_shards != args.shard_index: continue
                row=data[i]; shape=(int(row.get('height',args.image_size)),int(row.get('width',args.image_size)))
                name=f'{i:05d}_s{seed}.png'; imagepath=output/arm/name; imagepath.parent.mkdir(exist_ok=True)
                steps=args.num_timesteps
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                started=time.perf_counter()
                invalid_error = None
                try:
                    images, hashes=generator.generate([row['prompt']],[shape],[seed],num_timesteps=steps,
                        timestep_shift=args.timestep_shift,cfg_text_scale=args.cfg_text_scale,cfg_renorm_type=args.cfg_renorm_type)
                except InvalidGeneratedImage as error:
                    images = None; hashes = error.noise_hashes; invalid_error = str(error)
                torch.cuda.synchronize(); elapsed=time.perf_counter()-started
                peak=torch.cuda.max_memory_allocated()
                if images is not None: images[0].save(imagepath)
                record={'arm':arm,'prompt_id':ids[i],'index':i,'prompt':row['prompt'],'seed':seed,
                    'bucket':row.get('bucket','unclassified'),'height':shape[0],'width':shape[1],
                    'path':str(imagepath),'image_sha256':sha256(imagepath) if images is not None else None,'noise_sha256':hashes[0],
                    'valid_file':images is not None,'decode_error':invalid_error,'generation_seconds':elapsed,'peak_allocated_bytes':peak,
                    'timing_scope':'engineering_single_generation_no_warmups', 'extra_rounds':rounds,
                    'body_pass_count_scope':'configured_active_branch; full Memory requires nonempty prompt cache',
                    'native_prompt_lengths':list(generator.prompt_lengths),
                    'memory_capacity_policy':'full_prompt',
                    'full_memory_lengths':list(generator.prompt_lengths) if mode!='BASE' else None,
                    'writer_body_passes':rounds if mode!='BASE' else 0,
                    'writer_suffix_passes':int(mode!='BASE' and rounds>0 and cfg.end_layer<len(bundle.model.language_model.model.layers)),
                    'body_passes':1+rounds,
                    'num_timesteps':steps}
                record.update(start_layer=cfg.start_layer,end_layer=cfg.end_layer,
                    writer_suffix_layers=(plan['native_depth']-cfg.end_layer if mode!='BASE' else 0))
                with manifest.open('a') as f: f.write(json.dumps(record)+'\n'); f.flush()
                print(f'{arm} prompt={ids[i]} seed={seed} {elapsed:.2f}s peak={peak/2**30:.2f}GiB',flush=True)
        finally: runtime.close()
    print(f'Completed shard: {manifest}',flush=True)

def comparison(a,b):
    from qwen_latent_cot.evaluation.memory_rounds import tensor_metrics
    return tensor_metrics(a,b)

def validate(bundle,generator,runtime,depths):
    import torch
    from qwen_latent_cot.bagel.internal_loop import MODE
    from qwen_latent_cot.bagel.native_und import project_und
    from qwen_latent_cot.evaluation.memory_rounds import MemoryRoundCapture
    cfg=runtime.config
    device=next(bundle.model.language_model.parameters()).device
    results=dict(depths=list(depths),input_scope='seeded_gaussian_test_input_not_sampled_trajectory',
        timesteps={},state_contracts={},semantic_gain_verified=False,quality_retention_verified=False)
    with torch.inference_mode(),generator.autocast():
        flow,hashes=generator.prepare(['A red cube to the left of a blue sphere.',
            'Two yellow birds above a green tree.'],[(256,256),(256,384)],[123,456])
        noise=flow.pop('packed_init_noises');cache=flow['past_key_values']
        before={i:(k,k.clone(),cache.value_cache[i],cache.value_cache[i].clone()) for i,k in cache.key_cache.items()}
        seed=runtime.layerwise.seeds[cache];hidden_before={i:h.clone() for i,h in seed.layer_hidden.items()}
        cos,sin=runtime.decoder.rotary_emb(seed.hidden,seed.positions.unsqueeze(0));rope=cos.squeeze(0),sin.squeeze(0)
        results['native_hidden_seed_kv_parity']={}
        for i,h in seed.layer_hidden.items():
            _,k,v=project_und(runtime.decoder.layers[i],h,rope)
            results['native_hidden_seed_kv_parity'][str(i)]={'K':comparison(cache.key_cache[i][seed.source_indexes],k),
                                                         'V':comparison(cache.value_cache[i][seed.source_indexes],v)}
        results['native_prompt_lengths']=list(generator.prompt_lengths)
        results['full_memory_lengths']=list(seed.lengths)
        results['all_prompt_slots_preserved']=seed.lengths==tuple(generator.prompt_lengths)
        for t in (.7,.3):
            kwargs=dict(flow,x_t=noise,timestep=torch.full((len(noise),),t,device=device),
                        cfg_text_scale=4.,cfg_renorm_type='global')
            runtime.config=replace(cfg,mode='BASE');runtime.kv_observer=None
            native=bundle.model._forward_flow(**kwargs)
            native_cond=bundle.model._forward_flow(**{**kwargs,'cfg_text_scale':1.})
            runtime.config=replace(cfg,extra_rounds=0)
            checks={'R0':comparison(native,bundle.model._forward_flow(**kwargs)),
                    'R0_conditional':comparison(native_cond,bundle.model._forward_flow(**{**kwargs,'cfg_text_scale':1.}))}
            capture=MemoryRoundCapture(max(depths))
            for r in depths:
                runtime.config=replace(cfg,mode=MODE,extra_rounds=r)
                runtime.kv_observer=capture
                checks[f'conditional_R{r}']=comparison(native_cond,bundle.model._forward_flow(**{**kwargs,'cfg_text_scale':1.}))
                runtime.kv_observer=None
                checks[f'cfg_R{r}']=comparison(native,bundle.model._forward_flow(**kwargs))
            results['timesteps'][str(t)]=checks
            mask=seed.special_mask.cpu()
            results['state_contracts'][str(t)]={
                'all_read_layers_present':all(set(capture.reads[r])==set(range(cfg.start_layer,len(runtime.decoder.layers))) for r in depths),
                'writer_suffix_once_per_depth':capture.suffix_writer_count==len(depths)*(len(runtime.decoder.layers)-cfg.end_layer),
                'all_hidden_updates_finite':all(row['finite'] for row in capture.hidden_rows),
                'hidden_update_count':len([row for row in capture.hidden_rows if row['subset']=='all'])==max(depths)*(cfg.end_layer-cfg.start_layer),
                'special_hidden_pinned':all(row['equal'] for row in capture.hidden_rows if row['subset']=='special'),
                'special_kv_pinned':all(torch.equal(kv[0][mask],capture.native[i][0][mask]) and
                    torch.equal(kv[1][mask],capture.native[i][1][mask]) for reads in capture.reads.values() for i,kv in reads.items())}
        results['prompt_cache_unchanged']=all(cache.key_cache[i] is k and cache.value_cache[i] is v and
            torch.equal(k,kcopy) and torch.equal(v,vcopy) for i,(k,kcopy,v,vcopy) in before.items())
        results['native_hidden_seeds_unchanged']=all(torch.equal(seed.layer_hidden[i],h) for i,h in hidden_before.items())
        results['noise_sha256']=hashes
    runtime.config=cfg;runtime.kv_observer=None
    results['passed']=(results['all_prompt_slots_preserved'] and results['prompt_cache_unchanged'] and results['native_hidden_seeds_unchanged']
        and all(v['equal'] and v['finite'] for c in results['native_hidden_seed_kv_parity'].values() for v in c.values())
        and all(all(c.values()) for c in results['state_contracts'].values())
        and all(v['finite'] and (v['equal'] if name.startswith('R0') else True) for c in results['timesteps'].values() for name,v in c.items()))
    results['conditional_effect_observed']=any(not c[f'conditional_R{r}']['equal'] for c in results['timesteps'].values() for r in depths)
    return results

def validate_main():
    p=argparse.ArgumentParser(description='Real-weight numerical contracts for all six windows')
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    import torch
    from qwen_latent_cot.bagel.backbone import load_native
    from qwen_latent_cot.bagel.internal_loop import LoopConfig,InternalLoopRuntime
    from qwen_latent_cot.bagel.inferencer import T2IGenerator
    plan=read_plan(a.plan);torch.cuda.set_device(torch.device(a.device))
    bundle=load_native(plan['model_path'],a.device,plan['sampling']['timestep_shift'])
    results={}
    for arm,config in plan['arm_configs'].items():
        if arm=='BASE':continue
        runtime=InternalLoopRuntime(bundle.model,LoopConfig(**config))
        try:
            results[arm]=validate(bundle,T2IGenerator(bundle,runtime),runtime,(2,))
            print(f"Numerical contracts {arm}: {results[arm]['passed']}",flush=True)
        finally:runtime.close()
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'plan_sha256':sha256(a.plan),'windows':results},indent=2)+'\n')
    if not all(r['passed'] for r in results.values()):raise SystemExit(1)

def score_main():
    from qwen_latent_cot.evaluation.scoring import LocalScorer
    p=argparse.ArgumentParser(description='Paired GenEval2/TIIF semantics, quality proxy, Repair/Damage and prompt-cluster CI')
    p.add_argument('--manifests',nargs='+',required=True);p.add_argument('--benchmark',required=True)
    p.add_argument('--judge-model',required=True);p.add_argument('--geneval2-source',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--output-dir',required=True)
    p.add_argument('--bootstrap-replicates',type=int,default=10000)
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    args=p.parse_args()
    if not 0<=args.shard_index<args.num_shards:raise ValueError('invalid scoring shard')
    records,run=load_manifests(args.manifests)
    if run['benchmark_sha256']!=sha256(args.benchmark):raise ValueError('benchmark differs from generation')
    data=read_jsonl(args.benchmark)
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    work={key:i for i,key in enumerate(sorted({identity(r) for r in records}))}
    records=[r for r in records if work[identity(r)]%args.num_shards==args.shard_index]
    scorer=LocalScorer(args.judge_model,args.geneval2_source,args.device)
    provenance={'run':run,'scorer':scorer.provenance,'scoring_shard':[args.shard_index,args.num_shards]}
    binding=output/'scorer.json'
    if binding.exists() and json.loads(binding.read_text())!=provenance:raise ValueError('scorer binding changed')
    binding.write_text(json.dumps(provenance,indent=2)+'\n')
    scorefile=output/'scores.jsonl';done={}
    scorefile.touch(exist_ok=True)
    if scorefile.exists():
        for r in read_jsonl(scorefile):
            key=(r['arm'],*identity(r))
            if key in done:raise ValueError('duplicate scoring cache')
            done[key]=r
    scored=[]
    for row in records:
        key=(row['arm'],*identity(row));benchmark=data[row['index']]
        if benchmark['prompt']!=row['prompt']:raise ValueError('benchmark prompt mismatch')
        if key in done:
            result=done[key]
            if result['image_sha256']!=row['image_sha256']:raise ValueError('scored image changed')
        else:
            result=scorer.score(row,benchmark)
            with scorefile.open('a') as f:f.write(json.dumps(result)+'\n');f.flush()
        scored.append(result);print(f"scored {row['arm']} {row['prompt_id']} seed={row['seed']}",flush=True)
    if args.num_shards>1:
        print(f'Completed scoring shard: {scorefile}',flush=True)
        return
    write_summary(scored,run,output,args.bootstrap_replicates)

def write_summary(scored,run,output,resamples=10000):
    from qwen_latent_cot.evaluation.report import summarize
    summary={'status':'engineering_scored' if run['stage']=='engineering' else 'paired_scored',
        'comparison':'six_layer_windows_R2', 'arm_configs':run['arm_configs'],
        'stage':run['stage'],'source_sha256':run['source_sha256'],'model_sha256':run['model_sha256'],
        'quality_is_proxy':True,'manual_review':'pending','statistics_unit':'prompt_cluster_all_seeds_and_atoms',
        'generation_provenance_scope':'single_source',
        'arms':summarize(scored,resamples)}
    summary['arms']={arm:summary['arms'][arm] for arm in run['arms']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Frozen BAGEL: six layer windows at R2','','Quality is a VLM proxy; manual review is pending. Equal body width does not equal total compute; writer suffix lengths differ. Engineering timing does not establish budget compliance.','',
        '|Arm|Semantic GM|Quality proxy|Invalid|Net Repair vs Base (95% CI)|Repair / Damage|',
        '|---|---:|---:|---:|---|---:|']
    for arm,s in summary['arms'].items():
        delta=s['vs_BASE']['net_repair'];lo,hi=delta['ci95']
        lines.append(f"|{arm}|{s['semantic_gm']:.4f}|{s['quality_proxy']:.4f}|{s['invalid_rate']:.4f}|{delta['mean']:.4f} [{lo:.4f}, {hi:.4f}]|{s['vs_BASE']['repair_count']} / {s['vs_BASE']['damage_count']}|")
    lines+=['','|Arm|Window [start,end)|Writer suffix layers|Median seconds|Peak allocated GiB|',
        '|---|---|---:|---:|---:|']
    for arm,s in summary['arms'].items():
        cfg=run['arm_configs'][arm]
        window='native' if arm=='BASE' else f"[{cfg['start_layer']},{cfg['end_layer']})"
        suffix=0 if arm=='BASE' else run['native_depth']-cfg['end_layer']
        lines.append(f"|{arm}|{window}|{suffix}|{s['latency_median_seconds']:.2f}|{s['peak_allocated_bytes']/2**30:.2f}|")
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    print(output/'summary.md',flush=True)

def report_main():
    p=argparse.ArgumentParser(description='Validate all scoring shards and build paired quality/semantic report')
    p.add_argument('--manifests',nargs='+',required=True);p.add_argument('--score-dirs',nargs='+',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--bootstrap-replicates',type=int,default=10000)
    a=p.parse_args();records,run=load_manifests(a.manifests)
    expected={(r['arm'],*identity(r)):r for r in records};scored={};bindings=[]
    for d in a.score_dirs:
        path=Path(d);bindings.append(json.loads((path/'scorer.json').read_text()))
        for r in read_jsonl(path/'scores.jsonl'):
            key=(r['arm'],*identity(r))
            if key in scored or key not in expected:raise ValueError('unknown or duplicate scoring identity')
            if r['image_sha256']!=expected[key]['image_sha256']:raise ValueError('scored image changed')
            scored[key]=r
    if set(scored)!=set(expected):raise ValueError('incomplete scoring; no missing-pair removal')
    for binding in bindings:
        if binding['run']!=run:raise ValueError('scoring provenance differs from generation')
        if binding['run']!=bindings[0]['run'] or binding['scorer']!=bindings[0]['scorer']:raise ValueError('incompatible scoring provenance')
    output=Path(a.output_dir);output.mkdir(parents=True,exist_ok=True)
    values=[scored[k] for k in sorted(scored)]
    (output/'scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
    (output/'scorer.json').write_text(json.dumps({'run':run,'scorer':bindings[0]['scorer'],'scoring_shards':len(bindings)},indent=2)+'\n')
    write_summary(values,run,output,a.bootstrap_replicates)
    print(json.dumps(export(Path(a.run_dir),Path(a.run_dir)/'comparison.html')))

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
    </style><main><h1>BAGEL：六个层窗口 · R2</h1><p>同 prompt、同 noise seed。图片嵌入文件，可离线查看。点击图片查看原尺寸。评分是模型代理；Repair / Damage 未经人工确认。六个窗口的 body 均为8层；UND writer suffix 长度不同。生成时间为工程日志。</p>''']
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
            cfg=summary['arm_configs'][arm]
            label=arm if arm=='BASE' else f"{arm} [{cfg['start_layer']},{cfg['end_layer']}) R2"
            parts.append(f'<div><div class="label">{esc(label)} · quality {row["quality_proxy"]:.2f}</div>')
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
    commands={'prepare':prepare_main,'validate':validate_main,'generate':generate_main,
        'score':score_main,'report':report_main}
    if len(sys.argv)<2 or sys.argv[1] not in commands:
        raise SystemExit('usage: compare_windows.py {prepare,validate,generate,score,report} ...')
    command=sys.argv.pop(1)
    commands[command]()


if __name__=='__main__':main()
