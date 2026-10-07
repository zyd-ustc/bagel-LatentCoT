#!/usr/bin/env python
"""User-run fixed-x_t Memory/velocity diagnosis on native Base trajectories."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import MethodType
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True)
    p.add_argument('--prompts',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--prompt-count',type=int,default=32)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--image-size',type=int,default=512)
    p.add_argument('--num-timesteps',type=int,default=50)
    p.add_argument('--probe-steps',default='0,24,48')
    p.add_argument('--timestep-shift',type=float,default=3.)
    p.add_argument('--cfg-text-scale',type=float,default=4.)
    p.add_argument('--start-layer',type=int,default=0)
    p.add_argument('--end-layer',type=int,default=8)
    p.add_argument('--num-shards',type=int,default=8)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--plan')
    return p


def inputs(args):
    from qwen_latent_cot.evaluation.io import read_jsonl,sha256,source_hash
    rows=read_jsonl(args.prompts)
    if args.prompt_count<=0 or len(rows)<args.prompt_count:
        raise ValueError('prompt file must contain the requested number of prompts')
    rows=rows[:args.prompt_count]
    if len({r['prompt'] for r in rows})!=len(rows):
        raise ValueError('diagnostic prompts must be distinct')
    for i,row in enumerate(rows):
        row.setdefault('prompt_id',f'prompt_{i:03d}')
    if len({r['prompt_id'] for r in rows})!=len(rows):
        raise ValueError('prompt IDs must be distinct')
    steps=sorted(set(int(x) for x in args.probe_steps.split(',')))
    if not steps or not 0<=min(steps)<=max(steps)<args.num_timesteps-1:
        raise ValueError('probe steps must index native denoiser calls')
    if not 0<=args.shard_index<args.num_shards<=len(rows):
        raise ValueError('invalid prompt shard')
    modes=['LAYERWISE_UND_STATE_REPLACE']
    settings=dict(schema=5,memory_modes=modes,seed=args.seed,image_size=args.image_size,num_timesteps=args.num_timesteps,
        probe_steps=steps,timestep_shift=args.timestep_shift,cfg_text_scale=args.cfg_text_scale,
        cfg_renorm_type='global',start_layer=args.start_layer,end_layer=args.end_layer,
        num_shards=args.num_shards,depths=[0,1,2,3],prompt_ids=[r['prompt_id'] for r in rows],
        benchmark_sha256=sha256(args.prompts),source_sha256=source_hash(ROOT),
        model_path=str(Path(args.model_path).resolve()),input_scope='native_base_trajectory_same_x_t',
        writer_comparison_scope='body_only_within_R3',
        final_read_scope='body_and_suffix_across_independent_depths')
    return rows,steps,settings


def measure_probe(bundle,runtime,config,generator,original,kwargs,native):
    import torch
    from qwen_latent_cot.evaluation.memory_rounds import MemoryRoundCapture,tensor_metrics,tensor_hash,velocity_comparisons
    x_before=kwargs['x_t'].detach().clone()
    runtime.config=replace(config,mode='BASE');runtime.kv_observer=None
    native_cond=original(**{**kwargs,'cfg_text_scale':1.}).detach().cpu().clone()
    runtime.config=replace(config,extra_rounds=0)
    r0_cfg=tensor_metrics(native.detach().cpu(),original(**kwargs).detach().cpu())
    r0_cond=tensor_metrics(native_cond,original(**{**kwargs,'cfg_text_scale':1.}).detach().cpu())
    capture=MemoryRoundCapture(3)
    conditional={0:native_cond};guided={0:native.detach().cpu().clone()}
    for depth in (1,2,3):
        runtime.config=replace(config,extra_rounds=depth)
        runtime.kv_observer=capture
        conditional[depth]=original(**{**kwargs,'cfg_text_scale':1.}).detach().cpu().clone()
        runtime.kv_observer=None
        guided[depth]=original(**kwargs).detach().cpu().clone()
    seed=runtime.layerwise.seeds[kwargs['past_key_values']]
    memory=capture.comparisons([1,2,3]);hidden=capture.hidden_rows
    velocity=velocity_comparisons(conditional,'conditional')+velocity_comparisons(guided,'cfg')
    sanity=dict(r0_cfg_equal=r0_cfg['equal'],r0_conditional_equal=r0_cond['equal'],
        r0_finite=r0_cfg['finite'] and r0_cond['finite'],x_t_unchanged=torch.equal(x_before,kwargs['x_t']),
        all_finite=all(r['finite'] for r in memory+velocity+hidden),
        special_kv_pinned=all(r['equal'] for r in memory if r['subset']=='special'),
        special_hidden_pinned=all(r['equal'] for r in hidden if r['subset']=='special'),
        all_read_layers_present=all(set(capture.reads[r])==set(range(config.start_layer,len(runtime.decoder.layers))) for r in (1,2,3)),
        hidden_state_export_complete={(r['layer'],r['from_round'],r['to_round']) for r in hidden if r['subset']=='all'}=={
            (layer,r,r+1) for layer in range(config.start_layer,config.end_layer) for r in range(3)},
        hidden_state_export_count=len([r for r in hidden if r['subset']=='all'])==3*(config.end_layer-config.start_layer),
        writer_suffix_once_per_depth=capture.suffix_writer_count==3*(len(runtime.decoder.layers)-config.end_layer),
        all_prompt_slots_preserved=seed.lengths==tuple(generator.prompt_lengths))
    runtime.kv_observer=None
    return [dict(memory_mode=config.mode,x_t_sha256=tensor_hash(x_before),memory=memory,hidden=hidden,
                 velocity=velocity,sanity=sanity)]


def main():
    args=parser().parse_args()
    rows,steps,settings=inputs(args)
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    if args.prepare_only:
        from qwen_latent_cot.evaluation.io import sha256
        if (output/'plan.json').exists():raise ValueError('use a fresh output directory')
        weights=sorted(Path(args.model_path).glob('*.safetensors'))
        if not weights:raise ValueError('native model weights not found')
        plan=dict(settings,model_sha256={p.name:sha256(p) for p in weights})
        (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        print(f'Prepared {len(rows)} prompts, {len(steps)} probes each; model hashes recorded once',flush=True)
        return
    if not args.plan:raise ValueError('workers require a prepared shared --plan')
    plan=json.loads(Path(args.plan).read_text())
    if any(plan.get(k)!=v for k,v in settings.items()):
        raise ValueError('worker source, inputs or settings differ from prepared plan')
    if (output/'samples.jsonl').exists():raise ValueError('use a fresh worker directory')
    selected=[(i,r) for i,r in enumerate(rows) if i%args.num_shards==args.shard_index]
    (output/'run.json').write_text(json.dumps(dict(plan,shard_index=args.shard_index),indent=2)+'\n')
    import torch
    from qwen_latent_cot.bagel.backbone import load_native
    from qwen_latent_cot.bagel.internal_loop import LoopConfig,InternalLoopRuntime
    from qwen_latent_cot.bagel.inferencer import T2IGenerator
    torch.cuda.set_device(torch.device(args.device))
    bundle=load_native(args.model_path,args.device,args.timestep_shift)
    config=LoopConfig(extra_rounds=3,start_layer=args.start_layer,end_layer=args.end_layer)
    runtime=InternalLoopRuntime(bundle.model,config)
    generator=T2IGenerator(bundle,runtime)
    original=bundle.model._forward_flow
    completed=0
    try:
        with (output/'samples.jsonl').open('x') as stream,torch.inference_mode(),generator.autocast():
            for index,row in selected:
                runtime.config=config
                flow,hashes=generator.prepare([row['prompt']],[(args.image_size,args.image_size)],[args.seed])
                cache=flow['past_key_values'];seed=runtime.layerwise.seeds[cache]
                cache_before={i:(k,k.detach().cpu().clone(),cache.value_cache[i],cache.value_cache[i].detach().cpu().clone())
                              for i,k in cache.key_cache.items()}
                step=0
                def forward(this,**kwargs):
                    nonlocal step,completed
                    runtime.config=replace(config,mode='BASE')
                    runtime.kv_observer=None
                    runtime.step_index=step;runtime.progress=step/max(args.num_timesteps-2,1)
                    # Return this exact Base prediction to the native sampler.
                    native=original(**kwargs)
                    if step in steps:
                        measured=measure_probe(bundle,runtime,config,generator,original,kwargs,native)
                        for result in measured:
                            record=dict(prompt_id=row['prompt_id'],index=index,prompt=row['prompt'],seed=args.seed,
                                step=step,timestep=float(kwargs['timestep'][0]),input_noise_sha256=hashes[0],
                                native_prompt_lengths=list(generator.prompt_lengths),**result)
                            stream.write(json.dumps(record)+'\n');stream.flush()
                            if not all(record['sanity'].values()):raise ValueError(f"diagnostic contract failed: {record['sanity']}")
                            completed+=1
                            print(f"{record['memory_mode']} prompt {index+1}/{len(rows)} probe step={step} t={record['timestep']:.6f} complete",flush=True)
                    runtime.config=replace(config,mode='BASE');runtime.kv_observer=None
                    step+=1
                    return native
                bundle.model._forward_flow=MethodType(forward,bundle.model)
                try:
                    bundle.model.generate_image(**flow,num_timesteps=args.num_timesteps,
                        timestep_shift=args.timestep_shift,cfg_text_scale=args.cfg_text_scale,
                        cfg_img_scale=1.,cfg_renorm_type='global',enable_taylorseer=False)
                finally:
                    bundle.model._forward_flow=original
                    runtime.kv_observer=None
                if step!=args.num_timesteps-1:raise ValueError('native trajectory call count changed')
                if not all(cache.key_cache[i] is k and cache.value_cache[i] is v
                    and torch.equal(k.detach().cpu(),kc) and torch.equal(v.detach().cpu(),vc)
                    for i,(k,kc,v,vc) in cache_before.items()):
                    raise ValueError('native prompt cache was modified')
                runtime.clear_prompt_state()
                print(f'Finished prompt {index+1}/{len(rows)}; native prompt cache unchanged',flush=True)
        if completed!=len(selected)*len(steps)*len(plan['memory_modes']):raise ValueError('incomplete shard probes')
        (output/'complete.json').write_text(json.dumps(dict(probes=completed,prompt_cache_unchanged=True))+'\n')
    finally:
        bundle.model._forward_flow=original
        runtime.kv_observer=None
        runtime.close()


if __name__=='__main__':main()
