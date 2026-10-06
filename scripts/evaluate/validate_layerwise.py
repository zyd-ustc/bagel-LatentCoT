#!/usr/bin/env python
"""User-run E0 with native weights; velocity effect is not a semantic metric."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from qwen_latent_cot.bagel.backbone import load_native
from qwen_latent_cot.bagel.internal_loop import LoopConfig,InternalLoopRuntime
from qwen_latent_cot.bagel.inferencer import T2IGenerator
from qwen_latent_cot.evaluation.io import source_hash,sha256


def comparison(reference,candidate):
    difference=(candidate-reference).float()
    return {'max_abs':float(difference.abs().max()),
            'relative_l2':float(difference.norm()/reference.float().norm().clamp_min(1e-12)),
            'equal':bool(torch.equal(reference,candidate)),
            'finite':bool(torch.isfinite(candidate).all())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--loop-rounds',type=int,default=1)
    p.add_argument('--start-layer',type=int,default=0);p.add_argument('--end-layer',type=int,default=8)
    p.add_argument('--memory-slots',type=int,default=8);args=p.parse_args()
    if args.loop_rounds<1 or args.memory_slots<1:raise ValueError('feedback check requires R>=1 and K>=1')
    torch.cuda.set_device(torch.device(args.device))
    bundle=load_native(args.model_path,args.device)
    cfg=LoopConfig(extra_rounds=args.loop_rounds,start_layer=args.start_layer,
                   end_layer=args.end_layer,memory_slots=args.memory_slots)
    runtime=InternalLoopRuntime(bundle.model,cfg)
    generator=T2IGenerator(bundle,runtime)
    prompts=['A red cube to the left of a blue sphere.','Two yellow birds above a green tree.']
    shapes=[(256,256),(256,384)]
    try:
        with torch.inference_mode(),generator.autocast():
            flow,hashes=generator.prepare(prompts,shapes,[123,456])
            noise=flow.pop('packed_init_noises')
            kwargs={**flow,'x_t':noise,'timestep':torch.full((len(noise),),.7,device=args.device),
                    'cfg_text_scale':4.,'cfg_renorm_type':'global'}
            cache=flow['past_key_values']
            before={i:(k,k.clone(),cache.value_cache[i],cache.value_cache[i].clone()) for i,k in cache.key_cache.items()}
            runtime.config=replace(cfg,mode='BASE')
            native=bundle.model._forward_flow(**kwargs)
            results={}
            for config,label in [(replace(cfg,extra_rounds=0),'native_vs_R0'),
                                 (replace(cfg,mode='LAYERWISE_KV_NO_READ'),'native_vs_no_read'),
                                 (cfg,'native_vs_layerwise')]:
                runtime.config=config
                actual=bundle.model._forward_flow(**kwargs)
                results[label]=comparison(native,actual)
            # Isolate the conditional velocity from the native CFG renormalization.
            runtime.config=replace(cfg,mode='BASE')
            conditional=bundle.model._forward_flow(**{**kwargs,'cfg_text_scale':1.})
            runtime.config=cfg
            edited=bundle.model._forward_flow(**{**kwargs,'cfg_text_scale':1.})
            results['conditional_native_vs_layerwise']=comparison(conditional,edited)
            results['prompt_cache_unchanged']=all(cache.key_cache[i] is k and cache.value_cache[i] is v
                and torch.equal(k,kcopy) and torch.equal(v,vcopy) for i,(k,kcopy,v,vcopy) in before.items())
            seed=runtime.layerwise.seeds[cache]
            results['memory_slots_per_sample']=list(seed.lengths)
            results['prompt_source_indexes']=seed.source_indexes.tolist()
        results.update(packed_shapes=shapes,seeds=[123,456],noise_sha256=hashes,
            model_path=str(Path(args.model_path).resolve()),source_sha256=source_hash(ROOT),
            model_sha256={name:sha256(Path(args.model_path)/name) for name in ('ema.safetensors','ae.safetensors') if (Path(args.model_path)/name).exists()},
            loop={'R':args.loop_rounds,'start_layer':args.start_layer,'end_layer':args.end_layer,
                  'memory_slots':args.memory_slots,'read_layers':list(range(args.start_layer+1,args.end_layer))},
            semantic_gain_verified=False,quality_retention_verified=False)
        results['passed']=(results['native_vs_R0']['equal'] and results['native_vs_no_read']['equal']
            and results['prompt_cache_unchanged'] and results['native_vs_layerwise']['finite']
            and results['conditional_native_vs_layerwise']['finite']
            and not results['conditional_native_vs_layerwise']['equal'])
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2),flush=True)
        if not results['passed']:raise SystemExit(1)
    finally:runtime.close()


if __name__=='__main__':main()
