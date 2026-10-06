#!/usr/bin/env python
"""User-run native-weight velocity parity and effect checks for full Memory."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--depths',default='1,2,3')
    p.add_argument('--start-layer',type=int,default=0);p.add_argument('--end-layer',type=int,default=8)
    args=p.parse_args()
    import torch
    from qwen_latent_cot.bagel.backbone import load_native
    from qwen_latent_cot.bagel.internal_loop import LoopConfig,InternalLoopRuntime
    from qwen_latent_cot.bagel.inferencer import T2IGenerator
    from qwen_latent_cot.evaluation.io import source_hash,sha256
    from qwen_latent_cot.evaluation.loop_depth import parse_depths
    from validate_layerwise import comparison
    depths=parse_depths(args.depths)
    torch.cuda.set_device(torch.device(args.device))
    bundle=load_native(args.model_path,args.device)
    cfg=LoopConfig(mode='LAYERWISE_FULL_MEMORY_REPLACE',extra_rounds=max(depths),
                   start_layer=args.start_layer,end_layer=args.end_layer,memory_slots=0)
    runtime=InternalLoopRuntime(bundle.model,cfg)
    generator=T2IGenerator(bundle,runtime)
    prompts=['A red cube to the left of a blue sphere.','Two yellow birds above a green tree.']
    shapes=[(256,256),(256,384)]
    results={'depths':list(depths),'timesteps':{},'input_scope':'seeded_gaussian_test_input_not_sampled_trajectory'}
    try:
        with torch.inference_mode(),generator.autocast():
            flow,hashes=generator.prepare(prompts,shapes,[123,456])
            noise=flow.pop('packed_init_noises')
            cache=flow['past_key_values']
            before={i:(k,k.clone(),cache.value_cache[i],cache.value_cache[i].clone()) for i,k in cache.key_cache.items()}
            seed=runtime.layerwise.seeds[cache]
            results['native_prompt_lengths']=list(generator.prompt_lengths)
            results['full_memory_lengths']=list(seed.lengths)
            results['all_prompt_slots_preserved']=seed.lengths==tuple(generator.prompt_lengths)
            results['special_slots_per_sample']=[int(mask.sum()) for mask in seed.special_mask.split(seed.lengths)]
            for t in [.7,.3]:
                kwargs={**flow,'x_t':noise,'timestep':torch.full((len(noise),),t,device=args.device),
                        'cfg_text_scale':4.,'cfg_renorm_type':'global'}
                runtime.config=replace(cfg,mode='BASE')
                native=bundle.model._forward_flow(**kwargs)
                native_cond=bundle.model._forward_flow(**{**kwargs,'cfg_text_scale':1.})
                checks={}
                runtime.config=replace(cfg,extra_rounds=0)
                checks['R0']=comparison(native,bundle.model._forward_flow(**kwargs))
                for r in depths:
                    for kind,mode in [('static','LAYERWISE_FULL_SEED_REPLACE'),('dynamic','LAYERWISE_FULL_MEMORY_REPLACE')]:
                        runtime.config=replace(cfg,mode=mode,extra_rounds=r)
                        checks[f'{kind}_R{r}']=comparison(native,bundle.model._forward_flow(**kwargs))
                        checks[f'{kind}_conditional_R{r}']=comparison(native_cond,
                            bundle.model._forward_flow(**{**kwargs,'cfg_text_scale':1.}))
                results['timesteps'][str(t)]=checks
            results['prompt_cache_unchanged']=all(cache.key_cache[i] is k and cache.value_cache[i] is v
                and torch.equal(k,kcopy) and torch.equal(v,vcopy) for i,(k,kcopy,v,vcopy) in before.items())
        results.update(model_path=str(Path(args.model_path).resolve()),source_sha256=source_hash(ROOT),
            model_sha256={name:sha256(Path(args.model_path)/name) for name in ('ema.safetensors','ae.safetensors') if (Path(args.model_path)/name).exists()},
            packed_shapes=shapes,seeds=[123,456],noise_sha256=hashes,
            start_layer=args.start_layer,end_layer=args.end_layer,special_token_kv='pinned_to_native',
            semantic_gain_verified=False,quality_retention_verified=False)
        # Dynamic magnitude is reported, without treating any nonzero delta as
        # semantic gain or selecting an arbitrary minimum perturbation threshold.
        results['passed']=(results['all_prompt_slots_preserved'] and results['prompt_cache_unchanged']
            and all(value['finite'] and (value['equal'] if key=='R0' or key.startswith('static') else True)
                    for checks in results['timesteps'].values() for key,value in checks.items()))
        results['dynamic_conditional_effect_observed']=any(not checks[f'dynamic_conditional_R{r}']['equal']
            for checks in results['timesteps'].values() for r in depths)
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2),flush=True)
        if not results['passed']:raise SystemExit(1)
    finally:runtime.close()


if __name__=='__main__':main()
