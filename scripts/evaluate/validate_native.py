#!/usr/bin/env python
"""E0: identical weights/input, native bypass and GEN/no-read velocity parity."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import sys
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from qwen_latent_cot.bagel.backbone import load_native
from qwen_latent_cot.bagel.internal_loop import LoopConfig,InternalLoopRuntime
from qwen_latent_cot.bagel.inferencer import T2IGenerator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    torch.cuda.set_device(torch.device(args.device))
    bundle=load_native(args.model_path,args.device)
    generator=T2IGenerator(bundle)
    shapes=[(256,256),(256,384)]
    prompts=['A red cube to the left of a blue sphere.','Two yellow birds above a green tree.']
    flow,_=generator.prepare(prompts,shapes,[123,456])
    noise=flow.pop('packed_init_noises')
    kwargs={**flow,'x_t':noise,'timestep':torch.full((len(noise),),.7,device=args.device),
            'cfg_text_scale':4.,'cfg_renorm_type':'text_channel'}
    kwargs['packed_vae_position_ids']=kwargs.pop('packed_vae_position_ids')
    results={}
    with torch.inference_mode(),generator.autocast():
        native=bundle.model._forward_flow(**kwargs)
        cfg=LoopConfig(evaluations=1)
        runtime=InternalLoopRuntime(bundle.model,cfg)
        try:
            bypass=bundle.model._forward_flow(**kwargs)
        finally:runtime.close()
        results['native_vs_N1']={'max_abs':float((native-bypass).abs().max()),'equal':bool(torch.equal(native,bypass))}
        cfg=replace(cfg,evaluations=2,mode='GEN_LAYERWISE')
        runtime=InternalLoopRuntime(bundle.model,cfg)
        try:gen=bundle.model._forward_flow(**kwargs)
        finally:runtime.close()
        cfg=replace(cfg,mode='MEMORY_NO_READ')
        runtime=InternalLoopRuntime(bundle.model,cfg)
        try:
            flow,_=T2IGenerator(bundle,runtime).prepare(prompts,shapes,[123,456])
            flow.pop('packed_init_noises')
            kwargs={**kwargs,**flow}
            cache=flow['past_key_values'];before={i:(v.clone(),cache.value_cache[i].clone()) for i,v in cache.key_cache.items()}
            noread=bundle.model._forward_flow(**kwargs)
            cache_equal=all(torch.equal(cache.key_cache[i],k) and torch.equal(cache.value_cache[i],v) for i,(k,v) in before.items())
        finally:runtime.close()
        results['gen_vs_memory_no_read']={'max_abs':float((gen-noread).abs().max()),'equal':bool(torch.equal(gen,noread))}
        results['prompt_cache_unchanged']=cache_equal
    results['packed_shapes']=shapes;results['seed']=[123,456];results['same_checkpoint']=bundle.model_path
    results['passed']=results['native_vs_N1']['equal'] and results['gen_vs_memory_no_read']['equal'] and cache_equal
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)
    if not results['passed']:raise SystemExit(1)


if __name__=='__main__':main()
