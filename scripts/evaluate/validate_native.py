#!/usr/bin/env python
"""User-run E0: real native weights, native bypass/no-read and legacy velocity parity."""
import argparse
from pathlib import Path
from types import MethodType
import json
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from qwen_latent_cot.bagel.backbone import load_native
from qwen_latent_cot.bagel.internal_loop import LoopConfig,InternalLoopRuntime
from qwen_latent_cot.bagel.inferencer import T2IGenerator
from qwen_latent_cot.evaluation.io import source_hash
from oracles.parent_runner import legacy_kwargs
from oracles.legacy_memory_kernels import legacy_forward_inference


def comparison(a,b):
    return {'max_abs':float((a-b).abs().max()),'equal':bool(torch.equal(a,b))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--loop-rounds',type=int,default=1)
    p.add_argument('--start-layer',type=int,default=0);p.add_argument('--end-layer',type=int,default=8)
    p.add_argument('--memory-slots',type=int,default=8);args=p.parse_args()
    if args.loop_rounds<1:raise ValueError('legacy parity requires R>=1')
    torch.cuda.set_device(torch.device(args.device))
    bundle=load_native(args.model_path,args.device)
    generator=T2IGenerator(bundle)
    shapes=[(256,256),(256,384)]
    prompts=['A red cube to the left of a blue sphere.','Two yellow birds above a green tree.']
    flow,hashes=generator.prepare(prompts,shapes,[123,456])
    noise=flow.pop('packed_init_noises')
    kwargs={**flow,'x_t':noise,'timestep':torch.full((len(noise),),.7,device=args.device),
            'cfg_text_scale':4.,'cfg_renorm_type':'global'}
    decoder=bundle.model.language_model.model
    cache=flow['past_key_values'];copies={i:(v.clone(),cache.value_cache[i].clone()) for i,v in cache.key_cache.items()}
    results={}
    with torch.inference_mode(),generator.autocast():
        native=bundle.model._forward_flow(**kwargs)
        for mode,rounds,label in [('MEMORY_LOOP',0,'native_vs_R0'),
                                  ('MEMORY_NO_READ',args.loop_rounds,'native_vs_no_read')]:
            runtime=InternalLoopRuntime(bundle.model,LoopConfig(mode=mode,extra_rounds=rounds,
                start_layer=args.start_layer,end_layer=args.end_layer,memory_slots=args.memory_slots))
            try:actual=bundle.model._forward_flow(**kwargs)
            finally:runtime.close()
            results[label]=comparison(native,actual)
        original=decoder.forward_inference
        def parent(this,**kw):
            if kw.get('mode','und')!='gen':return original(**kw)
            expanded,indexes=legacy_kwargs(kw,args.memory_slots)
            output=legacy_forward_inference(this,**expanded,memory_loop_repeat=args.loop_rounds+1,
                memory_loop_start=args.start_layer,memory_loop_end=args.end_layer,block_gen_reads_memory=True)
            output.packed_query_sequence=output.packed_query_sequence[indexes]
            return output
        decoder.forward_inference=MethodType(parent,decoder)
        try:legacy=bundle.model._forward_flow(**kwargs)
        finally:decoder.forward_inference=original
        runtime=InternalLoopRuntime(bundle.model,LoopConfig(mode='MEMORY_LOOP',extra_rounds=args.loop_rounds,
            start_layer=args.start_layer,end_layer=args.end_layer,memory_slots=args.memory_slots))
        try:actual=bundle.model._forward_flow(**kwargs)
        finally:runtime.close()
        results['legacy_vs_memory_loop']=comparison(legacy,actual)
        results['prompt_cache_unchanged']=all(torch.equal(cache.key_cache[i],k) and torch.equal(cache.value_cache[i],v) for i,(k,v) in copies.items())
    results.update(packed_shapes=shapes,seed=[123,456],noise_sha256=hashes,
                   same_checkpoint=bundle.model_path,source_sha256=source_hash(ROOT),
                   loop={'R':args.loop_rounds,'start_layer':args.start_layer,'end_layer':args.end_layer,'memory_slots':args.memory_slots})
    results['passed']=all(results[k]['equal'] for k in ('native_vs_R0','native_vs_no_read','legacy_vs_memory_loop')) and results['prompt_cache_unchanged']
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)
    if not results['passed']:raise SystemExit(1)


if __name__=='__main__':main()
