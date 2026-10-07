#!/usr/bin/env python
"""User-run native-weight checks for persistent UND Memory; no image scoring."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--depths',default='2')
    p.add_argument('--start-layer',type=int,default=0);p.add_argument('--end-layer',type=int,default=8)
    args=p.parse_args()
    import torch
    from qwen_latent_cot.bagel.backbone import load_native
    from qwen_latent_cot.bagel.internal_loop import LoopConfig,InternalLoopRuntime
    from qwen_latent_cot.bagel.inferencer import T2IGenerator
    from qwen_latent_cot.evaluation.io import source_hash,sha256
    from qwen_latent_cot.evaluation.loop_depth import parse_depths
    depths=parse_depths(args.depths)
    if args.device.startswith('cuda'):torch.cuda.set_device(torch.device(args.device))
    bundle=load_native(args.model_path,args.device)
    runtime=InternalLoopRuntime(bundle.model,LoopConfig(extra_rounds=max(depths),start_layer=args.start_layer,end_layer=args.end_layer))
    try:
        result=validate(bundle,T2IGenerator(bundle,runtime),runtime,depths)
        result.update(source_sha256=source_hash(ROOT),model_path=str(Path(args.model_path).resolve()),
            model_sha256={name:sha256(Path(args.model_path)/name) for name in ('ema.safetensors','ae.safetensors') if (Path(args.model_path)/name).exists()},
            start_layer=args.start_layer,end_layer=args.end_layer)
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
        if not result['passed']:raise SystemExit(1)
    finally:runtime.close()


if __name__=='__main__':main()
