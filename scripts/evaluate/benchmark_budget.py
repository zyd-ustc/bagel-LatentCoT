#!/usr/bin/env python
"""End-to-end GPU timing includes prefill, denoising and decode, excludes file I/O."""
import argparse
from pathlib import Path
from dataclasses import asdict
import json
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from qwen_latent_cot.bagel.backbone import load_native
from qwen_latent_cot.bagel.internal_loop import LoopConfig,InternalLoopRuntime
from qwen_latent_cot.bagel.inferencer import T2IGenerator
from qwen_latent_cot.evaluation.io import read_jsonl,sha256,source_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True);p.add_argument('--prompts',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--loop-rounds',type=int,default=1)
    p.add_argument('--start-layer',type=int,default=0);p.add_argument('--end-layer',type=int,default=8)
    p.add_argument('--memory-slots',type=int,default=8)
    p.add_argument('--progress-start',type=float,default=0.);p.add_argument('--progress-end',type=float,default=1.)
    p.add_argument('--arms',default='BASE,MEMORY_LOOP')
    p.add_argument('--warmups',type=int,default=3);p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--image-size',type=int,default=512);p.add_argument('--num-timesteps',type=int,default=50)
    p.add_argument('--matched-base-timesteps',type=int);args=p.parse_args()
    if args.warmups<3 or args.repeats<20:raise ValueError('budget contract requires >=3 warmups and >=20 measured generations')
    torch.cuda.set_device(torch.device(args.device));bundle=load_native(args.model_path,args.device)
    prompts=read_jsonl(args.prompts);measurements={}
    arms=args.arms.split(',')
    if 'BASE' not in arms:raise ValueError('budget measurement requires native Base')
    for arm in arms:
        cfg=LoopConfig(mode='BASE' if arm=='BASE_MATCHED_LATENCY' else arm,extra_rounds=args.loop_rounds,
            start_layer=args.start_layer,end_layer=args.end_layer,memory_slots=args.memory_slots,
            progress_start=args.progress_start,progress_end=args.progress_end)
        runtime=InternalLoopRuntime(bundle.model,cfg)
        gen=T2IGenerator(bundle,runtime);elapsed=[];peaks=[]
        try:
            steps=args.matched_base_timesteps if arm=='BASE_MATCHED_LATENCY' else args.num_timesteps
            if steps is None:raise ValueError('matched Base requires calibrated steps')
            for i in range(args.warmups+args.repeats):
                row=prompts[i%len(prompts)];shape=(row.get('height',args.image_size),row.get('width',args.image_size))
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                gen.generate([row['prompt']],[shape],[i],num_timesteps=steps,timestep_shift=3.,cfg_text_scale=4.)
                torch.cuda.synchronize();dt=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
                if i>=args.warmups:elapsed.append(dt);peaks.append(peak)
                print(f'{arm} generation={i} seconds={dt:.3f}',flush=True)
            measurements[arm]={'seconds':elapsed,'peak_allocated_bytes':peaks,
                'median_seconds':float(np.median(elapsed)),'p95_seconds':float(np.quantile(elapsed,.95)),
                'peak_bytes':max(peaks),'num_timesteps':steps,'loop_config':asdict(cfg)}
        finally:runtime.close()
    base=measurements['BASE']
    for arm,v in measurements.items():
        v['ratios_vs_base']={'median':v['median_seconds']/base['median_seconds'],
            'p95':v['p95_seconds']/base['p95_seconds'],'peak':v['peak_bytes']/base['peak_bytes']}
        r=v['ratios_vs_base'];v['budget_pass']=r['median']<=1.35 and r['p95']<=1.5 and r['peak']<=1.2
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    model_path=Path(args.model_path)
    native_files=[model_path/'ema.safetensors',model_path/'ae.safetensors'] if (model_path/'ema.safetensors').exists() else sorted(model_path.glob('*.safetensors'))
    output.write_text(json.dumps({'source_sha256':source_hash(ROOT),'benchmark_sha256':sha256(args.prompts),
        'native_model_sha256':{f.name:sha256(f) for f in native_files},'native_model_path':str(model_path.resolve()),
        'sampling':{'timestep_shift':3.,'cfg_text_scale':4.,'cfg_renorm_type':'global'},
        'gpu':torch.cuda.get_device_name(),'precision':'bfloat16',
        'batch_size':1,'warmups':args.warmups,'repeats':args.repeats,'probes':False,
        'includes':['prefill','denoising','decode'],'excludes':['model_load','file_write'],
        'measurements':measurements},indent=2)+'\n')


if __name__=='__main__':main()
