#!/usr/bin/env python
"""One frozen BAGEL worker. Multi-GPU launch shards prompt/seed jobs, never arms."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qwen_latent_cot.evaluation.io import read_jsonl, sha256, source_hash


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', required=True); p.add_argument('--prompts', required=True)
    p.add_argument('--output-dir', required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--arms', default='BASE,MEMORY_LOOP,LAYERWISE_MEMORY_KV')
    p.add_argument('--loop-depths', help='Paired layerwise depths, e.g. 1,2,3; BASE generated once')
    p.add_argument('--loop-rounds', type=int, default=1, help='R: extra whole-body passes; R=0 is native bypass')
    p.add_argument('--start-layer', type=int, default=0); p.add_argument('--end-layer', type=int, default=8)
    p.add_argument('--memory-slots', type=int, default=8)
    p.add_argument('--memory-seed', type=int, default=0, help='Boundary + 1e-4 slot-noise seed, independent of image seed')
    p.add_argument('--progress-start', type=float, default=0); p.add_argument('--progress-end', type=float, default=1.)
    p.add_argument('--seeds', default='0'); p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--num-timesteps', type=int, default=50); p.add_argument('--timestep-shift', type=float, default=3)
    p.add_argument('--cfg-text-scale', type=float, default=4)
    p.add_argument('--cfg-renorm-type', choices=['global','text_channel'], default='global')
    p.add_argument('--max-prompts', type=int); p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--matched-base-timesteps', type=int, help='Calibrate on development timing only')
    p.add_argument('--stage', choices=['engineering','development','evaluation'], default='engineering')
    p.add_argument('--probe-arm', choices=['MEMORY_LOOP','LAYERWISE_MEMORY_KV'], help='Default: layerwise if present, otherwise legacy')
    p.add_argument('--probe-steps',help='Export Memory KV + guided x0 estimates at selected denoiser step indices; diagnostic cost')
    p.add_argument('--diagnostics', action='store_true', help='Additional Memory diagnostics; excludes this run from budget claims')
    return p


def main():
    args = parser().parse_args()
    import torch
    from qwen_latent_cot.bagel.backbone import load_native
    from qwen_latent_cot.bagel.inferencer import T2IGenerator, InvalidGeneratedImage
    from qwen_latent_cot.bagel.internal_loop import LoopConfig, InternalLoopRuntime, MODES
    from qwen_latent_cot.evaluation.loop_depth import parse_depths,expand_arms
    if args.num_timesteps<2: raise ValueError('native schedule needs at least two time points')
    if not 0 <= args.shard_index < args.num_shards: raise ValueError('invalid worker shard')
    if args.device.startswith('cuda'): torch.cuda.set_device(torch.device(args.device))
    seeds = [int(s) for s in args.seeds.split(',')]
    arms = args.arms.split(',')
    if len(set(seeds)) != len(seeds) or len(set(arms)) != len(arms) or not seeds:
        raise ValueError('duplicate/empty seeds or arms')
    if any(a not in (*MODES,'BASE_MATCHED_LATENCY') for a in arms) or 'BASE' not in arms:
        raise ValueError('arms require BASE and recognized modes')
    if 'BASE_MATCHED_LATENCY' in arms and not args.matched_base_timesteps:
        raise ValueError('matched latency Base requires calibrated time points')
    data = read_jsonl(args.prompts)
    if args.max_prompts: data = data[:args.max_prompts]
    ids = [str(r.get('prompt_id',r.get('id',i))) for i,r in enumerate(data)]
    if not data or len(set(ids)) != len(ids): raise ValueError('empty/duplicate prompt IDs')
    probe_steps = sorted(set(int(x) for x in args.probe_steps.split(','))) if args.probe_steps else []
    probe_arm = args.probe_arm or ('LAYERWISE_MEMORY_KV' if 'LAYERWISE_MEMORY_KV' in arms else 'MEMORY_LOOP')
    depths=parse_depths(args.loop_depths)
    if depths and probe_steps:raise ValueError('depth comparison and probe export require separate runs')
    arm_specs=expand_arms(arms,depths,args.loop_rounds)
    if probe_steps and (probe_arm not in arms or args.loop_rounds<1 or args.memory_slots<1):raise ValueError('probe export requires Memory loop and R>=1')
    if any(i<0 or i>=args.num_timesteps-1 or not args.progress_start<=i/max(args.num_timesteps-2,1)<=args.progress_end for i in probe_steps):raise ValueError('probe steps outside active schedule')
    cfg = LoopConfig(mode='LAYERWISE_MEMORY_KV' if any(a.startswith('LAYERWISE') for a in arms) else 'MEMORY_LOOP',
        extra_rounds=max(depths) if depths else args.loop_rounds, start_layer=args.start_layer, end_layer=args.end_layer,
        memory_slots=args.memory_slots, progress_start=args.progress_start, progress_end=args.progress_end, memory_seed=args.memory_seed)
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    weights = Path(args.model_path)
    model_files = sorted(weights.glob('*.safetensors')) if not (weights/'ema.safetensors').exists() else [weights/'ema.safetensors', weights/'ae.safetensors']
    model_files += [weights/f for f in ('llm_config.json','vit_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt') if (weights/f).exists()]
    print('Hashing native weights and source...', flush=True)
    provenance = {'schema':3, 'architecture':'native_layerwise_input_kv_v1_with_legacy_control', 'source_sha256':source_hash(ROOT),
        'model_sha256':{p.name:sha256(p) for p in model_files}, 'model_path':str(weights.resolve()),
        'benchmark_sha256':sha256(args.prompts), 'benchmark':str(Path(args.prompts).resolve()),
        'sampling':{'num_timesteps':args.num_timesteps,'actual_denoiser_calls':args.num_timesteps-1,
                    'timestep_shift':args.timestep_shift,'cfg_text_scale':args.cfg_text_scale,
                    'cfg_renorm_type':args.cfg_renorm_type,'matched_base_timesteps':args.matched_base_timesteps,
                    'image_size':args.image_size},
        'loop':asdict(cfg), 'loop_depths':list(depths),
        'arm_configs':{label:asdict(replace(cfg,mode=mode,extra_rounds=rounds)) for label,mode,rounds in arm_specs},
        'arms':[label for label,_,_ in arm_specs], 'seeds':seeds, 'prompt_ids':ids,
        'memory_topologies':{'MEMORY_LOOP':{'seed':'boundary_mean_plus_1e-4_noise',
            'carry':'body_end_hidden','suffix_reads_memory':True,'null_cfg':'branch_local_boundary_memory'},
            'LAYERWISE_MEMORY_KV':{'seed':'distinct_native_prompt_content_at_original_positions',
            'carry':'native_layer_input_kv','read_layers':list(range(cfg.start_layer+1,cfg.end_layer)),
            'suffix_reads_memory':False,'null_cfg':'native_bypass'}},
        'probe_steps':probe_steps, 'probe_arm':probe_arm if probe_steps else None, 'diagnostics':args.diagnostics, 'stage':args.stage, 'shard':[args.shard_index,args.num_shards],
        'gpu':torch.cuda.get_device_name(), 'precision':'bfloat16', 'kernel':'native_flash_attention_and_legacy_masked_sdpa',
        'torch':torch.__version__, 'training':False, 'quality_status':'pending'}
    runfile = output/'run.json'
    if runfile.exists() and json.loads(runfile.read_text()) != provenance:
        raise ValueError('output directory contains an incompatible run; choose a new directory')
    runfile.write_text(json.dumps(provenance, indent=2)+'\n')
    manifest = output/'manifest.jsonl'
    completed = {}
    if manifest.exists():
        for r in read_jsonl(manifest):
            key=(r['arm'],r['prompt_id'],r['seed'])
            if key in completed: raise ValueError('duplicate resume record')
            if r['valid_file'] and sha256(r['path']) != r['image_sha256']: raise ValueError('resume image changed')
            completed[key]=r
    bundle = load_native(args.model_path,args.device,args.timestep_shift)
    jobs = [(i,s) for i in range(len(data)) for s in seeds]
    for arm,mode,rounds in arm_specs:
        runtime = InternalLoopRuntime(bundle.model, replace(cfg,mode=mode,extra_rounds=rounds),diagnostics=args.diagnostics)
        generator = T2IGenerator(bundle,runtime)
        try:
            for ordinal,(i,seed) in enumerate(jobs):
                if ordinal%args.num_shards != args.shard_index: continue
                if (arm,ids[i],seed) in completed:
                    existing=completed[(arm,ids[i],seed)]
                    if arm==probe_arm and probe_steps:
                        from qwen_latent_cot.bagel.memory_probe import validate_capture
                        validate_capture(existing['probe_capture'],existing['probe_capture_sha256'])
                    continue
                row=data[i]; shape=(int(row.get('height',args.image_size)),int(row.get('width',args.image_size)))
                name=f'{i:05d}_s{seed}.png'; imagepath=output/arm/name; imagepath.parent.mkdir(exist_ok=True)
                steps=args.matched_base_timesteps if arm=='BASE_MATCHED_LATENCY' else args.num_timesteps
                if arm==probe_arm and probe_steps:
                    from qwen_latent_cot.bagel.memory_probe import ProbeCapture
                    runtime.probe_capture = ProbeCapture(probe_steps)
                else: runtime.probe_capture = None
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
                    'timing_scope':'diagnostic_unwarmed' if probe_steps or args.diagnostics else 'engineering_single_generation_no_warmups', 'extra_rounds':rounds,
                    'body_pass_count_scope':'configured_active_branch; layerwise requires prompt content',
                    'writer_body_passes':rounds if mode.startswith('LAYERWISE') and cfg.memory_slots else 0,
                    'body_passes':1 if mode=='BASE' or rounds==0 or cfg.memory_slots==0 else 1+rounds,
                    'num_timesteps':steps}
                if runtime.probe_capture is not None:
                    if images is None:raise ValueError('probe export failed: invalid generated image')
                    from qwen_latent_cot.bagel.memory_probe import finalize_capture
                    captured = runtime.probe_capture.save(output/'probe_captures'/f'{i:05d}_s{seed}',
                        {'prompt_id':ids[i],'seed':seed,'prompt':row['prompt'],'noise_sha256':hashes[0],
                         'num_timesteps':steps,'start_layer':cfg.start_layer,'end_layer':cfg.end_layer,'extra_rounds':cfg.extra_rounds, 'arm':arm,
                         'read_layers':list(range(cfg.start_layer+(arm=='LAYERWISE_MEMORY_KV'),cfg.end_layer)),
                         'memory_carries_across_layers_in_generation':arm=='MEMORY_LOOP',
                         'memory_state':'native_layer_input_kv' if arm=='LAYERWISE_MEMORY_KV' else 'body_end_hidden',
                         'seed_reference':'selected_native_prompt_layer_input_kv' if arm=='LAYERWISE_MEMORY_KV' else 'strict_read_layer_input_kv'})
                    capture_path = output/'probe_captures'/f'{i:05d}_s{seed}'/'capture.json'
                    finalize_capture(capture_path)
                    record['probe_capture'] = str(capture_path)
                    record['probe_capture_sha256'] = sha256(capture_path)
                if args.diagnostics: record['memory_diagnostics'] = runtime.diagnostics
                with manifest.open('a') as f: f.write(json.dumps(record)+'\n'); f.flush()
                print(f'{arm} prompt={ids[i]} seed={seed} {elapsed:.2f}s peak={peak/2**30:.2f}GiB',flush=True)
        finally: runtime.close()
    print(f'Completed shard: {manifest}',flush=True)


if __name__=='__main__': main()
