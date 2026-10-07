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
    p.add_argument('--arms', default='BASE,LAYERWISE_UND_STATE_REPLACE')
    p.add_argument('--loop-depths', help='Paired layerwise depths, e.g. 1,2,3; BASE generated once')
    p.add_argument('--loop-rounds', type=int, default=2, help='R: extra whole-body passes; R=0 is native bypass')
    p.add_argument('--start-layer', type=int, default=0); p.add_argument('--end-layer', type=int, default=8)
    p.add_argument('--progress-start', type=float, default=0); p.add_argument('--progress-end', type=float, default=1.)
    p.add_argument('--seeds', default='0'); p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--num-timesteps', type=int, default=50); p.add_argument('--timestep-shift', type=float, default=3)
    p.add_argument('--cfg-text-scale', type=float, default=4)
    p.add_argument('--cfg-renorm-type', choices=['global','text_channel'], default='global')
    p.add_argument('--max-prompts', type=int); p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--stage', choices=['engineering','development','evaluation'], default='engineering')
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
    if args.max_prompts is not None and args.max_prompts<1:raise ValueError('max prompts must be positive')
    if not 0 <= args.shard_index < args.num_shards: raise ValueError('invalid worker shard')
    if args.device.startswith('cuda'): torch.cuda.set_device(torch.device(args.device))
    seeds = [int(s) for s in args.seeds.split(',')]
    arms = args.arms.split(',')
    if len(set(seeds)) != len(seeds) or len(set(arms)) != len(arms) or not seeds:
        raise ValueError('duplicate/empty seeds or arms')
    if any(a not in MODES for a in arms) or 'BASE' not in arms:
        raise ValueError('arms require Base and the persistent UND Memory mode')
    data = read_jsonl(args.prompts)
    if args.max_prompts: data = data[:args.max_prompts]
    ids = [str(r.get('prompt_id',r.get('id',i))) for i,r in enumerate(data)]
    if not data or len(set(ids)) != len(ids): raise ValueError('empty/duplicate prompt IDs')
    depths=parse_depths(args.loop_depths)
    arm_specs=expand_arms(arms,depths,args.loop_rounds)
    cfg = LoopConfig(extra_rounds=max(depths) if depths else args.loop_rounds,
        start_layer=args.start_layer,end_layer=args.end_layer,
        progress_start=args.progress_start,progress_end=args.progress_end)
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    weights = Path(args.model_path)
    model_files = sorted(weights.glob('*.safetensors')) if not (weights/'ema.safetensors').exists() else [weights/'ema.safetensors', weights/'ae.safetensors']
    model_files += [weights/f for f in ('llm_config.json','vit_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt') if (weights/f).exists()]
    print('Hashing native weights and source...', flush=True)
    provenance = {'schema':9, 'architecture':'persistent_und_full_prompt_kv_v1', 'source_sha256':source_hash(ROOT),
        'model_sha256':{p.name:sha256(p) for p in model_files}, 'model_path':str(weights.resolve()),
        'benchmark_sha256':sha256(args.prompts), 'benchmark':str(Path(args.prompts).resolve()),
        'sampling':{'num_timesteps':args.num_timesteps,'actual_denoiser_calls':args.num_timesteps-1,
                    'timestep_shift':args.timestep_shift,'cfg_text_scale':args.cfg_text_scale,
                    'cfg_renorm_type':args.cfg_renorm_type,
                    'image_size':args.image_size},
        'loop':asdict(cfg), 'loop_depths':list(depths),
        'arm_configs':{label:asdict(replace(cfg,mode=mode,extra_rounds=rounds)) for label,mode,rounds in arm_specs},
        'arms':[label for label,_,_ in arm_specs], 'seeds':seeds, 'prompt_ids':ids,
        'memory_topologies':{'LAYERWISE_UND_STATE_REPLACE':{'seed':'all_native_prompt_layer_input_hidden_states',
            'capacity':'exact_native_prompt_length_per_sample',
            'carry':'per_body_layer_und_hidden; output_reprojected_at_same_layer',
            'special_token_hidden_and_kv':'pinned_to_native',
            'gen_round0':'native_prompt_body','gen_extra_rounds':'fixed_gen_entrance; memory_only_body',
            'first_body_layer':'dynamic_und_state_kv','writer_reads':'native_prompt_plus_current_gen_kv_plus_live_query_self_kv',
            'previous_memory_extra_kv':False,'suffix':'final_und_continuation_once; memory_only_gen_suffix_once',
            'state_lifetime':'one_denoiser_call_and_one_cfg_branch',
            'null_cfg':'native_bypass','prefix':'native_once'}},
        'diagnostics':args.diagnostics, 'stage':args.stage, 'shard':[args.shard_index,args.num_shards],
        'gpu':torch.cuda.get_device_name(), 'precision':'bfloat16', 'kernel':'native_flash_attention',
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
        arm_cfg=replace(cfg,mode=mode,extra_rounds=rounds)
        runtime = InternalLoopRuntime(bundle.model, arm_cfg,diagnostics=args.diagnostics)
        generator = T2IGenerator(bundle,runtime)
        try:
            for ordinal,(i,seed) in enumerate(jobs):
                if ordinal%args.num_shards != args.shard_index: continue
                if (arm,ids[i],seed) in completed:
                    continue
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
                    'timing_scope':'diagnostic_unwarmed' if args.diagnostics else 'engineering_single_generation_no_warmups', 'extra_rounds':rounds,
                    'body_pass_count_scope':'configured_active_branch; full Memory requires nonempty prompt cache',
                    'native_prompt_lengths':list(generator.prompt_lengths),
                    'memory_capacity_policy':'full_prompt',
                    'full_memory_lengths':list(generator.prompt_lengths) if mode!='BASE' else None,
                    'writer_body_passes':rounds if mode!='BASE' else 0,
                    'writer_suffix_passes':int(mode!='BASE' and rounds>0 and cfg.end_layer<len(bundle.model.language_model.model.layers)),
                    'body_passes':1+rounds,
                    'num_timesteps':steps}
                if args.diagnostics: record['memory_diagnostics'] = runtime.diagnostics
                with manifest.open('a') as f: f.write(json.dumps(record)+'\n'); f.flush()
                print(f'{arm} prompt={ids[i]} seed={seed} {elapsed:.2f}s peak={peak/2**30:.2f}GiB',flush=True)
        finally: runtime.close()
    print(f'Completed shard: {manifest}',flush=True)


if __name__=='__main__': main()
