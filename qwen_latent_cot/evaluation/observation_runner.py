"""Five paired arms: Base, static/observed native Memory at steps 9 and 19."""
import json
import time
from pathlib import Path
import torch
from .io import read_jsonl,sha256,validate_observation_record,validate_observation_pair
from ..bagel.inferencer import T2IGenerator,InvalidGeneratedImage
from ..bagel.observation_memory import ObservationMemoryGenerator
from ..bagel.accelerator import synchronize,reset_peak_memory_stats,max_memory_allocated


def generate_observation(bundle,plan,args,completed,output,manifest):
    if len(bundle.model.language_model.model.layers)!=plan['native_depth']:
        raise ValueError('loaded native decoder depth differs from plan')
    for record in completed.values():validate_observation_record(record)
    device=next(bundle.model.parameters()).device
    data=read_jsonl(plan['benchmark'])[:len(plan['prompt_ids'])];sampling=plan['sampling']
    jobs=[(i,s) for i in range(len(data)) for s in plan['seeds']]
    for arm,setting in plan['arm_configs'].items():
        for ordinal,(i,seed) in enumerate(jobs):
            if ordinal%args.num_shards!=args.shard_index:continue
            pid=plan['prompt_ids'][i]
            if (arm,pid,seed) in completed:continue
            row=data[i];shape=(int(row.get('height',sampling['image_size'])),int(row.get('width',sampling['image_size'])))
            name=f'{i:05d}_s{seed}';trace=output/'traces'/arm/name
            questions=[q for q,_ in row['vqa_list'][:plan['config']['probe_questions_per_image']]]
            generator=(T2IGenerator(bundle) if arm=='BASE' else ObservationMemoryGenerator(bundle,
                setting['observation_step'],setting['observe_image'],trace,questions,
                plan['config']['probe_max_tokens'],plan['config']['edit_sampling']))
            synchronize(device);reset_peak_memory_stats(device);begin=time.perf_counter()
            error=None
            try:
                images,hashes=generator.generate([row['prompt']],[shape],[seed],num_timesteps=sampling['num_timesteps'],
                    timestep_shift=sampling['timestep_shift'],cfg_text_scale=sampling['cfg_text_scale'],cfg_renorm_type=sampling['cfg_renorm_type'])
            except InvalidGeneratedImage as exc:images=None;hashes=exc.noise_hashes;error=str(exc)
            synchronize(device);elapsed=time.perf_counter()-begin
            path=output/arm/(name+'.png');path.parent.mkdir(exist_ok=True)
            if images is not None:images[0].save(path)
            events=getattr(generator,'events',[])
            if events:
                trace.mkdir(parents=True,exist_ok=True)
                (trace/'event.json').write_text(json.dumps({'arm':arm,'prompt_id':pid,'prompt':row['prompt'],
                    'seed':seed,'initial_noise_sha256':hashes[0],'events':events},ensure_ascii=False,indent=2)+'\n')
            record={'arm':arm,'prompt_id':pid,'index':i,'prompt':row['prompt'],'seed':seed,'bucket':row.get('bucket','unclassified'),
                'height':shape[0],'width':shape[1],'path':str(path),'image_sha256':sha256(path) if images is not None else None,
                'noise_sha256':hashes[0],'valid_file':images is not None,'decode_error':error,'generation_seconds':elapsed,
                'peak_allocated_bytes':max_memory_allocated(device),'timing_scope':'engineering_with_preview_and_probes',
                'probe_seconds':sum(e['probe_seconds'] for e in events),'observation_events':events,
                'observation_step':setting.get('observation_step'),'observe_image':setting.get('observe_image'),
                'extra_rounds':0 if arm=='BASE' else 1,'memory_capacity_policy':'full_original_prompt',
                'memory_update_count':len(events),'memory_lifecycle':'one_selected_call_only',
                'writer_depth':0 if arm=='BASE' else plan['native_depth'],'image_context_policy':'full_native_VAE_plus_ViT',
                'probe_uses_actual_image_labels':False,'manual_probe_labels':'pending',
                'observation_teacher_text_generated':False}
            validate_observation_record(record)
            if arm.endswith('_OBSERVED'):
                other=completed.get((arm.replace('_OBSERVED','_STATIC'),pid,seed))
                if other is None:raise ValueError('static paired arm missing before observed generation')
                validate_observation_pair(record,other)
            with manifest.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            completed[(arm,pid,seed)]=record
            print(f'{arm} prompt={pid} seed={seed} valid={record["valid_file"]} total={elapsed:.2f}s',flush=True)
