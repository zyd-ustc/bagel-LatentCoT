"""Paired native image editing with/without image-grounded textual feedback."""
import json
from pathlib import Path
import time
import torch
from PIL import Image
from .io import read_jsonl,sha256
from ..bagel.inferencer import T2IGenerator,InvalidGeneratedImage
from ..bagel.feedback import NativeFeedback


def generate_feedback(bundle,plan,args,completed,output,manifest):
    data=read_jsonl(plan['benchmark'])[:len(plan['prompt_ids'])]
    generator=T2IGenerator(bundle);editor=NativeFeedback(bundle)
    sampling=plan['sampling'];jobs=[(i,s) for i in range(len(data)) for s in plan['seeds']]
    for ordinal,(i,seed) in enumerate(jobs):
        if ordinal%args.num_shards!=args.shard_index:continue
        row=data[i];pid=plan['prompt_ids'][i];name=f'{i:05d}_s{seed}.png'
        shape=(int(row.get('height',sampling['image_size'])),int(row.get('width',sampling['image_size'])))
        if shape!=(512,512):raise ValueError('native feedback pilot currently binds 512x512 source images')
        base=completed.get(('BASE',pid,seed))
        if base is None:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            images,hashes=generator.generate([row['prompt']],[shape],[seed],num_timesteps=sampling['num_timesteps'],
                timestep_shift=sampling['timestep_shift'],cfg_text_scale=sampling['cfg_text_scale'],cfg_renorm_type=sampling['cfg_renorm_type'])
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            path=output/'BASE'/name;path.parent.mkdir(exist_ok=True);images[0].save(path)
            base={'arm':'BASE','prompt_id':pid,'index':i,'prompt':row['prompt'],'seed':seed,'bucket':row.get('bucket','unclassified'),
                'height':shape[0],'width':shape[1],'path':str(path),'image_sha256':sha256(path),'noise_sha256':hashes[0],
                'valid_file':True,'generation_seconds':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'timing_scope':'engineering_end_to_end_from_original_prompt','stage_seconds':{'base':elapsed},'extra_rounds':0}
            with manifest.open('a') as f:f.write(json.dumps(base)+'\n')
            completed[('BASE',pid,seed)]=base
            print(f'BASE prompt={pid} seed={seed} {elapsed:.2f}s',flush=True)
        if not base['valid_file']:raise ValueError('feedback requires a valid source Base image')
        with Image.open(base['path']) as image:source=image.convert('RGB')
        generic=('Edit the supplied image to satisfy the original request. Preserve all already-correct objects, '
            'attributes, relationships, composition and style. Make only necessary changes.\nOriginal request:\n'+row['prompt'])
        for arm in ('GENERIC_EDIT','FEEDBACK_EDIT'):
            if (arm,pid,seed) in completed:continue
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            feedback=None;observe_seconds=0.;error=None;result=None;meta={};noise_hash=base['noise_sha256']
            if arm=='FEEDBACK_EDIT':
                feedback=editor.observe(source,row['prompt'],plan['config']['feedback_max_tokens'])
                torch.cuda.synchronize();observe_seconds=time.perf_counter()-start
                trace=output/'feedback'/f'{i:05d}_s{seed}.json';trace.parent.mkdir(exist_ok=True)
                trace.write_text(json.dumps(dict(feedback,source_image_sha256=base['image_sha256'],
                    prompt=row['prompt'],generation_seed=seed,observation_verified=False),ensure_ascii=False,indent=2)+'\n')
                if not feedback['valid_format']:error=feedback.get('error','invalid feedback')
            instruction=generic if feedback is None else (generic+'\nImage-grounded feedback (check uncertain claims against the image):\n'+feedback['raw']+
                '\nUse the confident discrepancies to make a minimal edit. Preserve the other content. Do not render this feedback as text in the image.')
            edit_start=time.perf_counter()
            if error is None:
                try:result,noise_hash,meta=editor.edit(source,instruction,seed,sampling,plan['config']['edit_sampling'])
                except InvalidGeneratedImage as exc:error=str(exc)
            torch.cuda.synchronize();edit_seconds=time.perf_counter()-edit_start
            path=output/arm/name;path.parent.mkdir(exist_ok=True)
            if result is not None:result.save(path)
            if noise_hash!=base['noise_sha256']:raise ValueError('edit arms must use the bound common initial noise')
            record=dict(base,arm=arm,path=str(path),image_sha256=sha256(path) if result is not None else None,
                valid_file=result is not None,decode_error=error,source_image_sha256=base['image_sha256'],source_image_path=base['path'],
                edit_noise_sha256=noise_hash,feedback_text=feedback['raw'] if feedback else None,
                feedback_format_valid=feedback['valid_format'] if feedback else None,
                feedback_tokens=feedback['token_ids'] if feedback else None,
                native_edit_context=meta,generation_seconds=base['generation_seconds']+observe_seconds+edit_seconds,
                stage_seconds={'base':base['generation_seconds'],'observe':observe_seconds,'edit':edit_seconds},
                peak_allocated_bytes=max(base['peak_allocated_bytes'],torch.cuda.max_memory_allocated()))
            with manifest.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            completed[(arm,pid,seed)]=record
            print(f'{arm} prompt={pid} seed={seed} valid={record["valid_file"]} total={record["generation_seconds"]:.2f}s',flush=True)
