"""Resident-model Stage A generation: full prompts and explicit information gaps."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

ARM_LABELS = dict(native_full='原生 BAGEL · 完整 Prompt',loop_full='Stage A Loop · 完整 Prompt',
                  native_partial='原生 BAGEL · 删减 Prompt',loop_partial='Loop · GEN 删减 / UND 完整',
                  wrong_reader='Loop · GEN 删减 / UND 错配')


def load_cases(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Require nonempty cases with unique IDs')
    for row in rows:
        for key in ('full_prompt','partial_prompt'):
            if not row[key].strip():raise ValueError(f'Empty {key}: {row["id"]}')
        if row.get('removed_clauses') and row['full_prompt']==row['partial_prompt']:
            raise ValueError('A gap case must change the GEN prompt')
    return rows


def arms_for(row):
    return ['native_full','loop_full']+(['native_partial','loop_partial','wrong_reader'] if row.get('removed_clauses') else [])


def arm_inputs(row, arm, donor):
    if arm not in ARM_LABELS:raise ValueError('Unknown sampling arm: '+arm)
    prompt=row['partial_prompt'] if arm in ('native_partial','loop_partial','wrong_reader') else row['full_prompt']
    native=arm.startswith('native_')
    reader=None if native else donor['full_prompt'] if arm=='wrong_reader' else row['full_prompt']
    return dict(prompt=prompt,native=native,reader_prompt=reader)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--checkpoint',required=True,help='Pinned Stage A canvas v2 checkpoint file; not a training directory')
    parser.add_argument('--prompts',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--rank',type=int,required=True)
    parser.add_argument('--world',type=int,default=16)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--device',default='npu:0')
    args=parser.parse_args()
    from qwen_latent_cot.bagel.accelerator import set_device,synchronize,reset_peak_memory_stats,max_memory_allocated
    from qwen_latent_cot.diagnostics.inference import InferenceEngine,load_runtime
    from qwen_latent_cot.interaction.checkpoint import read,STAGE_A_KIND
    from qwen_latent_cot.diagnostics.record import write_json
    from qwen_latent_cot.interaction.sampling import sample
    from qwen_latent_cot.interaction.loop import active_updates
    if not 0<=args.rank<args.world:parser.error('rank must be in [0, world)')
    if not Path(args.checkpoint).is_file():parser.error('Pass a pinned checkpoint file')
    rows=load_cases(args.prompts)
    assigned=list(range(args.rank,len(rows),args.world))
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    worker=output/f'worker_{args.rank:02d}.json'
    started=time.perf_counter()
    meta=dict(rank=args.rank,world=args.world,pid=os.getpid(),checkpoint=args.checkpoint,
              seed=args.seed,prompt_indexes=assigned,visible_devices=os.getenv('ASCEND_RT_VISIBLE_DEVICES'))
    try:
        write_json(worker,dict(meta,status='loading'))
        device=set_device(args.device)
        saved=read(args.checkpoint)
        if saved.get('kind')!=STAGE_A_KIND or saved.get('stage')!='stage_a':raise ValueError('Require a trained Stage A canvas v2 checkpoint')
        runtime=load_runtime(args.model,saved['parent'],device)
        engine=InferenceEngine(runtime,saved['parent'],saved['config'],saved=saved)
        meta.update(checkpoint_step=saved['step'],checkpoint_kind=saved['kind'],model=args.model,
                    config=saved['config'],sampling={k:runtime.config[k] for k in ('resolution','sampler_steps','timestep_shift','cfg_text_scale','cfg_renorm_type','cfg_renorm_min')},
                    blocks=len(runtime.decoder.layers),active_updates=active_updates(runtime.config),
                    bridge_strength=float(engine.student.bridge.strength),velocity_scale=engine.student.bridge.vel_scale.detach().float().cpu().tolist(),
                    native_baseline='original_BAGEL_UND_no_adapters',wrong_reader_control='another case full prompt; not an independent-image Memory swap')
        del saved
        write_json(worker,dict(meta,status='sampling',load_seconds=time.perf_counter()-started))
        for index in assigned:
            row=rows[index]
            donor=next(r for r in rows[index+1:]+rows[:index] if r['full_prompt']!=row['full_prompt'])
            record=dict(row,prompt_index=index,seed=args.seed,rank=args.rank,checkpoint=args.checkpoint,
                        checkpoint_step=meta['checkpoint_step'],donor_id=donor['id'],donor_prompt=donor['full_prompt'],arms={})
            for arm in arms_for(row):
                filename=f'case_{index:03d}_{arm}.png'
                if (output/filename).exists():raise FileExistsError(output/filename)
                inputs=arm_inputs(row,arm,donor)
                reset_peak_memory_stats(device)
                began=time.perf_counter()
                def progress(node):
                    if node['step']==1 or node['step']%10==0 or node['next_t']==0:
                        print(f'PROGRESS rank={args.rank} case={index} arm={arm} step={node["step"]}',flush=True)
                image,nodes=sample(engine.student,inputs['prompt'],args.seed,native=inputs['native'],
                                   reader_prompt=inputs['reader_prompt'],on_step=progress)
                synchronize(device)
                image.save(output/filename)
                write_json(output/f'case_{index:03d}_{arm}.json',nodes)
                record['arms'][arm]=dict(image=filename,label=ARM_LABELS[arm],**inputs,seconds=time.perf_counter()-began,
                                        peak_memory_gib=max_memory_allocated(device)/1024**3,
                                        updates=len(nodes),loop_updates=sum(n['loop_active'] for n in nodes))
                write_json(output/f'case_{index:03d}.partial.json',record)
                print(f'ARM_COMPLETE rank={args.rank} case={index} arm={arm}',flush=True)
            write_json(output/f'case_{index:03d}.json',record)
            (output/f'case_{index:03d}.partial.json').unlink()
        write_json(worker,dict(meta,status='complete',seconds=time.perf_counter()-started))
    except Exception:
        write_json(worker,dict(meta,status='failed',error=traceback.format_exc()))
        raise


if __name__=='__main__':main()
