"""Resident-model, independent-device native versus full-loop sampling workers."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--config', default='configs/interaction_sft.json')
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world', type=int, default=16)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    import numpy as np
    import torch
    from qwen_latent_cot.bagel.accelerator import set_device, synchronize, reset_peak_memory_stats, max_memory_allocated
    from qwen_latent_cot.diagnostics.inference import InferenceEngine
    from qwen_latent_cot.diagnostics.record import write_json
    from qwen_latent_cot.interaction.loop import active_updates
    config = json.loads(Path(args.config).read_text())
    rows = [json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    if not 0 <= args.rank < args.world:
        parser.error('rank must be in [0, world)')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    assigned = list(range(args.rank, len(rows), args.world))
    device = set_device('npu:0')
    started = time.perf_counter()
    write_json(output/f'worker_{args.rank:02d}.json', dict(status='loading', pid=os.getpid(), rank=args.rank,
               world=args.world, prompt_indexes=assigned, visible_devices=os.getenv('ASCEND_RT_VISIBLE_DEVICES')))
    print(f'WORKER {args.rank}: loading model, assigned={assigned}', flush=True)
    engine = InferenceEngine.from_parent(args.model, args.parent, config, device)
    runtime = engine.runtime
    metadata = dict(parent=args.parent, parent_step=engine.student.parent['step'], model=args.model,
                    config=config, seed=args.seed, native_baseline='original_BAGEL_UND_no_adapters',
                    initialized_only=True, memory_strength='full', bridge_strength=float(engine.student.bridge.strength),
                    blocks=len(runtime.decoder.layers), active_updates=active_updates(runtime.config),
                    sampling={k: runtime.config[k] for k in ('resolution','sampler_steps','timestep_shift',
                               'cfg_text_scale','cfg_renorm_type','cfg_renorm_min')})
    write_json(output/f'worker_{args.rank:02d}.json', dict(metadata,status='sampling',pid=os.getpid(),
               rank=args.rank,prompt_indexes=assigned,model_load_seconds=time.perf_counter()-started))
    print(f'WORKER {args.rank}: model ready; {json.dumps(metadata)}', flush=True)
    for index in assigned:
        prompt = rows[index]['prompt']
        record = dict(prompt_index=index,prompt=prompt,seed=args.seed,rank=args.rank,arms={})
        images = {}
        for arm, native in (('native',True),('loop',False)):
            filename = f'prompt_{index:02d}_{arm}.png'
            if (output/filename).exists():
                raise FileExistsError(output/filename)
            reset_peak_memory_stats(device)
            began = time.perf_counter()
            def progress(node):
                if node['step'] == 1 or node['step'] % 5 == 0 or node['next_t'] == 0:
                    print(f"WORKER {args.rank} prompt={index} arm={arm} step={node['step']} loop={node['loop_active']}", flush=True)
            image, nodes = engine.generate(prompt,args.seed,native=native,on_step=progress)
            synchronize(device)
            image.save(output/filename)
            write_json(output/f'prompt_{index:02d}_{arm}.json', nodes)
            images[arm]=np.asarray(image,dtype=np.float32)
            record['arms'][arm]=dict(image=filename,seconds=time.perf_counter()-began,
                  peak_memory_gib=max_memory_allocated(device)/1024**3,updates=len(nodes),
                  loop_updates=sum(node['loop_active'] for node in nodes))
            write_json(output/f'prompt_{index:02d}.partial.json', record)
            print(f'ARM_COMPLETE prompt={index} arm={arm} {json.dumps(record["arms"][arm])}',flush=True)
        record['mean_abs_rgb_255']=float(np.abs(images['loop']-images['native']).mean())
        write_json(output/f'prompt_{index:02d}.json',record)
        (output/f'prompt_{index:02d}.partial.json').unlink()
        print(f'PAIR_COMPLETE prompt={index} mae={record["mean_abs_rgb_255"]:.4f}',flush=True)
    write_json(output/f'worker_{args.rank:02d}.json',dict(metadata,status='complete',pid=os.getpid(),
               rank=args.rank,prompt_indexes=assigned,seconds=time.perf_counter()-started))
    print(f'WORKER_COMPLETE rank={args.rank}',flush=True)


if __name__ == '__main__':
    main()
