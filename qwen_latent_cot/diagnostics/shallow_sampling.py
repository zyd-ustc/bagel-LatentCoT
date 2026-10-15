"""Prompt-only generation with trained continuous Memory and shallow GEN passes."""
import argparse
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
from types import MethodType
import torch
from .native import capture_gen, gen_memory_call
from .memory import rope_for
from .record import write_json
from .data import file_hash
from .schedule import native_times
from .feedback_source import prompt_source


@contextmanager
def shallow_loop(student, flow, lengths, positions, t, passes, end_layer=8):
    """Reset GEN entrance each pass; reset neutral writer seeds each feedback.

    UND runs continuously through all layers. Only the shallow window supplies
    GEN sources; the UND suffix continues to read prompt. Final GEN suffix reads the
    final bank once. Unconditional CFG stays on its native path.
    """
    decoder = student.runtime.decoder
    if passes < 2 or not 0 < end_layer <= len(decoder.layers):
        raise ValueError('require at least two passes and a valid shallow window')
    original = decoder.forward_inference
    cache = flow['past_key_values']
    prompt = prompt_source(flow, lengths, positions)
    source_layers = tuple(student.runtime.config.get('source_layers', range(end_layer)))
    if not source_layers or max(source_layers) >= end_layer:
        raise ValueError('shallow pass must reach every trained source layer')

    def forward(self, **kwargs):
        if kwargs.get('mode') != 'gen' or kwargs.get('past_key_values') is not cache:
            return original(**kwargs)
        if kwargs['update_past_key_values']:
            raise ValueError('shallow generation requires an immutable prompt cache')
        entrance = kwargs['packed_query_sequence']
        bank = None
        for iteration in range(passes):
            hidden = entrance.clone()
            query_positions = kwargs['packed_query_position_ids'] + (bank.slots if bank else 0)
            rope = rope_for(decoder, hidden, query_positions)
            inputs = {k: v for k, v in kwargs.items()
                      if k not in ('packed_query_sequence', 'packed_query_position_ids')}
            inputs['packed_query_position_embeddings'] = rope
            observer = capture_gen(decoder, cache, t=t, round_index=iteration,
                                   positions=query_positions, layers=source_layers,
                                   source_policy=student.runtime.config.get('visual_source_policy', 'full_query'))
            consume = (gen_memory_call(decoder, cache, bank, lengths, positions,
                                       wrap_decoder=False) if bank else nullcontext())
            # Capture only nonfinal shallow passes. The suffix runs exactly once.
            with consume, observer if iteration < passes-1 else nullcontext({}) as captured:
                limit = len(decoder.layers) if iteration == passes-1 else end_layer
                for layer in decoder.layers[:limit]:
                    hidden, _ = layer(packed_query_sequence=hidden, **inputs)
            if iteration < passes-1:
                bank = student.write(tuple(captured[i] for i in source_layers), t,
                                     prompt=prompt, source_layers=source_layers)
        result = torch.zeros_like(hidden)
        ti, ii = kwargs['packed_text_indexes'], kwargs['packed_vae_token_indexes']
        result[ti] = decoder.norm(hidden[ti])
        result[ii] = decoder.norm_moe_gen(hidden[ii])
        from ..bagel.modeling.bagel.qwen2_navit import BaseNavitOutputWithPast
        return BaseNavitOutputWithPast(packed_query_sequence=result, past_key_values=cache)

    try:
        decoder.forward_inference = MethodType(forward, decoder)
        yield
    finally:
        decoder.forward_inference = original


@torch.no_grad()
def sample(student, prompt, seed, passes, early, *, steps=None):
    runtime = student.runtime
    flow, lengths, positions = runtime.flow_inputs(prompt)
    model = runtime.model
    size = runtime.config['resolution']//model.latent_downsample
    width = model.latent_channel*model.latent_patch_size**2
    generator = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(size*size, width, generator=generator).to(runtime.device)
    times = native_times(runtime.config)
    records = []
    for index, (current, following) in enumerate(zip(times, times[1:])):
        if steps is not None and index >= steps:
            break
        active = passes > 1 and index < early
        end_layer = max(runtime.config.get('source_layers', range(min(8, len(runtime.decoder.layers)))))+1
        with shallow_loop(student, flow, lengths, positions, current, passes, end_layer) if active else nullcontext():
            _, velocity = runtime.velocity(x, current, flow)
        x = x-(current-following)*velocity
        records.append(dict(step=index, t=current, next_t=following,
                            shallow_passes=passes if active else 1,
                            passes=passes if active else 1, feedback_updates=passes-1 if active else 0,
                            conditional_gen_layer_calls=len(runtime.decoder.layers)+(passes-1)*end_layer if active else len(runtime.decoder.layers),
                            und_layer_calls=(passes-1)*len(runtime.decoder.layers) if active else 0))
    return runtime.image(x), records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--text-checkpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--passes', nargs='+', type=int, default=[2, 3, 4])
    parser.add_argument('--early', nargs='+', type=int, default=[10, 20])
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--smoke-steps', type=int)
    parser.add_argument('--selection-seed', type=int)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    if any(r < 2 for r in args.passes) or any(e <= 0 for e in args.early):
        parser.error('passes must be >=2 and early windows must be positive')
    from .shallow_review import prepare
    if args.prepare_only:
        prepare(args.prompts, args.output, args.count, args.passes, args.early, args.selection_seed)
        return
    output = Path(args.output)
    plan_path = output/'selection.json'
    if not plan_path.exists():
        if world != 1:
            raise ValueError('prepare the shared selection before launching workers')
        prepare(args.prompts, args.output, args.count, args.passes, args.early, args.selection_seed)
    plan = json.loads(plan_path.read_text())
    rows = plan['rows']
    from ..bagel.accelerator import set_device
    device = set_device(f'npu:{rank}')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = dict(saved['config'])
    from ..bagel.backbone import load_native
    from .runtime import DiagnosticRuntime
    from .visual_record import initialize_student
    runtime = DiagnosticRuntime(load_native(args.model, device, cfg['timestep_shift']), cfg, 'C')
    runtime.native_files = saved['native_files']
    student = initialize_student(runtime, args.text_checkpoint, args.checkpoint).eval()
    metadata = dict(checkpoint=args.checkpoint, checkpoint_sha256=file_hash(args.checkpoint),
                    checkpoint_step=saved['step'], trained_source_prompt_policy=cfg['source_prompt_policy'],
                    inference_source_prompt_policy='native_prompt', slots=student.writer.slots,
                    memory_initialization='fixed_neutral_prefill', writer_source_layers=cfg.get('source_layers', list(range(8))),
                    writer_suffix='prompt and memory self-attention', writer_reset='each_feedback',
                    gen_layer_window=cfg.get('source_layers', list(range(8))), rounds_meaning='total_GEN_passes',
                    seed=args.seed, sampling_config={k: cfg[k] for k in ('resolution','sampler_steps','timestep_shift','cfg_text_scale')},
                    world_size=world, smoke_steps=args.smoke_steps, selection_seed=plan['selection_seed'])
    write_json(output/f'worker_{rank:02d}.json', metadata)
    for index in range(rank, len(rows), world):
        row = rows[index]
        arms = [(1, 0, 'base')] + [(r, e, f'early{e}_R{r}')
                                    for e in plan['early'] for r in plan['passes']]
        for passes, early, suffix in arms:
            name = f'p{index:02d}_{suffix}'
            if (output/(name+'.png')).exists():
                raise FileExistsError(name)
            image, nodes = sample(student, row['prompt'], args.seed, passes, early, steps=args.smoke_steps)
            image.save(output/(name+'.png'))
            write_json(output/(name+'.json'), dict(metadata, prompt_index=index,
                       source_index=row['source_index'], arm=suffix,
                       prompt_id=row.get('prompt_id'), prompt=row['prompt'], early=early,
                       passes=passes, nodes=nodes))
            print(f'Rank {rank}: saved {name}', flush=True)
    if world == 1:
        from .shallow_review import render
        render(output)


if __name__ == '__main__':
    main()
