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
def shallow_loop(student, flow, lengths, positions, t, passes, end_layer=None, *,
                 masked=False, donor_bank=None, on_memory=None):
    """Reset GEN entrance each pass; reset neutral writer seeds each feedback.

    Nonfinal passes run the selected GEN prefix; the final pass runs all layers.
    UND always runs all layers, reading prompt everywhere and GEN in the prefix.
    Unconditional CFG stays on its native path.
    """
    decoder = student.runtime.decoder
    end_layer = len(decoder.layers) if end_layer is None else end_layer
    if passes < 2 or not 0 < end_layer <= len(decoder.layers):
        raise ValueError('require at least two passes and a valid GEN prefix depth')
    original = decoder.forward_inference
    cache = flow['past_key_values']
    prompt = prompt_source(flow, lengths, positions)
    source_layers = tuple(range(end_layer))

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
                                       masked=masked, wrap_decoder=False) if bank else nullcontext())
            # Capture only nonfinal shallow passes. The suffix runs exactly once.
            adapter_scope = student.consumer_context() if bank is not None and hasattr(student, 'consumer_context') else nullcontext
            with adapter_scope(), consume, observer if iteration < passes-1 else nullcontext({}) as captured:
                limit = len(decoder.layers) if iteration == passes-1 else end_layer
                for layer in decoder.layers[:limit]:
                    hidden, _ = layer(packed_query_sequence=hidden, **inputs)
            if iteration < passes-1:
                bank = student.write(tuple(captured[i] for i in source_layers), t,
                                     prompt=prompt, source_layers=source_layers)
                if donor_bank is not None:
                    if passes != 2:
                        raise ValueError('donor memory currently requires R2')
                    bank = donor_bank
                if on_memory is not None:
                    on_memory(bank)
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
def sample(student, prompt, seed, passes, early, *, steps=None, loop_layers=None,
           memory_mode='correct', donor_seed=None):
    runtime = student.runtime
    flow, lengths, positions = runtime.flow_inputs(prompt)
    model = runtime.model
    size = runtime.config['resolution']//model.latent_downsample
    width = model.latent_channel*model.latent_patch_size**2
    generator = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(size*size, width, generator=generator).to(runtime.device)
    if memory_mode not in ('correct', 'masked', 'swap'):
        raise ValueError('select correct/masked/swap memory')
    donor_x = None
    if memory_mode == 'swap':
        if donor_seed is None or donor_seed == seed or passes != 2:
            raise ValueError('swap requires a distinct donor seed and R2')
        donor_x = torch.randn(size*size, width, generator=torch.Generator(device='cpu').manual_seed(donor_seed)).to(runtime.device)
    times = native_times(runtime.config)
    depth=len(runtime.decoder.layers)
    end_layer=depth if loop_layers in (None,'full') else int(loop_layers)
    if not 0 < end_layer <= depth:raise ValueError('loop layers must fit the GEN depth')
    active_steps=len(times)-1 if early=='full' else int(early)
    records = []
    for index, (current, following) in enumerate(zip(times, times[1:])):
        if steps is not None and index >= steps:
            break
        active = passes > 1 and index < active_steps
        donor_bank = None
        swap_metadata = {}
        if active and donor_x is not None:
            # The donor follows its OWN correct-memory trajectory. Its bank is
            # written at the same t before advancing either state.
            from .data import tensor_hash
            swap_metadata = dict(donor_seed=donor_seed, donor_t=current,
                                 recipient_state_hash=tensor_hash(x), donor_state_hash=tensor_hash(donor_x),
                                 donor_policy=('interaction_memory' if student.adapters_enabled else 'parent_memory')
                                 if hasattr(student, 'adapters_enabled') else 'c2_memory', donor_prompt=prompt)
            banks = []
            with shallow_loop(student, flow, lengths, positions, current, 2, end_layer,
                              on_memory=banks.append):
                _, donor_velocity = runtime.velocity(donor_x, current, flow)
            donor_bank = banks[0]
            donor_x = donor_x-(current-following)*donor_velocity
        with shallow_loop(student, flow, lengths, positions, current, passes, end_layer,
                          masked=memory_mode == 'masked', donor_bank=donor_bank) if active else nullcontext():
            _, velocity = runtime.velocity(x, current, flow)
        x = x-(current-following)*velocity
        records.append(dict(step=index, t=current, next_t=following, memory_active=active, loop_layers=end_layer,
                            shallow_passes=passes if active else 1,
                            passes=passes if active else 1, feedback_updates=passes-1 if active else 0,
                            conditional_gen_layer_calls=len(runtime.decoder.layers)+(passes-1)*end_layer if active else len(runtime.decoder.layers),
                            und_layer_calls=(passes-1)*len(runtime.decoder.layers) if active else 0,
                            memory_mode=memory_mode if active else 'none', memory_visible=active and memory_mode != 'masked',
                            donor_conditional_gen_layer_calls=len(runtime.decoder.layers)+end_layer if active and donor_bank is not None else 0,
                            donor_und_layer_calls=len(runtime.decoder.layers) if active and donor_bank is not None else 0,
                            **swap_metadata))
    return runtime.image(x), records


def window(value):
    value=str(value).lower()
    if value=='full':return value
    try:result=int(value.removeprefix('early'))
    except ValueError:raise argparse.ArgumentTypeError('use early10, 10, 20, 30 or full')
    if result<=0:raise argparse.ArgumentTypeError('early update count must be positive')
    return result


def layer_depth(value):
    if str(value).lower()=='full':return 'full'
    try:result=int(value)
    except ValueError:raise argparse.ArgumentTypeError('use a GEN block count or full')
    if result<=0:raise argparse.ArgumentTypeError('GEN block count must be positive')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--passes', nargs='+', type=int, default=[2, 3, 4])
    parser.add_argument('--early', nargs='+', type=window, default=[10,20,30,'full'],
                        help='first N Euler updates: 10 20 30 full; early10 also accepted')
    parser.add_argument('--loop-layers', nargs='+', type=layer_depth, default=['full'],
                        help='repeated prefix GEN block counts, e.g. 8 16 28 or full')
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--smoke-steps', type=int)
    parser.add_argument('--selection-seed', type=int)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    if any(r < 2 for r in args.passes):
        parser.error('passes are total GEN passes and must be >=2')
    from .shallow_review import prepare
    if args.prepare_only:
        prepare(args.prompts, args.output, args.count, args.passes, args.early, args.selection_seed, args.loop_layers)
        return
    output = Path(args.output)
    plan_path = output/'selection.json'
    if not plan_path.exists():
        if world != 1:
            raise ValueError('prepare the shared selection before launching workers')
        prepare(args.prompts, args.output, args.count, args.passes, args.early, args.selection_seed, args.loop_layers)
    plan = json.loads(plan_path.read_text())
    rows = plan['rows']
    from ..bagel.accelerator import set_device
    device = set_device(f'npu:{rank}')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = dict(saved['config'])
    checkpoint_step=saved['step'];checkpoint_stage=saved['stage'];native_files=saved['native_files']
    del saved
    from ..bagel.backbone import load_native
    from .runtime import DiagnosticRuntime
    from .visual_record import initialize_student
    runtime = DiagnosticRuntime(load_native(args.model, device, cfg['timestep_shift']), cfg, 'C')
    runtime.native_files = native_files
    student = initialize_student(runtime, visual_init=args.checkpoint).eval().requires_grad_(False)
    metadata = dict(checkpoint=args.checkpoint, checkpoint_sha256=file_hash(args.checkpoint),
                    checkpoint_step=checkpoint_step, trained_stage=checkpoint_stage,
                    writer_source_policy='prompt_every_layer_gen_in_repeated_prefix',
                    inference_source_prompt_policy='native_prompt', slots=student.writer.slots,
                    memory_initialization='fixed_neutral_prefill', trained_source_layers=cfg['source_layers'], writer_und_depth=len(runtime.decoder.layers),
                    writer_suffix='prompt_only_outside_GEN_prefix', writer_reset='each_feedback',
                    loop_layers=plan['loop_layers'], gen_layer_window='per_arm_prefix', rounds_meaning='total_GEN_passes',
                    seed=args.seed, euler_updates=len(native_times(cfg))-1, sampling_config={k: cfg[k] for k in ('resolution','sampler_steps','timestep_shift','cfg_text_scale','cfg_renorm_type','cfg_renorm_min')},
                    world_size=world, smoke_steps=args.smoke_steps, selection_seed=plan['selection_seed'])
    write_json(output/f'worker_{rank:02d}.json', metadata)
    for index in range(rank, len(rows), world):
        row = rows[index]
        for arm in plan['arms']:
            passes,early,loop_layers,suffix=(arm[k] for k in ('passes','early','loop_layers','name'))
            name = f'p{index:02d}_{suffix}'
            if (output/(name+'.png')).exists():
                raise FileExistsError(name)
            image, nodes = sample(student, row['prompt'], args.seed, passes, early, steps=args.smoke_steps, loop_layers=loop_layers)
            image.save(output/(name+'.png'))
            write_json(output/(name+'.json'), dict(metadata, prompt_index=index,
                       source_index=row['source_index'], arm=suffix,
                       prompt_id=row.get('prompt_id'), prompt=row['prompt'], early=early,
                       loop_layers=loop_layers, resolved_loop_layers=len(runtime.decoder.layers) if loop_layers=='full' else loop_layers,
                       writer_source_layers=list(range(len(runtime.decoder.layers) if loop_layers=='full' else loop_layers)),
                       passes=passes, nodes=nodes))
            print(f'Rank {rank}: saved {name}', flush=True)
    if world == 1:
        from .shallow_review import render
        render(output)


if __name__ == '__main__':
    main()
