"""G-base/layout/fixed/text/T*/V*/swap staircase at fixed state, once only."""
from copy import deepcopy
import random
import time
import torch
from .native import device_inputs
from .record import load_checkpoint, append_json, write_json, FrozenGuard
from .data import velocity_stats, tensor_hash
from .numerics import autocast_for
from .schedule import nearest_node, native_suffix
from .memory import KV, MemoryBank
from ..bagel.accelerator import seeded_context, synchronize, max_memory_allocated


def supplemental_flow(runtime, flow, lengths, positions, caption):
    """Append native text prefill to intact original prompt KV."""
    cache = deepcopy(flow['past_key_values'])
    inputs, new_lengths, new_positions = runtime.model.prepare_prompts(lengths, positions, [caption],
                                          runtime.bundle.tokenizer, runtime.bundle.token_ids)
    with autocast_for(runtime.device):
        cache = runtime.model.forward_cache_update_text(cache, **device_inputs(inputs, runtime.device))
    size = runtime.config['resolution']
    with seeded_context(runtime.device, 0):
        prepared = runtime.model.prepare_vae_latent(new_lengths, new_positions, [(size, size)], runtime.bundle.token_ids)
    prepared.pop('packed_init_noises')
    # Keep native unconditional CFG inputs; only conditional text is augmented.
    out = dict(flow); out.update(device_inputs(prepared, runtime.device)); out['past_key_values'] = cache
    return out


@torch.no_grad()
def diagnose_gen(runtime, rows, output, text_checkpoint, visual_checkpoint, continuation=False):
    if not text_checkpoint:
        raise ValueError('GEN staircase requires a trained C1 text checkpoint')
    runtime.writer.eval().requires_grad_(False)
    guard = FrozenGuard(runtime.model, runtime.bundle.vae)
    by_id = {r['sample_id']: r for r in rows}; blind = []; result_rows = []
    fixed = runtime.writer.fixed_bank(runtime.seeds)
    zero = MemoryBank(tuple(KV(kv.layer, torch.zeros_like(kv.keys), torch.zeros_like(kv.values),
                              kv.lengths, False) for kv in fixed.layers), 'fixed', fixed.slots)
    for row in rows:
        states = [(t, seed) for t in runtime.config['eval_t'] if t < 1 for seed in runtime.config['noise_seeds']]
        if row.get('state'):
            saved = torch.load(row['state'], map_location='cpu', weights_only=True)
            states = [(float(saved['t']), int(saved['noise_seed']))]
        for state_index, (t, seed) in enumerate(states):
            tic = time.perf_counter()
            requested_t = t; continuation_step = None
            if continuation and not row.get('state'):
                continuation_step, t = nearest_node(runtime.config, requested_t)
            if continuation:
                remaining = native_suffix(runtime.config, t)
            x, state_flow, _, _, metadata = runtime.state(row, t, seed)
            metadata.update(requested_t=requested_t, continuation_step_index=continuation_step if continuation_step is not None else metadata['step_index'],
                            continuation_times=[t]+remaining if continuation else None,
                            continuation_policy='nearest_native_node_rebuild_x_t' if continuation else 'local_exact_t')
            # Controlled GEN task keeps the underspecified ORIGINAL dataset prompt.
            # Caption writer source remains neutral; it cannot see this prompt/caption.
            flow, lengths, positions = runtime.flow_inputs(row['prompt'])
            if metadata['state_source'] == 'trajectory':
                flow = state_flow
            native = runtime.velocity(x, t, flow)
            repeat = runtime.velocity(x, t, flow)
            layout = runtime.gen_arm(x, t, flow, fixed, lengths, positions, masked=True)
            shifted_flow = dict(flow)
            shifted_flow['packed_position_ids'] = flow['packed_position_ids']+fixed.slots
            shifted_native = runtime.velocity(x, t, shifted_flow)
            parity = {kind: velocity_stats(a, b) for kind, a, b in
                      zip(('conditional', 'cfg'), layout, shifted_native)}
            arms = {'G-base': native, 'G-layout': layout,
                    'G-fixed': runtime.gen_arm(x, t, flow, zero, lengths, positions),
                    'G-neutral': runtime.gen_arm(x, t, flow, fixed, lengths, positions),
                    'G-text': runtime.velocity(x, t, supplemental_flow(runtime, flow, lengths, positions, row['caption']))}
            load_checkpoint(text_checkpoint, runtime, 'C1')
            text_source = runtime.text(row['caption'])[0]
            text_bank = runtime.memory(text_source)
            arms['G-T'+runtime.writer.topology] = runtime.gen_arm(x, t, flow, text_bank, lengths, positions)
            banks = {'text': text_bank, 'fixed_zero': zero, 'fixed_neutral': fixed}
            other = by_id.get(row.get('swap_id'))
            if row.get('swap_id') and other is None:
                raise ValueError('GEN evaluation slice omitted a swap partner')
            if other:
                swap_text = runtime.memory(runtime.text(other['caption'])[0])
                arms['G-swap-T'+runtime.writer.topology] = runtime.gen_arm(x, t, flow, swap_text, lengths, positions)
                banks['text_swap'] = swap_text
            if visual_checkpoint:
                source = runtime.visual_source(x, t, state_flow)
                load_checkpoint(visual_checkpoint, runtime, 'C2')
                visual_bank = runtime.memory(source)
                banks['visual'] = visual_bank
                arms['G-V'+runtime.writer.topology] = runtime.gen_arm(x, t, flow, visual_bank, lengths, positions)
                if other:
                    ox, oflow, _, _, ometa = runtime.state(other, t, seed)
                    if (ometa['prompt'], ometa['t']) != (metadata['prompt'], metadata['t']):
                        raise ValueError('visual swap must keep source prompt and actual t')
                    swap_visual = runtime.memory(runtime.visual_source(ox, t, oflow))
                    banks['visual_swap'] = swap_visual
                    arms['G-swap-V'+runtime.writer.topology] = runtime.gen_arm(x, t, flow, swap_visual, lengths, positions)
            repeat_error = {kind: velocity_stats(a, b) for kind, a, b in zip(('conditional', 'cfg'), repeat, native)}
            key = f'{row["sample_id"]}_{state_index:03d}'
            # Safe IDs are enforced by CLI; hashes bind tensor content to records.
            torch.save(dict(x_t=x.cpu(), t=t, metadata=metadata,
                            banks={name: [dict(layer=kv.layer, k=kv.keys.cpu(), v=kv.values.cpu(),
                                         lengths=kv.lengths, rotated=False) for kv in bank.layers] for name, bank in banks.items()},
                            velocities={arm: {'conditional': v[0].cpu(), 'cfg': v[1].cpu()} for arm, v in arms.items()}),
                       output/(key+'.pt'))
            for arm, (conditional, final) in arms.items():
                record = dict(metadata, sample_id=row['sample_id'], scene_id=row['scene_id'], family_id=row.get('family_id') or row.get('pair_id') or row['scene_id'],
                              pair_id=row.get('pair_id'), split=row['split'], arm=arm, topology=runtime.writer.topology,
                              prompt=row['prompt'], supplementary_caption=row['caption'],
                              fixed_bank_content={'G-fixed': 'zero_kv', 'G-neutral': 'neutral_text_seed_kv'}.get(arm),
                              target_facts=row['facts'], swap_id=row.get('swap_id') if 'swap' in arm else None,
                              writer_prompt=metadata['prompt'], conditional=velocity_stats(conditional, layout[0]),
                              cfg=velocity_stats(final, layout[1]), repeat_error=repeat_error,
                              layout_native_shift_parity=parity,
                              velocity_hash=tensor_hash(final), frozen_guard='passed',
                              semantic_verdict='pending_blind_annotation')
                name = f'{key}_{arm}'
                runtime.image(x-t*final).save(output/(name+'_x0_hat.png'))
                if continuation:
                    final_state = runtime.continue_once(x, t, final, flow, remaining)
                    runtime.image(final_state).save(output/(name+'_final.png'))
                    blind.append(dict(scene_id=row['scene_id'], family_id=row.get('family_id') or row.get('pair_id') or row['scene_id'], sample_id=row['sample_id'], t=t, seed=seed,
                                      arm=arm, file=name+'_final.png'))
                append_json(output/'gen.jsonl', record); result_rows.append(record)
            guard.check(); synchronize(runtime.device)
            append_json(output/'timing.jsonl', dict(sample_id=row['sample_id'], t=t, noise_seed=seed,
                        seconds=time.perf_counter()-tic, peak_memory_bytes=max_memory_allocated(runtime.device)))
    random.Random(runtime.config['seed']).shuffle(blind)
    mapping = []
    for i, item in enumerate(blind):
        anonymous = f'blind_{i:05d}.png'
        (output/anonymous).write_bytes((output/item['file']).read_bytes())
        mapping.append(dict(blind_file=anonymous, **item))
    write_json(output/'blind_key.json', mapping)
    # Review images without the key; facts/quality are separate annotation fields.
    (output/'blind_annotations.csv').write_text('blind_file,target_facts,unchanged_facts,quality,artifacts,notes\n'+
                                              ''.join(f'{r["blind_file"]},,,,,\n' for r in mapping))
    write_json(output/'gen_summary.json', dict(samples=len(rows), records=len(result_rows),
                continuation=continuation, reference='G-layout', layout_reference='G-base',
                semantic_quality='unverified; numerical sensitivity is not semantic success',
                backend_parity='review layout_native_shift_parity against native shifted-position call on target device',
                visual_bridge='included' if visual_checkpoint else 'not tested', frozen_guard='passed'))
