"""Single native-node explicit teacher and paired R2 feedback inference.

All arms restart the saved x_t/t. Preview and final suffixes use the existing
native sampler. No image-understanding KV or preview latent enters GEN.
"""
from dataclasses import asdict, dataclass, replace
import html
import json
import time
import torch
from .native import capture_gen, cache_kv, device_inputs, empty_cache
from .gen import supplemental_flow
from .feedback_source import prompt_source, write_feedback
from .numerics import autocast_for
from .schedule import native_times, native_suffix, nearest_node
from .data import controlled_noise, tensor_hash, velocity_stats
from .record import FrozenGuard, write_json
from ..bagel.accelerator import synchronize


@dataclass(frozen=True)
class Feedback:
    observation: str
    gap: str
    correction: str
    satisfied: bool = False
    origin: str = 'manual_checked'
    raw: str | None = None

    def __post_init__(self):
        if any(not isinstance(value, str) for value in (self.observation, self.gap, self.correction)):
            raise ValueError('feedback observation/gap/correction must be strings')
        if type(self.satisfied) is not bool:
            raise ValueError('feedback satisfied must be a boolean')
        if not self.satisfied and not self.correction.strip():
            raise ValueError('an unsatisfied preview requires a positive correction')
        if self.origin not in ('manual_checked', 'bagel_auto', 'source_annotation'):
            raise ValueError('unknown feedback provenance')

    @classmethod
    def parse(cls, raw):
        text = raw.strip()
        if text.startswith('```') and text.endswith('```'):
            text = '\n'.join(text.splitlines()[1:-1])
        value = json.loads(text)
        if set(value) != {'observation', 'gap', 'correction', 'satisfied'}:
            raise ValueError('reflection must return the four declared JSON fields')
        return cls(**value, origin='bagel_auto', raw=raw)


@torch.no_grad()
def reflect(runtime, preview, prompt, image_transform, max_tokens=384):
    """Native BAGEL chat stages with backend-aware autocast (including NPU).

    bagel.chat hardcodes CUDA autocast. Reuse its preparation/cache/generation
    calls here without modifying the historical chat implementation.
    """
    model = runtime.model; bundle = runtime.bundle
    if not model.config.visual_und:
        raise ValueError('automatic reflection requires the native visual UND model')
    instruction = (
        'Compare this image with the original target prompt below. Report only visible facts. '
        'Return one JSON object with exactly these fields: observation (string), gap (string), '
        'correction (string), satisfied (boolean). correction must be a short positive description '
        'of the desired image, preserving the correct main content and composition. '
        'Do not send analysis, descriptions of errors, or negative instructions in correction. '
        'If the target is already satisfied, set satisfied=true and correction="".\n'
        'Original target prompt:\n'+prompt)
    cache = empty_cache(runtime.decoder)
    with autocast_for(runtime.device):
        inputs, lengths, positions = model.prepare_vit_images([0], [0], [preview], image_transform, bundle.token_ids)
        cache = model.forward_cache_update_vit(cache, **device_inputs(inputs, runtime.device))
        inputs, lengths, positions = model.prepare_prompts(lengths, positions, [instruction],
                                                            bundle.tokenizer, bundle.token_ids)
        cache = model.forward_cache_update_text(cache, **device_inputs(inputs, runtime.device))
        inputs = model.prepare_start_tokens(lengths, positions, bundle.token_ids)
        generated = model.generate_text(past_key_values=cache, max_length=max_tokens,
                                       do_sample=False, end_token_id=bundle.token_ids['eos_token_id'],
                                       **device_inputs(inputs, runtime.device))
    raw = bundle.tokenizer.decode(generated[:, 0].tolist(), skip_special_tokens=True)
    try:
        return Feedback.parse(raw)
    except (ValueError, TypeError) as error:
        # Persist failed automatic text for review; never guess a correction.
        error.raw_reflection = raw
        raise


@torch.no_grad()
def correction_source(runtime, flow, lengths, positions, correction):
    """Native append on a clean prompt clone; extract only C, never duplicate P."""
    if len(lengths) != 1:
        raise ValueError('explicit feedback inference is batch one')
    augmented = supplemental_flow(runtime, flow, lengths, positions, correction)
    total = tuple(int(n) for n in augmented['key_values_lens'].tolist())
    count = total[0]-lengths[0]
    source = tuple(replace(kv, keys=kv.keys[lengths[0]:], values=kv.values[lengths[0]:],
                           lengths=(count,), positions=tuple(range(positions[0], positions[0]+count)),
                           token_roles=('correction',)*count)
                   for kv in cache_kv(augmented['past_key_values'], total))
    return source, augmented, count


def shifted_layout(flow, count):
    layout = dict(flow)
    layout['packed_position_ids'] = flow['packed_position_ids']+count
    return layout


def response_diagnostics(x, t, next_t, velocities):
    """Separate content response from position-only effects, before/after CFG.

    Nonzero response is sensitivity, not proof of correct final-image edits.
    """
    matched = {'Text': 'Layout-text', 'Text-memory': 'Layout-memory',
               'Swapped-memory': 'Layout-memory', 'Implicit-memory': 'Layout-memory'}
    dt = t-next_t
    if dt <= 0:
        raise ValueError('response diagnostics require a positive native Euler step')
    records = {}
    for arm, pair in velocities.items():
        reference = matched.get(arm, 'Base')
        # A strict satisfied bypass has no reserved layout or extra KV.
        if reference not in velocities:
            reference = 'Base'
        baseline = velocities[reference]
        delta = dt*(pair[1].float()-baseline[1].float())
        records[arm] = dict(reference=reference,
            conditional=velocity_stats(pair[0], baseline[0]), cfg=velocity_stats(pair[1], baseline[1]),
            conditional_hash=tensor_hash(pair[0]), cfg_hash=tensor_hash(pair[1]),
            first_step_delta_l2=float(delta.norm()),
            first_step_delta_rms=float(delta.square().mean().sqrt()),
            first_step_delta_over_x_norm=float(delta.norm()/x.float().norm().clamp_min(1e-12)))
    return dict(dt=dt, remaining_euler_steps=None, arms=records,
                interpretation='numerical response only; no semantic quality verdict')


def image_diagnostics(images):
    """Measure final RGB content against the corresponding layout control."""
    import numpy as np
    matched = {'Text': 'Layout-text', 'Text-memory': 'Layout-memory',
               'Swapped-memory': 'Layout-memory', 'Implicit-memory': 'Layout-memory'}
    out = {}
    for arm, image in images.items():
        prefix, _, kind = arm.rpartition('_')
        reference = prefix+'_'+matched[kind] if prefix and kind in matched else matched.get(arm, 'Base')
        if reference not in images:
            reference = 'Base'
        a = np.asarray(image.convert('RGB'), dtype=np.float32)
        b = np.asarray(images[reference].convert('RGB'), dtype=np.float32)
        if a.shape != b.shape:
            raise ValueError('paired final images must have matching shapes')
        delta = np.abs(a-b)
        out[arm] = dict(reference=reference, identical=bool(np.array_equal(a, b)),
            mae_255=float(delta.mean()), max_channel_delta=int(delta.max()),
            changed_pixel_fraction=float(np.any(delta != 0, axis=-1).mean()))
    return out


@torch.no_grad()
def intervention_state(runtime, prompt, seed, requested_t):
    """Prompt-only native prefix; no alternate x0/noise is introduced at R2."""
    flow, lengths, positions = runtime.flow_inputs(prompt)
    step, actual = nearest_node(runtime.config, requested_t)
    width = runtime.model.latent_channel*runtime.model.latent_patch_size**2
    x = controlled_noise(torch.zeros(len(flow['packed_vae_token_indexes']), width,
                                     device=runtime.device), 1, seed)
    times = native_times(runtime.config)
    for current, following in zip(times[:step], times[1:step+1]):
        x = x-(current-following)*runtime.velocity(x, current, flow)[1]
    return x, actual, flow, lengths, positions, dict(prompt=prompt, noise_seed=seed,
        requested_t=requested_t, t=actual, step_index=step, x_t_hash=tensor_hash(x),
        sampler_steps=runtime.config['sampler_steps'], timestep_shift=runtime.config['timestep_shift'])


@torch.no_grad()
def compare(runtime, x, t, flow, lengths, positions, *, feedback=None, reflector=None,
            source_layers=None, student=None, donor=None, original_prompt=None, preview_kind='complete'):
    """R1 full exit, isolated teacher preview, same-state Text/Memory/Implicit R2.

    donor contains checked feedback and metadata (prompt, t, x_t_hash). A donor
    with the same correction is recorded as non-discriminating swap evidence.
    """
    if runtime.writer.topology != 'C' or runtime.writer.slots != 16:
        raise ValueError('explicit feedback requires continuous 16-slot writer')
    if preview_kind not in ('complete', 'local_cfg'):
        raise ValueError('unknown teacher preview kind')
    if (feedback is None) == (reflector is None):
        raise ValueError('select exactly one checked feedback or automatic reflector')
    runtime.writer.eval().requires_grad_(False)
    guarded = [runtime.model, runtime.bundle.vae, runtime.writer]
    if student is not None:
        student.eval().requires_grad_(False)
        guarded.append(student)
    guard = FrozenGuard(*guarded)
    layers = tuple(range(len(runtime.decoder.layers))) if source_layers is None else tuple(source_layers)
    if not layers or len(set(layers)) != len(layers) or any(i < 0 or i >= len(runtime.decoder.layers) for i in layers):
        raise ValueError('invalid GEN source window')
    original = x.clone(); state_hash = tensor_hash(x)
    prompt = prompt_source(flow, lengths, positions)
    remaining = native_suffix(runtime.config, t)
    timings = {}
    def measured(name, operation):
        synchronize(runtime.device); start = time.perf_counter()
        result = operation()
        synchronize(runtime.device); timings[name] = time.perf_counter()-start
        return result
    def first_pass():
        with capture_gen(runtime.decoder, flow['past_key_values'], t=t, layers=layers,
                         positions=flow['packed_position_ids']) as captured:
            velocity = runtime.velocity(x, t, flow)
        return velocity, tuple(captured[i] for i in layers)
    r1, gen_source = measured('r1_gen', first_pass)
    # Always finish the native exit before decoding; never decode shallow hidden.
    base_latent = measured('preview_suffix', lambda: runtime.continue_once(x, t, r1[1], flow, remaining))
    base_image = measured('preview_decode', lambda: runtime.image(base_latent))
    preview = base_image if preview_kind == 'complete' else measured('local_preview_decode', lambda: runtime.image(x-t*r1[1]))
    if reflector is not None:
        feedback = measured('reflection', lambda: reflector(preview))
    if not isinstance(feedback, Feedback):
        raise ValueError('reflection must return validated Feedback')
    images = {'Base': base_image}; velocities = {'Base': r1}; banks = {}
    counts = {'memory': 16, 'text': 0}; swap_discriminating = None; swap_details = None
    def final_arm(name, velocity):
        velocities[name] = velocity
        latent = measured(name+'_suffix', lambda: runtime.continue_once(x, t, velocity[1], flow, remaining))
        images[name] = measured(name+'_decode', lambda: runtime.image(latent))
    if feedback.satisfied:
        # Strict bypass: do not call a shifted/zero/neutral KV path.
        images.update({'Text': base_image, 'Text-memory': base_image})
        velocities.update({'Text': r1, 'Text-memory': r1})
    else:
        correction, augmented, count = measured('text_prefill', lambda: correction_source(runtime, flow, lengths, positions, feedback.correction))
        counts['text'] = count
        bank = measured('text_writer', lambda: write_feedback(runtime, prompt, correction))
        banks['Text-memory'] = bank
        fixed = runtime.writer.fixed_bank(runtime.seeds)
        final_arm('Layout-memory', measured('Layout-memory_r2', lambda: runtime.gen_arm(x, t, flow, fixed, lengths, positions, masked=True)))
        final_arm('Layout-text', measured('Layout-text_r2', lambda: runtime.velocity(x, t, shifted_layout(flow, count))))
        final_arm('Text', measured('Text_r2', lambda: runtime.velocity(x, t, augmented)))
        final_arm('Text-memory', measured('Text-memory_r2', lambda: runtime.gen_arm(x, t, flow, bank, lengths, positions)))
        if donor is not None:
            meta = donor['metadata']
            if not original_prompt or (meta.get('prompt'), meta.get('t')) != (original_prompt, t) or not meta.get('x_t_hash') or meta['x_t_hash'] == state_hash:
                raise ValueError('swap requires a different state at the same prompt/t')
            donor_feedback = donor['feedback']
            if donor_feedback.satisfied:
                raise ValueError('swap donor must have a correction')
            # Different strings do not prove different error types. Review the
            # donor facts before treating this as evidence of state dependence.
            swap_discriminating = False if donor_feedback.correction == feedback.correction else None
            swap_details = dict(metadata=meta, feedback=asdict(donor_feedback),
                                error_type_validation='pending human review')
            source, _, _ = correction_source(runtime, flow, lengths, positions, donor_feedback.correction)
            swapped = measured('swap_writer', lambda: write_feedback(runtime, prompt, source))
            banks['Swapped-memory'] = swapped
            final_arm('Swapped-memory', measured('Swapped-memory_r2', lambda: runtime.gen_arm(x, t, flow, swapped, lengths, positions)))
    if student is not None:
        if 'Layout-memory' not in images:
            fixed = runtime.writer.fixed_bank(runtime.seeds)
            final_arm('Layout-memory', measured('Layout-memory_r2', lambda: runtime.gen_arm(x, t, flow, fixed, lengths, positions, masked=True)))
        bank = measured('implicit_writer', lambda: write_feedback(runtime, prompt, gen_source, layers=layers,
                                writer=student.writer, visual_adapter=student.visual, t=t))
        banks['Implicit-memory'] = bank
        final_arm('Implicit-memory', measured('Implicit-memory_r2', lambda: runtime.gen_arm(x, t, flow, bank, lengths, positions)))
    if tensor_hash(x) != state_hash or not torch.equal(x, original):
        raise RuntimeError('preview/feedback changed the saved intervention state')
    guard.check()
    diagnostics = response_diagnostics(x, t, remaining[0], velocities)
    diagnostics['remaining_euler_steps'] = len(remaining)
    return dict(preview=preview, images=images, velocities=velocities, banks=banks,
                feedback=feedback, timings=timings, token_counts=counts,
                t=t, x_t_hash=state_hash, source_layers=layers, preview_kind=preview_kind,
                continuation_times=[t]+remaining, bypass=feedback.satisfied,
                swap_discriminating=swap_discriminating, swap_details=swap_details, frozen_guard='passed',
                response_diagnostics=diagnostics)


def save_comparison(output, name, result, metadata):
    result['preview'].save(output/(name+'_preview.png'))
    files = {}
    for arm, image in result['images'].items():
        files[arm] = name+'_'+arm+'.png'
        image.save(output/files[arm])
    velocity_file = name+'_velocities.pt'
    torch.save(dict(t=result['t'], x_t_hash=result['x_t_hash'],
                    velocities={arm: dict(conditional=pair[0].detach().cpu(), cfg=pair[1].detach().cpu())
                                for arm, pair in result['velocities'].items()}), output/velocity_file)
    record = dict(metadata, **{k: result[k] for k in ('t', 'x_t_hash', 'source_layers', 'preview_kind',
        'continuation_times', 'bypass', 'swap_discriminating', 'swap_details', 'frozen_guard', 'timings', 'token_counts')},
        feedback=asdict(result['feedback']), preview=name+'_preview.png', images=files,
        response_diagnostics=result['response_diagnostics'], final_image_diagnostics=image_diagnostics(result['images']),
        velocity_file=velocity_file,
        semantic_quality='unverified; requires paired final-image annotation',
        suffix_policy='single intervention followed by native original-prompt suffix')
    write_json(output/(name+'.json'), record)
    return record


def render_review(output, records):
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>显式反馈配对审查</title>',
             '<style>body{font-family:system-ui;margin:24px;background:#faf9f6;color:#222} '
             '.images{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:260px} '
             'img{width:100%}pre{white-space:pre-wrap}section{border-top:1px solid #bbb;padding:24px 0}</style>',
             '<h1>显式反馈配对审查</h1><p>终图纠错、正确事实保持与质量均待人工评价。自动反馈与核对反馈分别审查。</p>']
    for r in records:
        parts += ['<section><h2>'+html.escape(r['sample_id'])+'</h2><p>'+html.escape(r['prompt'])+'</p>',
                  '<pre>'+html.escape(json.dumps({k:r[k] for k in ('feedback','t','noise_seed','timings')}, ensure_ascii=False, indent=2))+'</pre><div class="images">']
        for arm, file in [('R1 preview', r['preview']), *r['images'].items()]:
            parts.append('<figure><img src="'+html.escape(file, quote=True)+'"><figcaption>'+html.escape(arm)+'</figcaption></figure>')
        parts.append('</div>')
        if 'response_diagnostics' in r:
            parts.append('<details><summary>条件响应与终图差异（不代表纠错成功）</summary><pre>'+html.escape(
                json.dumps({key:r[key] for key in ('response_diagnostics', 'final_image_diagnostics')}, ensure_ascii=False, indent=2))+'</pre></details>')
        parts.append('</section>')
    parts.append('</html>')
    (output/'review.html').write_text('\n'.join(parts))
