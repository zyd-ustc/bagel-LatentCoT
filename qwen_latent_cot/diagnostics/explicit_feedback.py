"""Complete BAGEL generation, text reflection and fixed-memory regeneration."""
from dataclasses import asdict, dataclass, replace
import html
import json
import time
import torch
from .native import cache_kv, device_inputs, empty_cache
from .gen import supplemental_flow
from .feedback_source import prompt_source, write_feedback
from .numerics import autocast_for
from .schedule import native_times
from .data import tensor_hash
from .record import FrozenGuard, write_json
from ..bagel.accelerator import synchronize

PROTOCOL = 'explicit_feedback'



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



def image_diagnostics(images):
    """Measure final RGB content against the corresponding layout control."""
    import numpy as np
    matched = {'Text': 'Layout-text', 'Text-memory': 'Layout-memory'}
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



def validate_windows(config, early):
    steps = len(native_times(config))-1
    if not early or len(set(early)) != len(early) or any(type(n) is not int or not 0 < n <= steps for n in early):
        raise ValueError('early windows must be distinct positive counts within the native Euler schedule')



@torch.no_grad()
def initial_noise(runtime, flow, seed):
    """Same CPU generator, float32 noise and patch layout as shallow sample()."""
    size = runtime.config['resolution']//runtime.model.latent_downsample
    width = runtime.model.latent_channel*runtime.model.latent_patch_size**2
    if len(flow['packed_vae_token_indexes']) != size*size:
        raise ValueError('native flow query and shallow-sampling latent layout differ')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randn(size*size, width, generator=generator).to(runtime.device)



@torch.no_grad()
def sample_fixed(runtime, initial, flow, lengths, positions, *, bank=None, early=0,
                 masked=False, text_flow=None, position_shift=0):
    """Repeat the native Euler loop; use the SAME bank for every active step."""
    if bank is not None and text_flow is not None:
        raise ValueError('choose one additional condition per sampling arm')
    if masked and bank is None:
        raise ValueError('masked layout requires a memory bank')
    if bank is not None or text_flow is not None or position_shift:
        validate_windows(runtime.config, [early])
    elif early:
        raise ValueError('native baseline has no intervention window')
    state = initial.clone()
    entrance_hash = tensor_hash(state)
    records = []
    times = native_times(runtime.config)
    for index, (current, following) in enumerate(zip(times, times[1:])):
        active = index < early
        if active and bank is not None:
            conditional, velocity = runtime.gen_arm(state, current, flow, bank, lengths, positions, masked=masked)
        else:
            inputs = flow
            if active and text_flow is not None:
                inputs = text_flow
            elif active and position_shift:
                inputs = dict(flow, packed_position_ids=flow['packed_position_ids']+position_shift)
            conditional, velocity = runtime.velocity(state, current, inputs)
        records.append(dict(step=index, t=current, next_t=following, feedback_active=active,
            memory_visible=bool(active and bank is not None and not masked),
            memory_masked=bool(active and masked), conditional_gen_passes=1,
            conditional_gen_layer_calls=len(runtime.decoder.layers),
            cfg_velocity_hash=tensor_hash(velocity), conditional_velocity_hash=tensor_hash(conditional),
            cfg_velocity_norm=float(velocity.float().norm()), conditional_velocity_norm=float(conditional.float().norm())))
        state = state-(current-following)*velocity
    if tensor_hash(initial) != entrance_hash:
        raise RuntimeError('generation mutated its shared initial noise')
    return dict(latent=state, nodes=records, initial_noise_hash=entrance_hash,
                initial_noise_shape=list(initial.shape), gen_passes_per_step=1,
                active_updates=sum(r['feedback_active'] for r in records))



@torch.no_grad()
def generate_base(runtime, prompt, seed, initial=None):
    flow, lengths, positions = runtime.flow_inputs(prompt)
    expected = initial_noise(runtime, flow, seed)
    if initial is None:
        initial = expected
    elif initial.shape != expected.shape or initial.dtype != expected.dtype or not torch.equal(initial, expected):
        raise ValueError('saved initial noise differs from the declared seed/native layout')
    runtime.writer.eval().requires_grad_(False)
    guard = FrozenGuard(runtime.model, runtime.bundle.vae, runtime.writer)
    synchronize(runtime.device); start = time.perf_counter()
    base = sample_fixed(runtime, initial, flow, lengths, positions)
    synchronize(runtime.device); generation_seconds = time.perf_counter()-start
    start = time.perf_counter()
    image = runtime.image(base['latent'])
    synchronize(runtime.device); decode_seconds = time.perf_counter()-start
    guard.check()
    return dict(prompt=prompt, seed=seed, initial=initial, flow=flow, lengths=lengths, positions=positions,
                base=base, image=image, generation_seconds=generation_seconds, decode_seconds=decode_seconds)



@torch.no_grad()
def compare(runtime, prepared, *, early=(10, 20), feedback=None, reflector=None, text_control=False):
    validate_windows(runtime.config, early)
    if runtime.writer.topology != 'C' or runtime.writer.slots != 16:
        raise ValueError('restart feedback requires continuous C1 memory with 16 slots')
    if (feedback is None) == (reflector is None):
        raise ValueError('choose exactly one checked feedback or automatic reflection')
    runtime.writer.eval().requires_grad_(False)
    guard = FrozenGuard(runtime.model, runtime.bundle.vae, runtime.writer)
    initial = prepared['initial']; flow = prepared['flow']
    lengths = prepared['lengths']; positions = prepared['positions']
    timings = {'base_generation':prepared['generation_seconds'], 'base_decode':prepared['decode_seconds']}
    def measured(name, operation):
        synchronize(runtime.device); start = time.perf_counter()
        result = operation()
        synchronize(runtime.device); timings[name] = time.perf_counter()-start
        return result
    if reflector is not None:
        feedback = measured('reflection', lambda: reflector(prepared['image']))
    if not isinstance(feedback, Feedback):
        raise ValueError('reflection must return validated Feedback')
    images = {'Base':prepared['image']}; arms = {'Base':prepared['base']}
    prompt = prompt_source(flow, lengths, positions)
    bank = None; count = 0
    if not feedback.satisfied:
        # Encode only reflection text on a clean original prompt cache. UND
        # vision-session KV and the first generated image are never GEN inputs.
        correction, augmented, count = measured('reflection_text_prefill', lambda:
            correction_source(runtime, flow, lengths, positions, feedback.correction))
        bank = measured('text_writer_once', lambda: write_feedback(runtime, prompt, correction))
    for window in early:
        name = f'early{window}_Text-memory'
        if feedback.satisfied:
            arms[name] = dict(prepared['base'], requested_early=window, strict_bypass=True)
            images[name] = prepared['image']
            continue
        variants = [(f'early{window}_Layout-memory', dict(bank=bank, masked=True)),
                    (name, dict(bank=bank))]
        if text_control:
            variants += [(f'early{window}_Layout-text', dict(position_shift=count)),
                         (f'early{window}_Text', dict(text_flow=augmented))]
        for arm, options in variants:
            result = measured(arm+'_generation', lambda options=options:
                sample_fixed(runtime, initial, flow, lengths, positions, early=window, **options))
            times = native_times(runtime.config)
            result.update(requested_early=window, active_time_range=[times[0], times[window]])
            arms[arm] = result
            images[arm] = measured(arm+'_decode', lambda: runtime.image(result['latent']))
    entrance_hash = tensor_hash(initial)
    if any(arm['initial_noise_hash'] != entrance_hash for arm in arms.values()):
        raise RuntimeError('paired generations did not start at the identical initial noise')
    guard.check()
    return dict(protocol=PROTOCOL, feedback=feedback, images=images, arms=arms, bank=bank,
        timings=timings, early=list(early), initial_noise_hash=entrance_hash,
        correction_tokens=count, bypass=feedback.satisfied, frozen_guard='passed',
        writer_updates=int(bank is not None), memory_policy='one fixed bank shared across all active steps/windows',
        feedback_text=feedback.correction, sampling_times=native_times(runtime.config))



def save_comparison(output, name, result, metadata):
    files = {}
    for arm, image in result['images'].items():
        files[arm] = name+'_'+arm+'.png'
        image.save(output/files[arm])
    # Keep only scalar/hash trajectory evidence in JSON; final latents are not
    # recycled into a subsequent generation.
    arms = {arm:{key:value for key,value in item.items() if key != 'latent'}
            for arm,item in result['arms'].items()}
    bank_file = None
    if result['bank'] is not None:
        bank_file = name+'_memory.pt'
        torch.save(dict(initial_noise_hash=result['initial_noise_hash'], feedback_text=result['feedback_text'],
            slots=result['bank'].slots, topology=result['bank'].topology,
            layers=[dict(layer=kv.layer, keys=kv.keys.cpu(), values=kv.values.cpu(),
                         lengths=kv.lengths, rotated=kv.rotated) for kv in result['bank'].layers]), output/bank_file)
    record = dict(metadata, **{key:result[key] for key in ('protocol','timings','early','initial_noise_hash',
        'correction_tokens','bypass','frozen_guard','writer_updates','memory_policy','feedback_text','sampling_times')},
        feedback=asdict(result['feedback']), base_image=files['Base'], images=files, arms=arms,
        memory_file=bank_file, final_image_diagnostics=image_diagnostics(result['images']),
        semantic_quality='unverified; requires paired final-image annotation',
        suffix_policy='fixed memory for first early updates; original-prompt native GEN afterward',
        shallow_loop=False, memory_refresh=False)
    write_json(output/(name+'.json'), record)
    return record



def render_review(output, records):
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>显式反馈：相同噪声重新生成</title>',
        '<style>body{font-family:system-ui;margin:24px;background:#faf9f6}.images{display:flex;flex-wrap:wrap;gap:12px} '
        'figure{margin:0;width:300px}img{width:100%}pre{white-space:pre-wrap}section{border-top:1px solid #bbb;padding:20px 0}</style>',
        '<h1>完整生成 → 图像理解反思 → 固定 memory → 相同噪声重新生成</h1>',
        '<p>第二次生成在 early 窗口内使用同一份 memory。每个采样步只执行一次完整 GEN。终图纠错效果待人工评价。</p>']
    for record in records:
        parts += ['<section><h2>'+html.escape(record['sample_id'])+'</h2><p>'+html.escape(record['prompt'])+'</p>',
            '<pre>'+html.escape(json.dumps({key:record[key] for key in ('noise_seed','initial_noise_hash','feedback','early','timings')},
                                          ensure_ascii=False, indent=2))+'</pre><div class="images">']
        for arm, file in record['images'].items():
            parts.append('<figure><img src="'+html.escape(file, quote=True)+'"><figcaption>'+html.escape(arm)+'</figcaption></figure>')
        parts += ['</div><details><summary>采样窗口与逐步执行记录</summary><pre>'+html.escape(
            json.dumps(record['arms'], ensure_ascii=False, indent=2))+'</pre></details></section>']
    parts.append('</html>')
    (output/'review.html').write_text('\n'.join(parts))
