"""Stage 1 image editing and UND-guided T2I feedback."""

import torch

from .diagnosis import review_reflection
from .field import NativeT2I, edit_state, euler, phase_grid, preview_image, segment_times
from .images import load_canvas


def _edit(editor, state, time, instruction, steps):
    prefixes = editor.prefixes(instruction)
    base = editor.image_clean(state, time, prefixes['image'])
    return edit_state(editor, state, time, prefixes, steps, base_prediction=base)[0]


@torch.no_grad()
def edit_image(rt, editor, cfg, image_path, instruction, seed):
    image, _, _ = load_canvas(image_path, rt.width, rt.height, rt.device)
    time = float(cfg['read_time'])
    generator = torch.Generator(device=rt.device).manual_seed(int(seed))
    noise = torch.randn(image.shape, device=rt.device, generator=generator) * rt.noise_scale()
    state = time * image + (1 - time) * noise
    edited = _edit(editor, state, time, instruction, int(cfg['edit_steps']))
    grid = phase_grid(rt, cfg)
    native = NativeT2I(rt, None, cfg)
    preview = preview_image(edited, time, grid, native)
    result = euler(edited, segment_times(grid, time, 1.0), native)
    return {'source': image, 'edited_preview': preview, 'result': result}, edited, dict(
        mode='edit', instruction=instruction, seed=seed, read_time=time,
        continuation='unconditional native T2I', state_initialization='linear source and noise')


@torch.no_grad()
def t2i_feedback(rt, editor, cfg, prompt, seed):
    grid = phase_grid(rt, cfg)
    time = float(cfg['read_time'])
    native = NativeT2I(rt, prompt, cfg)
    generator = torch.Generator(device=rt.device).manual_seed(int(seed))
    noise = torch.randn((1, 3, rt.height, rt.width), device=rt.device, generator=generator) * rt.noise_scale()
    state = euler(noise, segment_times(grid, 0, time), native)
    preview = preview_image(state, time, grid, native)
    review = review_reflection(rt, preview, prompt, cfg)['reflection']
    if review['outcome'] in ('failed_parse', 'failed_unconverted'):
        raise ValueError('UND reflection could not produce a decision: ' + review['raw'])
    edited = None
    images = {'preview': preview}
    if review['outcome'] == 'edit':
        edited = _edit(editor, state, time, review['instruction'], int(cfg['edit_steps']))
        images['edited_preview'] = preview_image(edited, time, grid, native)
    result = euler(edited if edited is not None else state, segment_times(grid, time, 1), native)
    images['result'] = result
    return images, edited, dict(mode='t2i', prompt=prompt, seed=seed, read_time=time,
                                reflection=review)
