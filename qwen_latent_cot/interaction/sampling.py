"""Early-half full-depth recurrence; native CFG and Euler integration."""
from contextlib import nullcontext
import torch
from ..diagnostics.flow_training import TrainingState
from ..diagnostics.schedule import native_times
from .loop import active_updates


@torch.no_grad()
def sample(student, prompt, seed=0, *, native=False, on_step=None):
    with student.native_context() if native else nullcontext():
        return _sample(student, prompt, seed, native=native, on_step=on_step)


def _sample(student, prompt, seed, *, native, on_step):
    runtime = student.runtime
    flow, lengths, positions = runtime.flow_inputs(prompt)
    model = runtime.model
    size = runtime.config['resolution']//model.latent_downsample
    width = model.latent_channel*model.latent_patch_size**2
    x = torch.randn(size*size, width, generator=torch.Generator(device='cpu').manual_seed(seed)).to(runtime.device)
    times = native_times(runtime.config)
    records = []
    for index, (current, following) in enumerate(zip(times, times[1:])):
        active = not native and index < active_updates(runtime.config)
        if active:
            state = TrainingState(x, None, None, current, flow, tuple(lengths), positions,
                                  dict(prompt=prompt, flow_supervision=False))
            result = student(state, guidance=True)
            velocity = result.velocity
            passes = 1+len(result.exits)
        else:
            velocity = runtime.velocity(x, current, flow)[1]
            passes = 1
        x = x-(current-following)*velocity
        records.append(dict(step=index+1, t=current, next_t=following, loop_active=active,
                            passes=passes, loop_blocks=len(runtime.decoder.layers),
                            conditional_gen_layer_calls=passes*len(runtime.decoder.layers),
                            und_layer_calls=(passes-1)*len(runtime.decoder.layers)))
        if on_step is not None:
            on_step(records[-1])
    return runtime.image(x), records
