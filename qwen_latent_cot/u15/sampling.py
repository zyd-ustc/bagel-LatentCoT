"""Native 50 Euler updates, with a single optional post-CFG correction."""
import torch
from PIL import Image
import numpy as np
from .objective import project,coefficient


@torch.no_grad()
def sample(runtime,prompt,seed,corrector=None,arm='native'):
    if arm not in ('native','global','corrected'): raise ValueError(arm)
    if arm != 'native' and corrector is None: raise ValueError('Correction checkpoint required')
    cfg=runtime.config; prefix=runtime.prefix(prompt)
    reference=runtime.prefix(None) if cfg['cfg_scale']>1 else prefix
    dtype=next(runtime.model.parameters()).dtype
    image=runtime.noise_scale()*torch.randn((1,3,runtime.height,runtime.width),device=runtime.device,
        dtype=dtype,generator=torch.Generator(device=runtime.device).manual_seed(seed))
    trace=[];times=runtime.times()
    for index,(t,following) in enumerate(zip(times,times[1:])):
        active=arm!='native' and index<cfg['active_steps']
        prediction=runtime.predict(image,t,prefix,capture=active and arm=='corrected')
        unconditional=runtime.predict(image,t,reference).velocity if cfg['cfg_scale']>1 else prediction.velocity
        base=runtime.guided(prediction.velocity,unconditional,cfg['cfg_scale'])
        raw=torch.zeros_like(base,dtype=torch.float32)
        if active:
            raw=(corrector.global_delta(index).expand_as(base) if arm=='global' else
                 corrector(prediction.observation,t,prefix.hidden))
        guidance=base.float()-prediction.velocity.float();delta=project(raw,guidance)
        packed=runtime.model.patchify(image,runtime.patch)
        packed=packed+(following-t)*runtime.model.patchify(base,runtime.patch)
        # Exact native numerical path for zero correction, including BF16 dtype.
        if active and bool(delta.any()): packed=packed.float()+(following-t)*runtime.model.patchify(delta,runtime.patch)
        image=runtime.model.unpatchify(packed,runtime.patch,runtime.height,runtime.width)
        if not torch.isfinite(image).all():raise FloatingPointError('Nonfinite RGB sampling state')
        trace.append(dict(step=index,t=float(t),next_t=float(following),und_passes=int(active and arm=='corrected'),
            delta_rms=float(delta.square().mean().sqrt()),raw_projection=float(coefficient(raw,guidance)[0]),
            applied_projection=float(coefficient(delta,guidance)[0])))
    array=((image[0].float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy()
    return Image.fromarray(np.asarray(array)),trace
