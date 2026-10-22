"""One post-CFG correction per active Euler update; retain v1 checkpoint semantics."""
import math
import torch
from .model import proposal_from
from .numerics import guided
from .objective import guidance_safe,applied_correction,guidance_coefficient


def times(config):
    nodes=torch.linspace(1,0,config['sampler_steps']);shift=config['timestep_shift']
    return (shift*nodes/(1+(shift-1)*nodes)).tolist()


def active_steps(config): return math.ceil((config['sampler_steps']-1)/2)


@torch.no_grad()
def sample(corrector,prompt,seed,*,native=False,global_only=False):
    runtime=corrector.runtime;cfg=corrector.config;flow=runtime.flow(prompt)
    n=len(flow['packed_vae_token_indexes']);c=runtime.model.latent_channel*runtime.model.latent_patch_size**2
    x=torch.randn((n,c),generator=torch.Generator().manual_seed(seed)).to(runtime.device)
    trace=[];grid=times(cfg)
    for index,(t,following) in enumerate(zip(grid,grid[1:])):
        active=not native and index<active_steps(cfg)
        deployed = cfg.get('correction_space') == 'post_cfg'
        if deployed:
            if guidance_safe(cfg):
                base,context,conditional=runtime.proposal(x,t,flow,capture=active and not global_only,with_conditional=True)
            else:
                base,context=runtime.proposal(x,t,flow,capture=active and not global_only);conditional=None
            delta = (corrector.calibration(t).expand_as(base) if global_only else
                     corrector(proposal_from(x,t,base),t,flow,context)) if active else torch.zeros_like(base)
            raw_delta=delta
            delta=applied_correction(raw_delta,base,cfg,conditional)
            velocity = base.float()+delta.float()
        else:
            # Existing v1 checkpoints keep their original conditional-before-CFG contract.
            if global_only: raise ValueError('Global calibration requires post-CFG training')
            base,context=runtime.predict(x,t,flow,capture=active)
            delta=corrector(proposal_from(x,t,base),t,flow,context) if active else torch.zeros_like(base)
            conditional=(base.float()+delta.float()).to(base.dtype)
            reference=runtime.reference(x,t,flow) if cfg['cfg_text_scale']>1 else conditional
            velocity=guided(conditional,reference,cfg)
        if not torch.isfinite(velocity).all():raise FloatingPointError('Nonfinite sampling velocity')
        if deployed:
            # Keep the native BF16 Euler product unchanged, add the correction in FP32 once.
            x=x-(t-following)*base
            x=x-(t-following)*delta.float()
        else:
            x=x-(t-following)*velocity
        trace.append(dict(step=index+1,t=t,next_t=following,correction_active=active,
                          delta_rms=float(delta.float().square().mean().sqrt()),gen_passes=1+int(cfg['cfg_text_scale']>1),
                          und_passes=int(active and not global_only)))
        if deployed and guidance_safe(cfg) and active:
            g=base.float()-conditional.float()
            trace[-1].update(raw_delta_rms=float(raw_delta.float().square().mean().sqrt()),
                guidance_projection_raw=float(guidance_coefficient(raw_delta,g)),
                guidance_projection_applied=float(guidance_coefficient(delta,g)))
    return runtime.image(x),trace
