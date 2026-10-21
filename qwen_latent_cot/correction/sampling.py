"""One UND correction before native CFG/renorm and each selected Euler update."""
import math
import torch
from .model import proposal_from


def times(config):
    nodes=torch.linspace(1,0,config['sampler_steps']);shift=config['timestep_shift']
    return (shift*nodes/(1+(shift-1)*nodes)).tolist()


def active_steps(config): return math.ceil((config['sampler_steps']-1)/2)


def guided(conditional, reference, config):
    scale=config['cfg_text_scale']
    if scale<=1:return conditional
    result=reference+scale*(conditional-reference)
    kind=config['cfg_renorm_type']
    if kind=='global': original_norm=conditional.norm();new_norm=result.norm()
    elif kind in ('channel','text_channel'):
        original_norm=conditional.norm(dim=-1,keepdim=True);new_norm=result.norm(dim=-1,keepdim=True)
    else:raise ValueError('Unknown native CFG renorm')
    return result*(original_norm/(new_norm+1e-8)).clamp(min=config['cfg_renorm_min'],max=1.)


@torch.no_grad()
def sample(corrector,prompt,seed,*,native=False):
    runtime=corrector.runtime;cfg=corrector.config;flow=runtime.flow(prompt)
    n=len(flow['packed_vae_token_indexes']);c=runtime.model.latent_channel*runtime.model.latent_patch_size**2
    x=torch.randn((n,c),generator=torch.Generator().manual_seed(seed)).to(runtime.device)
    trace=[];grid=times(cfg)
    for index,(t,following) in enumerate(zip(grid,grid[1:])):
        active=not native and index<active_steps(cfg)
        base,context=runtime.predict(x,t,flow,capture=active)
        delta=corrector(proposal_from(x,t,base),t,flow,context) if active else torch.zeros_like(base)
        # Preserve native CFG arithmetic and dtype at the sampler boundary.
        conditional=(base.float()+delta.float()).to(base.dtype)
        reference=runtime.reference(x,t,flow) if cfg['cfg_text_scale']>1 else conditional
        velocity=guided(conditional,reference,cfg)
        if not torch.isfinite(velocity).all():raise FloatingPointError('Nonfinite sampling velocity')
        x=x-(t-following)*velocity
        trace.append(dict(step=index+1,t=t,next_t=following,correction_active=active,
                          delta_rms=float(delta.float().square().mean().sqrt()),gen_passes=1+int(cfg['cfg_text_scale']>1),
                          und_passes=int(active)))
    return runtime.image(x),trace
