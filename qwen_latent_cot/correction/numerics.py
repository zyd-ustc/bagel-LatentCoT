"""BAGEL mixed-precision execution on accelerator and the tiny CPU harness."""
import torch


def autocast_for(device):
    return torch.autocast(device.type,dtype=torch.bfloat16)


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
