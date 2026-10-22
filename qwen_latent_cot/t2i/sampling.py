"""Native BAGEL sampling using an exported online or EMA GEN overlay."""
import json
from pathlib import Path
import torch
from PIL import Image
from safetensors.torch import load_file
from ..bagel.accelerator import seeded_context
from ..bagel.modeling.bagel.qwen2_navit import NaiveCache
from .model import gen_parameter
from .checkpoint import KIND


def load_overlay(bundle,directory):
    directory=Path(directory);meta=json.loads((directory/'manifest.json').read_text())
    if meta['kind']!=KIND:raise ValueError('Expected native T2I export')
    weights=load_file(str(directory/'gen.safetensors'))
    expected={n for n,_ in bundle.model.named_parameters() if gen_parameter(n)}
    if set(weights)!=expected:raise ValueError('Export does not cover exactly native GEN parameters')
    result=bundle.model.load_state_dict(weights,strict=False)
    if result.unexpected_keys:raise ValueError('Unexpected export weights')
    return meta


@torch.no_grad()
def sample(bundle,prompt,seed,height,width,steps=50,shift=4.,cfg_scale=4.):
    model=bundle.model.eval();device=next(model.parameters()).device
    if height%16 or width%16 or max(height,width)>1024:raise ValueError('Use 16-aligned dimensions up to1024')
    move=lambda values:{k:v.to(device) if torch.is_tensor(v) else v for k,v in values.items()}
    empty=lambda:NaiveCache(len(model.language_model.model.layers))
    with seeded_context(device,seed),torch.autocast(device.type,dtype=torch.bfloat16):
        text,lengths,positions=model.prepare_prompts([0],[0],[prompt],bundle.tokenizer,bundle.token_ids)
        cache=model.forward_cache_update_text(empty(),**move(text))
        flow=model.prepare_vae_latent(lengths,positions,[(height,width)],bundle.token_ids)
        flow=move(flow);flow['past_key_values']=cache
        reference=model.prepare_vae_latent_cfg([0],[0],[(height,width)])
        flow.update({k.replace('cfg_','cfg_text_',1):v for k,v in move(reference).items()})
        flow['cfg_text_past_key_values']=empty()
        latent=model.generate_image(**flow,num_timesteps=steps,timestep_shift=shift,cfg_text_scale=cfg_scale,cfg_img_scale=1.,cfg_renorm_min=0.,cfg_renorm_type='global')[0]
        p=model.latent_patch_size;h=height//model.latent_downsample;w=width//model.latent_downsample
        latent=torch.einsum('hwpqc->chpwq',latent.reshape(h,w,p,p,model.latent_channel)).reshape(1,model.latent_channel,h*p,w*p)
        image=bundle.vae.decode(latent.to(next(bundle.vae.parameters()).dtype))[0]
    array=((image.float().clamp(-1,1)+1)*127.5).permute(1,2,0).cpu().byte().numpy()
    return Image.fromarray(array)
