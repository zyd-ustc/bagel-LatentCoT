"""Expensive training-free reference: observe a clean image, then native BAGEL edit.

Uses refs/Bagel/inferencer.py context ordering, native prepare/cache/generate APIs,
and native image transforms. Text is re-prefilled in full at native positions.
No image tokens or text tokens are pooled, selected, transplanted or truncated.
"""
from copy import deepcopy
import hashlib
import json
import torch
from .inferencer import T2IGenerator,to_device
from .modeling.bagel.qwen2_navit import NaiveCache
from .modeling.image_transforms import ImageTransform

OBSERVE = '''Compare the supplied image with the original image request below.
Report what is visibly present, independently of what the request says should be present.
Do not assume a requested count or relation is already visible. Mark uncertainty explicitly.
Return one JSON object with exactly these fields:
"observed": a string describing visible counts, attributes and relations;
"discrepancies": a list of specific, confident mismatches;
"preserve": a list of already-correct objects, attributes, layout and style to preserve;
"uncertain": a list of facts you cannot determine;
"edit": a minimal image-editing instruction addressing confident mismatches only.
If there are no confident mismatches, request preservation of the image.
Original request:
'''


def parse_feedback(text):
    raw=text.strip()
    if raw.startswith('```'):
        lines=raw.splitlines()
        if len(lines)<3 or lines[-1].strip()!='```':raise ValueError('incomplete feedback fence')
        raw='\n'.join(lines[1:-1])
    obj=json.loads(raw)
    if not isinstance(obj,dict) or set(obj)!={'observed','discrepancies','preserve','uncertain','edit'}:
        raise ValueError('feedback schema mismatch')
    if any(not isinstance(obj[k],str) or not obj[k].strip() for k in ('observed','edit')):
        raise ValueError('feedback requires observation and edit strings')
    if any(not isinstance(obj[k],list) or any(not isinstance(v,str) for v in obj[k]) for k in ('discrepancies','preserve','uncertain')):
        raise ValueError('feedback requires lists of strings')
    return obj


class NativeFeedback:
    def __init__(self,bundle):
        self.bundle=bundle;self.model=bundle.model;self.decoder=T2IGenerator(bundle)
        self.device=self.decoder.device
        if hasattr(self.model.language_model.model,'_memory_loop_runtime'):
            raise ValueError('native editing requires the original decoder, without an implicit loop wrapper')
        self.vae_transform=ImageTransform(1024,512,16)
        self.vit_transform=ImageTransform(980,224,14)

    def context(self):
        return dict(kv_lens=[0],ropes=[0],past_key_values=NaiveCache(self.model.config.llm_config.num_hidden_layers))

    def text(self,text,context):
        inputs,lengths,ropes=self.model.prepare_prompts(context['kv_lens'],context['ropes'],[text],self.bundle.tokenizer,self.bundle.token_ids)
        context['past_key_values']=self.model.forward_cache_update_text(context['past_key_values'],**to_device(inputs,self.device))
        context.update(kv_lens=lengths,ropes=ropes)
        return context

    def image(self,image,context,vae=True):
        image=self.vae_transform.resize_transform(image.convert('RGB'))
        if vae:
            inputs,lengths,ropes=self.model.prepare_vae_images(context['kv_lens'],context['ropes'],[image],self.vae_transform,self.bundle.token_ids)
            context['past_key_values']=self.model.forward_cache_update_vae(self.bundle.vae,context['past_key_values'],**to_device(inputs,self.device))
            context.update(kv_lens=lengths,ropes=ropes)
        inputs,lengths,ropes=self.model.prepare_vit_images(context['kv_lens'],context['ropes'],[image],self.vit_transform,self.bundle.token_ids)
        context['past_key_values']=self.model.forward_cache_update_vit(context['past_key_values'],**to_device(inputs,self.device))
        context.update(kv_lens=lengths,ropes=ropes)
        return context

    @torch.inference_mode()
    def observe(self,image,prompt,max_tokens=1024):
        with self.decoder.autocast():
            context=self.image(image,self.context(),vae=False)
            self.text(OBSERVE+prompt,context)
            inputs=self.model.prepare_start_tokens(context['kv_lens'],context['ropes'],self.bundle.token_ids)
            tokens=self.model.generate_text(past_key_values=context['past_key_values'],**to_device(inputs,self.device),
                max_length=max_tokens,do_sample=False,temperature=1.,end_token_id=self.bundle.token_ids['eos_token_id'])
        ids=tokens[:,0].tolist()
        raw=self.bundle.tokenizer.decode(ids).split('<|im_start|>',1)[-1].split('<|im_end|>',1)[0].strip()
        result={'raw':raw,'token_ids':ids,'complete':len(ids)<max_tokens,'valid_format':False,'parsed':None}
        try:
            if not result['complete']:raise ValueError('feedback reached token limit; do not silently truncate')
            result['parsed']=parse_feedback(raw);result['valid_format']=True
        except (ValueError,TypeError) as error:result['error']=str(error)
        return result

    @torch.inference_mode()
    def prepare_edit(self,image,instruction,seed):
        # Native interleave order: image -> instruction. Text-removed CFG keeps
        # the image; image-removed CFG independently prefills the same text.
        with self.decoder.autocast():
            context=self.image(image,self.context(),vae=True)
            image_only=deepcopy(context)
            self.text(instruction,context)
            text_only=self.text(instruction,self.context())
            shape=(image.height,image.width)
            flow=to_device(self.model.prepare_vae_latent(context['kv_lens'],context['ropes'],[shape],self.bundle.token_ids),self.device)
            count=(shape[0]//self.model.latent_downsample)*(shape[1]//self.model.latent_downsample)
            noise=torch.randn(count,self.model.patch_latent_dim,generator=torch.Generator().manual_seed(seed),dtype=torch.float32)
            noise_hash=hashlib.sha256(noise.numpy().tobytes()).hexdigest()
            flow['packed_init_noises']=noise.to(self.device);flow['past_key_values']=context['past_key_values']
            for branch,pre in (('text',image_only),('img',text_only)):
                cfg=self.model.prepare_vae_latent_cfg(pre['kv_lens'],pre['ropes'],[shape])
                flow.update({k.replace('cfg_',f'cfg_{branch}_',1):v for k,v in to_device(cfg,self.device).items()})
                flow[f'cfg_{branch}_past_key_values']=pre['past_key_values']
        meta={'conditional_lengths':context['kv_lens'],'image_only_lengths':image_only['kv_lens'],
            'text_only_lengths':text_only['kv_lens'],'conditional_rope':context['ropes'],
            'instruction':instruction,'instruction_token_ids':self.bundle.tokenizer.encode(instruction),
            'image_tokens_retained':'all_native_VAE_and_ViT','text_tokens_retained':'all',
            'kv_policy':'native_full_interleaved_context; no cross-context KV transplant'}
        return flow,noise_hash,meta

    @torch.inference_mode()
    def edit(self,image,instruction,seed,sampling,edit_sampling):
        flow,noise_hash,meta=self.prepare_edit(image,instruction,seed)
        with self.decoder.autocast():
            latents=self.model.generate_image(**flow,num_timesteps=sampling['num_timesteps'],
                timestep_shift=sampling['timestep_shift'],cfg_renorm_type=sampling['cfg_renorm_type'],
                enable_taylorseer=False,**edit_sampling)
            output=self.decoder.decode(latents[0],(image.height,image.width))
        return output,noise_hash,meta


def validate_native_edit(bundle,config):
    from PIL import Image
    engine=NativeFeedback(bundle)
    with torch.inference_mode(),engine.decoder.autocast():
        flow,noise_hash,meta=engine.prepare_edit(Image.new('RGB',(512,512),(128,128,128)),
            'Preserve the image composition and colors.',123)
        noise=flow.pop('packed_init_noises')
        kwargs=dict(flow,x_t=noise,timestep=torch.full((len(noise),),.7,device=engine.device),
            cfg_text_scale=config['edit_sampling']['cfg_text_scale'],cfg_img_scale=config['edit_sampling']['cfg_img_scale'],
            cfg_renorm_type='global')
        frozen={k:[(i,t.clone(),cache.value_cache[i].clone()) for i,t in cache.key_cache.items()]
            for k,cache in flow.items() if k.endswith('past_key_values')}
        v=bundle.model._forward_flow(**kwargs)
        repeated=bundle.model._forward_flow(**kwargs)
        same=all(torch.equal(flow[k].key_cache[i],a) and torch.equal(flow[k].value_cache[i],b) for k,rows in frozen.items() for i,a,b in rows)
        valid=bool(torch.isfinite(v).all() and torch.equal(v,repeated)) and same
    return {'passed':valid,'repeat_velocity_equal':torch.equal(v,repeated),'caches_immutable':same,
        'input_scope':'synthetic RGB + seeded noise; numerical check only','contexts':meta,'noise_sha256':noise_hash}
