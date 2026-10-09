"""User-run real-weight contracts, not semantic/quality verification."""
import torch
from PIL import Image
from .observation_memory import ObservationConditions
from .feedback import NativeFeedback


def validate_observation(bundle,config):
    engine=ObservationConditions(bundle);native=NativeFeedback(bundle)
    image=Image.new('RGB',(512,512),(128,128,128));prompt='A gray square.'
    versions={n:p._version for n,p in bundle.model.named_parameters()}
    with torch.inference_mode(),engine.decoder.autocast():
        observed,context,visual,meta=engine.prepare_conditions(image,prompt,True)
        static,static_context,_,static_meta=engine.prepare_conditions(image,prompt,False)
        reference,_,_=native.prepare_edit(image,prompt,123)
        reference.pop('packed_init_noises')
        same=lambda a,b:all(torch.equal(a.key_cache[i],b.key_cache[i]) and torch.equal(a.value_cache[i],b.value_cache[i]) for i in a.key_cache)
        parity=same(context['past_key_values'],reference['past_key_values'])
        layout_equal=set(observed)==set(reference) and all(
            same(value,reference[name]) if name.endswith('past_key_values') else
            torch.equal(value,reference[name]) if isinstance(value,torch.Tensor) else value==reference[name]
            for name,value in observed.items())
        p=meta['visual_prefix_length']
        prefix_same=all(torch.equal(observed['past_key_values'].key_cache[i][:p],static['past_key_values'].key_cache[i][:p]) and
            torch.equal(observed['past_key_values'].value_cache[i][:p],static['past_key_values'].value_cache[i][:p]) for i in context['past_key_values'].key_cache)
        length_same=meta['text_ids']==static_meta['text_ids'] and meta['text_positions']==static_meta['text_positions'] and meta['conditional_lengths']==static_meta['conditional_lengths']
        cache=observed['past_key_values'];frozen={i:(k.clone(),cache.value_cache[i].clone()) for i,k in cache.key_cache.items()}
        count=1024;x=torch.randn(count,bundle.model.patch_latent_dim,generator=torch.Generator().manual_seed(123)).to(engine.device)
        t=torch.full((count,),.7,device=engine.device);x_before=x.clone();t_before=t.clone()
        kwargs=dict(observed,x_t=x,timestep=t,cfg_text_scale=3.,cfg_img_scale=1.5,cfg_renorm_type='global')
        v=bundle.model._forward_flow(**kwargs);repeat=bundle.model._forward_flow(**kwargs)
        reference_v=bundle.model._forward_flow(**dict(reference,x_t=x,timestep=t,
            cfg_text_scale=3.,cfg_img_scale=1.5,cfg_renorm_type='global'))
        # Probe must not mutate the canonical cache or reach the generation path.
        engine.answer_context(context,'What is the dominant color?',max_tokens=2)
        immutable=all(torch.equal(cache.key_cache[i],a) and torch.equal(cache.value_cache[i],b) for i,(a,b) in frozen.items())
    result={'native_edit_context_exact_parity':parity,'visual_prefix_equal':prefix_same,
        'native_edit_flow_layout_exact_parity':layout_equal,'native_edit_velocity_exact_parity':torch.equal(v,reference_v),
        'static_observed_token_positions_capacity_equal':length_same,'velocity_repeat_equal':torch.equal(v,repeat),
        'velocity_finite':bool(torch.isfinite(v).all()),'x_t_unchanged':torch.equal(x,x_before),
        'timestep_unchanged':torch.equal(t,t_before),'canonical_cache_unchanged_after_flow_and_probe':immutable,
        'weights_unchanged':versions=={n:p._version for n,p in bundle.model.named_parameters()}}
    passed=all(result.values())
    return {'passed':passed,'contracts':result,'contexts':meta,'static_contexts':static_meta,
        'scope':'synthetic RGB and seeded noise; no semantics or quality claim'}
