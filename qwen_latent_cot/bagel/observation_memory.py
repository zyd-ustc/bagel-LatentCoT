"""One native visual observation and full-depth UND write at fixed x_t/t.

The image/text edit context is native. The static diagnostic only disables
image reads during text prefill; it preserves text IDs, positions, capacity and
GEN's complete visual context. No body-end hidden is sent back to layer zero.
"""
from copy import deepcopy
from types import MethodType
import hashlib
import time
import torch
from .feedback import NativeFeedback
from .inferencer import T2IGenerator, InvalidGeneratedImage, to_device
from .modeling.bagel.qwen2_navit import NaiveCache
from .accelerator import synchronize,seeded_context


def tensor_hash(tensor):
    raw=tensor.detach().contiguous().cpu().view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def memory_context(context, prefix_length):
    """Diagnostic text-only context; keep native shifted RoPE, drop image KV.

    This context is deliberately non-native and must be calibrated. Full-context
    QA alone cannot establish information in text Memory, as it also sees images.
    """
    cache=context['past_key_values'];out=NaiveCache(cache.num_layers)
    for layer,keys in cache.key_cache.items():
        out.key_cache[layer]=keys[prefix_length:].clone()
        out.value_cache[layer]=cache.value_cache[layer][prefix_length:].clone()
    return dict(past_key_values=out,kv_lens=[context['kv_lens'][0]-prefix_length],ropes=list(context['ropes']))


class ObservationConditions(NativeFeedback):
    @torch.inference_mode()
    def prepare_conditions(self,image,prompt,observe=True,visual_seed=0):
        with seeded_context(self.device,visual_seed):
            flow,conditional,visual,meta=self._prepare_conditions(image,prompt,observe)
        meta['visual_posterior_seed']=int(visual_seed)
        return flow,conditional,visual,meta

    def _prepare_conditions(self,image,prompt,observe):
        visual=self.image(image,self.context(),vae=True)
        prefix=visual['kv_lens'][0]
        conditional=deepcopy(visual)
        # prepare_prompts is the native tokenizer/embedding route. Static text is
        # encoded without image KV at the exact same shifted text positions.
        if observe:
            self.text(prompt,conditional)
        else:
            text_only=dict(past_key_values=NaiveCache(self.model.config.llm_config.num_hidden_layers),
                kv_lens=[0],ropes=list(visual['ropes']))
            self.text(prompt,text_only)
            for layer in range(self.model.config.llm_config.num_hidden_layers):
                conditional['past_key_values'].key_cache[layer]=torch.cat([
                    visual['past_key_values'].key_cache[layer],text_only['past_key_values'].key_cache[layer]])
                conditional['past_key_values'].value_cache[layer]=torch.cat([
                    visual['past_key_values'].value_cache[layer],text_only['past_key_values'].value_cache[layer]])
            conditional.update(kv_lens=[prefix+text_only['kv_lens'][0]],ropes=list(text_only['ropes']))
        # Native text-removed CFG keeps the full image, image-removed CFG has
        # only the original text at its own native positions.
        image_removed=self.text(prompt,self.context())
        shape=(image.height,image.width)
        flow=to_device(self.model.prepare_vae_latent(conditional['kv_lens'],conditional['ropes'],[shape],self.bundle.token_ids),self.device)
        flow.pop('packed_init_noises')  # Always reuse the sampler's current x_t.
        flow['past_key_values']=conditional['past_key_values']
        for branch,context in (('text',visual),('img',image_removed)):
            cfg=self.model.prepare_vae_latent_cfg(context['kv_lens'],context['ropes'],[shape])
            flow.update({k.replace('cfg_',f'cfg_{branch}_',1):v for k,v in to_device(cfg,self.device).items()})
            flow[f'cfg_{branch}_past_key_values']=context['past_key_values']
        inputs,_,_=self.model.prepare_prompts([prefix],visual['ropes'],[prompt],self.bundle.tokenizer,self.bundle.token_ids)
        text_length=conditional['kv_lens'][0]-prefix
        if text_length!=len(inputs['packed_text_ids']):raise ValueError('full prompt capacity changed')
        meta={'observation_enabled':observe,'visual_prefix_length':prefix,'memory_length':text_length,
            'text_ids':inputs['packed_text_ids'].tolist(),'text_positions':inputs['packed_text_position_ids'].tolist(),
            'conditional_lengths':conditional['kv_lens'],'conditional_rope':conditional['ropes'],
            'text_removed_lengths':visual['kv_lens'],'image_removed_lengths':image_removed['kv_lens'],
            'writer':'one_continuous_native_full_depth_causal_UND_prefill',
            'memory_kv':'native_attention_input_projection; no output re-projection',
            'visual_capacity':'all_native_VAE_and_ViT_tokens','capacity':'all_original_prompt_tokens',
            'static_control':'image_read_disabled_only_during_text_prefill' if not observe else None}
        return flow,conditional,visual,meta

    @torch.inference_mode()
    def answer_context(self,context,question,max_tokens=16):
        # Never append probe questions/answers to a cache consumed by GEN.
        probe=deepcopy(context)
        with self.decoder.autocast():
            self.text(question+' Answer briefly using visible evidence. If unclear, answer unknown.',probe)
            inputs=self.model.prepare_start_tokens(probe['kv_lens'],probe['ropes'],self.bundle.token_ids)
            tokens=self.model.generate_text(past_key_values=probe['past_key_values'],**to_device(inputs,self.device),
                max_length=max_tokens,do_sample=False,temperature=1.,end_token_id=self.bundle.token_ids['eos_token_id'])
        ids=tokens[:,0].tolist()
        raw=self.bundle.tokenizer.decode(ids).split('<|im_start|>',1)[-1].split('<|im_end|>',1)[0].strip()
        return {'answer':raw,'token_ids':ids,'complete':len(ids)<max_tokens,
            'question':question,'labels':'pending_manual_observation_of_saved_early_preview'}

    def probe(self,image,conditional,visual,meta,questions,max_tokens):
        with self.decoder.autocast():
            vit_only=self.image(image,self.context(),vae=False)
        contexts={'native_vit_image':vit_only,'native_edit_visual':visual,'full_edit_context':conditional,
            'memory_only_diagnostic':memory_context(conditional,meta['visual_prefix_length'])}
        return {name:{'native_context':name!='memory_only_diagnostic',
            'answers':[self.answer_context(context,q,max_tokens) for q in questions]}
            for name,context in contexts.items()}


class ObservationMemoryGenerator(T2IGenerator):
    def __init__(self,bundle,step,observe,trace_dir,questions=(),probe_max_tokens=16,edit_sampling=None,end_step=None):
        super().__init__(bundle)
        if hasattr(self.model.language_model.model,'_memory_loop_runtime'):
            raise ValueError('observation Memory requires an unwrapped native decoder')
        self.observation_step=step;self.observe=observe;self.trace_dir=trace_dir
        self.end_step=step+1 if end_step is None else end_step
        self.questions=list(questions);self.probe_max_tokens=probe_max_tokens
        self.edit_sampling=edit_sampling or {'cfg_text_scale':3.,'cfg_img_scale':1.5,'cfg_interval':[.4,1.]}
        self.engine=ObservationConditions(bundle);self.events=[]

    @torch.inference_mode()
    def generate(self,prompts,shapes,seeds,num_timesteps=50,**sampler):
        if len(prompts)!=1 or len(shapes)!=1 or len(seeds)!=1:raise ValueError('observation Memory runs one paired sample at a time')
        end_step=getattr(self,'end_step',self.observation_step+1)
        if not 0<=self.observation_step<end_step<=num_timesteps-1:raise ValueError('observation window outside native schedule')
        flow,hashes=self.prepare(prompts,shapes,seeds);self.events=[]
        original=self.model._forward_flow;step=0;held=None;conditional=None;visual=None;meta=None
        schedule=torch.linspace(1,0,num_timesteps,device=self.device)
        shift=sampler.get('timestep_shift',1.)
        schedule=shift*schedule/(1+(shift-1)*schedule)
        def fingerprint():
            prefix=meta['visual_prefix_length'];cache=conditional['past_key_values']
            return {str(i):[tensor_hash(k[prefix:]),tensor_hash(cache.value_cache[i][prefix:])]
                    for i,k in cache.key_cache.items()}
        def forward(this,**kwargs):
            nonlocal step,held,conditional,visual,meta
            current_step=step;step+=1
            native=original(**kwargs)
            if not self.observation_step<=current_step<end_step:
                if current_step==end_step:held=conditional=visual=meta=None
                return native
            x=kwargs['x_t'];t=kwargs['timestep'];before=tensor_hash(x);time_before=tensor_hash(t)
            first=current_step==self.observation_step
            if first:
                estimate=x-t[:,None]*native
                if not torch.isfinite(estimate).all():raise InvalidGeneratedImage('nonfinite early clean-image estimate')
                preview=self.decode(estimate,shapes[0])
                self.trace_dir.mkdir(parents=True,exist_ok=True)
                preview_path=self.trace_dir/'early_prediction.png';preview.save(preview_path)
                visual_seed=int(before[:16],16)%(2**63)
                held,conditional,visual,meta=self.engine.prepare_conditions(preview,prompts[0],self.observe,visual_seed)
                if 'x_t' in held or 'timestep' in held or 'packed_init_noises' in held:
                    raise ValueError('conditions may not replace the current noise/time')
            args={**kwargs,**held}
            interval=self.edit_sampling['cfg_interval'];tv=float(t[0])
            active=interval[0]<tv<=interval[1]
            args.update(cfg_text_scale=self.edit_sampling['cfg_text_scale'] if active else 1.,
                cfg_img_scale=self.edit_sampling['cfg_img_scale'] if active else 1.)
            corrected=original(**args)
            if not torch.isfinite(corrected).all():raise InvalidGeneratedImage('nonfinite conditioned velocity')
            if tensor_hash(x)!=before or tensor_hash(t)!=time_before:raise RuntimeError('feedback modified x_t or timestep')
            probes={};probe_seconds=0.
            if first:
                prefix=meta['visual_prefix_length'];cache=conditional['past_key_values']
                payload={'x_t':x.detach().cpu(),'timestep':t.detach().cpu(),'native_velocity':native.detach().cpu(),
                    'updated_velocity':corrected.detach().cpu(),'clean_image_estimate':estimate.detach().cpu(),
                    'memory_keys':{i:k[prefix:].detach().cpu() for i,k in cache.key_cache.items()},
                    'memory_values':{i:v[prefix:].detach().cpu() for i,v in cache.value_cache.items()}}
                torch.save(payload,self.trace_dir/'state.pt')
                begin=time.perf_counter()
                probes=self.engine.probe(preview,conditional,visual,meta,self.questions,self.probe_max_tokens) if self.questions else {}
                synchronize(self.device)
                probe_seconds=time.perf_counter()-begin if self.questions else 0.
            delta=(corrected-native).float();dt=float(schedule[current_step]-schedule[current_step+1])
            event={'step_index':current_step,'timestep':tv,'x_t_sha256':before,'timestep_sha256':time_before,
                'x_t_unchanged':True,'timestep_unchanged':True,'updates':int(first),'contexts':meta,'probes':probes,
                'probe_seconds':probe_seconds,'velocity_max_abs_change':float((corrected-native).abs().max()),
                'velocity_delta_rms':float(delta.square().mean().sqrt()),
                'velocity_relative_delta':float(delta.norm()/native.float().norm().clamp_min(1e-12)),
                'euler_dt':dt,'euler_delta_relative_xt':float(dt*delta.norm()/x.float().norm().clamp_min(1e-12)),
                'observation_source_step':self.observation_step,'conditioning_end_step':end_step,
                'prediction_scope':'early_model_estimate_not_final_image; image clipped only by native VAE display',
                'scope':'diagnostics_only; questions_never_enter_GEN',
                'active_edit_cfg':{'text':args['cfg_text_scale'],'image':args['cfg_img_scale']},
                'lifecycle':'one_writer_then_fixed_cache_until_exclusive_end_step'}
            if first:
                event.update(source_preview=str(preview_path),preview_sha256=file_hash(preview_path),
                    saved_state=str(self.trace_dir/'state.pt'),state_sha256=file_hash(self.trace_dir/'state.pt'),
                    memory_fingerprint=fingerprint())
            if current_step==end_step-1:
                original_fingerprint=event['memory_fingerprint'] if first else self.events[0]['memory_fingerprint']
                event['held_memory_unchanged']=fingerprint()==original_fingerprint
                if not event['held_memory_unchanged']:raise RuntimeError('held Memory changed without a writer')
            self.events.append(event)
            return corrected
        self.model._forward_flow=MethodType(forward,self.model)
        try:
            with self.autocast():
                latents=self.model.generate_image(**flow,num_timesteps=num_timesteps,cfg_img_scale=1.,enable_taylorseer=False,**sampler)
                images=[self.decode(latent,shape) for latent,shape in zip(latents,shapes)]
            if step!=num_timesteps-1 or len(self.events)!=end_step-self.observation_step or sum(e['updates'] for e in self.events)!=1:
                raise RuntimeError('feedback coverage or writer count differs from configured window')
            return images,hashes
        except InvalidGeneratedImage as error:
            error.noise_hashes=hashes;raise
        finally:
            held=conditional=visual=meta=None
            self.model._forward_flow=original
