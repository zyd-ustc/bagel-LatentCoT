"""Read-only round diagnostics on a common sampled x_t/t; no quality claims."""
from dataclasses import replace
from types import MethodType
from pathlib import Path
import hashlib
import json
import torch
from .memory_rounds import tensor_metrics


def fingerprint(tensors):
    digest=hashlib.sha256()
    for name,tensor in tensors:
        x=tensor.detach().cpu().contiguous()
        digest.update(str((name,tuple(x.shape),str(x.dtype))).encode())
        digest.update(x.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def detailed_metrics(a,b):
    stats=tensor_metrics(a,b)
    delta=b.float()-a.float()
    token_rms=delta.reshape(len(delta),-1).square().mean(-1).sqrt()
    stats.update(changed_fraction=float((a!=b).float().mean()),
        token_delta_rms_p50=float(token_rms.median()),
        token_delta_rms_p90=float(torch.quantile(token_rms,.9)),
        token_delta_rms_max=float(token_rms.max()))
    return stats


class IncrementCapture:
    """CPU snapshots of one previous round and its first-round reference.

    Memory rounds are writer counts (1..R); GEN rounds are body pass indexes
    (0..R). Adjacent snapshots are never compared across x_t or timesteps.
    """
    def __init__(self,case_dir,max_rounds,end_layer,tensor_layers=()):
        self.case_dir=Path(case_dir);self.case_dir.mkdir(parents=True,exist_ok=True)
        self.max_rounds=max_rounds;self.end_layer=end_layer;self.depth=0
        self.first={};self.previous={};self.raw={};self.rows=[];self.native_memory={}
        self.tensor_layers=set(tensor_layers)

    def record(self,series,component,layer,round_index,tensor,subsets,**labels):
        x=tensor.detach().cpu().clone()
        if layer in self.tensor_layers:
            self.raw[f'{series}/{component}/layer{layer}/round{round_index}']=x
        for subset,mask in subsets.items():
            selected=x if mask is None else x[mask.detach().cpu()]
            if not selected.numel():continue
            key=(series,component,layer,subset)
            row=dict(series=series,component=component,layer=layer,subset=subset,
                round=round_index,phase='body' if layer<self.end_layer else 'suffix',**labels)
            if key in self.previous:
                old_round,old=self.previous[key]
                if round_index<=old_round:raise ValueError('non-increasing diagnostic rounds')
                self.rows.append(dict(row,comparison='adjacent_round',from_round=old_round,
                                      **detailed_metrics(old,selected)))
                first_round,first=self.first[key]
                self.rows.append(dict(row,comparison='vs_first_round',from_round=first_round,
                                      **detailed_metrics(first,selected)))
            else:
                self.first[key]=(round_index,selected)
                self.rows.append(dict(row,comparison='first_snapshot',from_round=None,
                    finite=bool(torch.isfinite(selected).all()),candidate_norm=float(selected.float().norm()),
                    shape=list(selected.shape)))
            self.previous[key]=(round_index,selected)

    def final_hidden(self,layer,round_index,before,after,image_indexes,text_indexes):
        subsets={'all':None,'image':image_indexes,'boundary':text_indexes}
        self.record('final_depth','gen_input_hidden',layer,round_index,before,subsets)
        self.record('final_depth','gen_output_hidden',layer,round_index,after,subsets)

    def __call__(self,*,event,layer,phase,round,input_hidden,output_hidden,input_kv,read_kv,
                 image_indexes=None,text_indexes=None,reference_kv=None,special_mask=None,gen_feedback=None):
        if event=='gen_layer':
            if round==self.depth:
                self.final_hidden(layer,round,input_hidden,output_hidden,image_indexes,text_indexes)
                for component,x in (('gen_input_K',input_kv.keys),('gen_input_V',input_kv.values)):
                    self.record('final_depth',component,layer,round,x,{'all':None,'image':image_indexes,'boundary':text_indexes})
                for component,x in (('gen_read_memory_K',read_kv.keys),('gen_read_memory_V',read_kv.values)):
                    self.record('final_depth',component,layer,round,x,{'all':None,'content':~special_mask,'special':special_mask})
            if self.depth!=self.max_rounds or phase!='body':return
            subsets={'all':None,'image':image_indexes,'boundary':text_indexes}
            for component,x in (('gen_input_hidden',input_hidden),('gen_output_hidden',output_hidden),
                                ('gen_input_K',input_kv.keys),('gen_input_V',input_kv.values)):
                self.record('within_deepest',component,layer,round,x,subsets)
            memory_subsets={'all':None,'content':~special_mask,'special':special_mask}
            for component,x in (('gen_read_memory_K',read_kv.keys),('gen_read_memory_V',read_kv.values)):
                self.record('within_deepest',component,layer,round,x,memory_subsets)
        elif event=='memory_layer':
            if self.depth!=self.max_rounds:return
            subsets={'all':None,'content':~special_mask,'special':special_mask}
            for component,x in (('memory_input_hidden',input_hidden),('memory_output_hidden',output_hidden),
                                ('memory_self_K',input_kv.keys),('memory_self_V',input_kv.values),
                                ('memory_read_K',read_kv.keys),('memory_read_V',read_kv.values)):
                self.record('within_deepest',component,layer,round,x,subsets,gen_feedback=gen_feedback)
            for subset,mask in subsets.items():
                a=input_hidden.detach().cpu();b=output_hidden.detach().cpu();mask=None if mask is None else mask.cpu()
                a=a if mask is None else a[mask];b=b if mask is None else b[mask]
                if a.numel():self.rows.append(dict(series='within_deepest',component='memory_block_update',layer=layer,
                    phase=phase,round=round,from_round=round,subset=subset,comparison='within_block',
                    **detailed_metrics(a,b)))
            for component,x,reference in (('memory_read_K',read_kv.keys,reference_kv.keys),
                                          ('memory_read_V',read_kv.values,reference_kv.values)):
                key=(layer,component)
                if round==1:self.native_memory[key]=reference.detach().cpu().clone()
                for subset,mask in subsets.items():
                    a=self.native_memory[key];b=x.detach().cpu();mask=None if mask is None else mask.cpu()
                    a=a if mask is None else a[mask];b=b if mask is None else b[mask]
                    if a.numel():self.rows.append(dict(series='within_deepest',component=component,layer=layer,
                        phase=phase,round=round,from_round=0,subset=subset,comparison='vs_native_prompt_KV',
                        gen_feedback=gen_feedback,**detailed_metrics(a,b)))
        else:raise ValueError('unknown increment event')

    def save(self):
        if not self.rows or not all(row['finite'] for row in self.rows):
            raise ValueError('missing or nonfinite increment snapshots')
        (self.case_dir/'layers.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        if self.raw:torch.save(self.raw,self.case_dir/'selected_layer_tensors.pt')


def cache_fingerprint(runtime,cache):
    seed=runtime.layerwise.seeds[cache]
    tensors=[]
    for i in range(cache.num_layers):
        tensors.extend([(f'K{i}',cache.key_cache[i]),(f'V{i}',cache.value_cache[i]),
                        (f'H{i}',seed.layer_hidden[i])])
    return fingerprint(tensors)


@torch.no_grad()
def diagnose_fixed_state(model,runtime,flow_kwargs,case_dir,step_index,dt,max_rounds=4,
                         save_previews=None,tensor_layers=(),forward_flow=None):
    """Run Base and R1..Rmax independently, then replay the unobserved sampler call.

    Every candidate executes its ordinary suffix/readout exactly once. Raw
    conditional and null-text velocity come from native llm2vae forward hooks.
    CFG output is returned by BAGEL, including native global renormalization.
    """
    case_dir=Path(case_dir);capture=IncrementCapture(case_dir,max_rounds,runtime.config.end_layer,tensor_layers)
    cfg=runtime.config;old_observer=runtime.increment_observer;original=forward_flow or model._forward_flow
    x=flow_kwargs['x_t'];t=flow_kwargs['timestep'];x_before=x.clone();t_before=t.clone()
    cache=flow_kwargs['past_key_values'];before_cache=cache_fingerprint(runtime,cache)
    velocities=[];previous={};native={};deepest_velocity=None
    try:
        for depth in range(max_rounds+1):
            runtime.config=replace(cfg,mode='BASE' if depth==0 else cfg.mode,extra_rounds=depth)
            runtime.increment_observer=capture if depth>0 else None;capture.depth=depth
            outputs=[];handles=[];layer_originals=[]
            def head_hook(module,args,out):
                if not outputs:
                    capture.record('final_depth','gen_normalized_hidden',len(runtime.decoder.layers)-1,depth,args[0],
                        {'all':None,'image':flow_kwargs['packed_vae_token_indexes']})
                outputs.append(out.detach()[flow_kwargs['packed_vae_token_indexes']].cpu().clone())
            handles.append(model.llm2vae.register_forward_hook(head_hook))
            if depth==0:
                for index,layer in enumerate(runtime.decoder.layers):
                    fn=layer.forward_inference;layer_originals.append((layer,fn))
                    def observe_native(*,i=index,original_layer=fn,**kw):
                        result=original_layer(**kw)
                        if kw.get('mode')=='gen' and kw.get('past_key_values') is cache:
                            capture.final_hidden(i,0,kw['packed_query_sequence'],result[0],
                                kw['packed_vae_token_indexes'],kw['packed_text_indexes'])
                        return result
                    layer.forward_inference=observe_native
            try:v=original(**flow_kwargs)
            finally:
                for handle in handles:handle.remove()
                for layer,fn in layer_originals:layer.forward_inference=fn
            expected=2 if flow_kwargs.get('cfg_text_scale',1.)>1 else 1
            if len(outputs)!=expected:raise ValueError('unexpected native CFG/head branch count')
            current={'conditional':outputs[0],'cfg_post_renorm':v.detach().cpu().clone()}
            if expected==2:
                current['null_text']=outputs[1]
                current['cfg_pre_renorm_reconstructed_float32']=outputs[1].float()+flow_kwargs['cfg_text_scale']*(outputs[0].float()-outputs[1].float())
            for component,z in current.items():
                if not torch.isfinite(z).all():raise ValueError('nonfinite velocity')
                if depth==0:native[component]=z
                for comparison,reference in [('vs_base',native[component])]+([('adjacent_round',previous[component])] if depth else []):
                    metrics=detailed_metrics(reference,z)
                    velocities.append(dict(component=component,round=depth,from_round=0 if comparison=='vs_base' else depth-1,
                        comparison=comparison,step_index=step_index,timestep=float(t[0]),dt=dt,**metrics,
                        euler_delta_norm=dt*metrics['delta_norm'],
                        euler_delta_relative_xt=dt*metrics['delta_norm']/float(x_before.float().norm().clamp_min(1e-12))))
            previous=current
            if save_previews is not None:save_previews(depth,x_before-t_before[:,None]*v)
            if depth==max_rounds:deepest_velocity=v
        runtime.config=cfg;runtime.increment_observer=None
        replay=original(**flow_kwargs)
        contracts=dict(null_text_velocity_invariant=all(r['equal'] for r in velocities if r['component']=='null_text' and r['comparison']=='vs_base'),
            observer_velocity_parity=torch.equal(deepest_velocity,replay),
            xt_unchanged=torch.equal(x,x_before),timestep_unchanged=torch.equal(t,t_before),
            prompt_cache_and_seeds_unchanged=before_cache==cache_fingerprint(runtime,cache))
        if not all(contracts.values()):raise ValueError('diagnostic observation changed production state: '+str(contracts))
        capture.save()
        (case_dir/'velocity.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in velocities))
        torch.save({'x_t':x_before.cpu(),'timestep':t_before.cpu(),'step_index':step_index,'dt':dt},case_dir/'state.pt')
        summary=dict(step_index=step_index,timestep=float(t[0]),dt=dt,contracts=contracts,
            xt_sha256=fingerprint([('x_t',x_before)]),layer_rows=len(capture.rows),velocity_rows=len(velocities))
        (case_dir/'case.json').write_text(json.dumps(summary,indent=2)+'\n')
        return replay,summary
    finally:
        runtime.config=cfg;runtime.increment_observer=old_observer


@torch.no_grad()
def sample_with_diagnostics(generator,output_dir,prompt,shape,seed,steps=(0,4,9,19),max_rounds=4,
                            tensor_layers=(),previews=True):
    """Use the native sampler unchanged, on the Early20 Rmax trajectory."""
    runtime=generator.runtime;model=generator.model;out=Path(output_dir)
    flow,hashes=generator.prepare([prompt],[shape],[seed]);original=model._forward_flow
    schedule=torch.linspace(1,0,50,device=generator.device);schedule=3*schedule/(1+2*schedule)
    cases=[];index=0
    def forward(this,**kwargs):
        nonlocal index
        step=index;index+=1
        runtime.progress=step/48;runtime.step_index=step
        if step not in steps:return original(**kwargs)
        directory=out/f'step_{step:02d}';directory.mkdir(parents=True,exist_ok=True)
        def preview(depth,latent):generator.decode(latent,shape).save(directory/f'x0_R{depth}.png')
        result,case=diagnose_fixed_state(model,runtime,kwargs,directory,step,float(schedule[step]-schedule[step+1]),
            max_rounds,preview if previews else None,tensor_layers,forward_flow=original)
        cases.append(case)
        print(f'Diagnostic step={step} t={case["timestep"]:.5f} contracts={case["contracts"]}',flush=True)
        return result
    model._forward_flow=MethodType(forward,model)
    try:
        with generator.autocast():
            latents=model.generate_image(**flow,num_timesteps=50,timestep_shift=3.,cfg_text_scale=4.,
                cfg_img_scale=1.,cfg_renorm_type='global',enable_taylorseer=False)
            generator.decode(latents[0],shape).save(out/'trajectory_final.png')
        if index!=49 or [c['step_index'] for c in cases]!=list(steps):raise ValueError('incomplete native trajectory diagnostic')
        return dict(noise_sha256=hashes[0],cases=cases,denoiser_steps=index)
    finally:
        model._forward_flow=original;runtime.clear_prompt_state()
