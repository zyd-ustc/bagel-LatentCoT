"""User-run real-weight contracts for joint micro loops; no quality claims."""
from dataclasses import replace
import torch
from .joint_micro import residual_step
from ..evaluation.memory_rounds import tensor_metrics


def validate_joint_micro(bundle, generator, runtime):
    cfg = runtime.config
    decoder = runtime.decoder
    saved_progress, saved_observer = runtime.progress, runtime.joint_observer
    results = dict(micro_steps=cfg.micro_steps, step_scale=1/cfg.micro_steps,
        input_scope='seeded_gaussian_test_input_not_sampled_trajectory',
        K1_is_native_pipeline=False, semantic_gain_verified=False, quality_retention_verified=False,
        timesteps={})
    try:
        runtime.progress = cfg.progress_start
        with torch.inference_mode(), generator.autocast():
            flow, hashes = generator.prepare(['A red cube to the left of a blue sphere.',
                'Two yellow birds above a green tree.'], [(256,256), (256,384)], [123,456])
            noise = flow.pop('packed_init_noises')
            cache = flow['past_key_values']
            frozen = {i: (k, k.clone(), cache.value_cache[i], cache.value_cache[i].clone())
                      for i, k in cache.key_cache.items()}
            seed = runtime.layerwise.seeds[cache]
            hidden = {i: h.clone() for i, h in seed.layer_hidden.items()}
            for t in (1., .85):
                inputs = dict(flow, x_t=noise, timestep=torch.full((len(noise),), t, device=noise.device),
                              cfg_text_scale=1., cfg_renorm_type='global')
                runtime.joint_observer = None
                plain = bundle.model._forward_flow(**inputs)
                rows, contracts, actual, counts = [], [], {}, {}
                previous = None
                originals = []

                def observe(**e):
                    nonlocal previous
                    i, k = e['layer'], e['micro']
                    g, m = actual.pop((i,k,'gen')), actual.pop((i,k,'und'))
                    content = ~e['special_mask']
                    pparts = cache.key_cache[i].split(seed.lengths)
                    vparts = cache.value_cache[i].split(seed.lengths)
                    gparts = e['gen_kv'].keys.split(e['gen_kv'].lengths)
                    gvparts = e['gen_kv'].values.split(e['gen_kv'].lengths)
                    expected_k = torch.cat([torch.cat([p,x]) for p,x in zip(pparts,gparts)])
                    expected_v = torch.cat([torch.cat([p,x]) for p,x in zip(vparts,gvparts)])
                    check = dict(
                        native_GEN_input_KV_parity=torch.equal(g['keys'],e['gen_kv'].keys)
                            and torch.equal(g['values'],e['gen_kv'].values),
                        native_UND_input_KV_parity=torch.equal(m['keys'],e['memory_kv'].keys)
                            and torch.equal(m['values'],e['memory_kv'].values),
                        gen_reads_memory_only=torch.equal(g['past_keys'],e['memory_kv'].keys)
                            and torch.equal(g['past_values'],e['memory_kv'].values),
                        und_reads_prompt_and_old_GEN=torch.equal(m['past_keys'],expected_k)
                            and torch.equal(m['past_values'],expected_v),
                        gen_step_equation=torch.equal(e['gen_after'],residual_step(e['gen_before'],e['gen_native'],e['micro_steps'])),
                        memory_step_equation=torch.equal(e['memory_after'][content],
                            residual_step(e['memory_before'],e['memory_native'],e['micro_steps'])[content]),
                        special_hidden_pinned=torch.equal(e['memory_after'][~content],seed.layer_hidden[i][~content]),
                        special_KV_pinned=torch.equal(e['memory_kv'].keys[~content],e['native_prompt_kv'].keys[~content])
                            and torch.equal(e['memory_kv'].values[~content],e['native_prompt_kv'].values[~content]))
                    if previous is not None:
                        check['gen_continues'] = torch.equal(previous['gen_after'],e['gen_before'])
                        check['memory_content_continues'] = torch.equal(previous['memory_after'][content],e['memory_before'][content])
                    row = dict(layer=i,phase=e['phase'],micro=k,micro_steps=e['micro_steps'],step_scale=e['step_scale'],
                        gen_hidden_update=tensor_metrics(e['gen_before'][e['image_indexes']],e['gen_after'][e['image_indexes']]),
                        memory_hidden_update=tensor_metrics(e['memory_before'][content],e['memory_after'][content]))
                    check['finite'] = row['gen_hidden_update']['finite'] and row['memory_hidden_update']['finite']
                    if previous is not None and previous['layer']==i:
                        row['GEN_input_K_change'] = tensor_metrics(previous['gen_kv'].keys,e['gen_kv'].keys)
                        row['Memory_read_K_change'] = tensor_metrics(previous['memory_kv'].keys[content],e['memory_kv'].keys[content])
                        check['finite'] &= row['GEN_input_K_change']['finite'] and row['Memory_read_K_change']['finite']
                    contracts.append(check);rows.append(row);previous=e

                try:
                    for i in range(cfg.start_layer,len(decoder.layers)):
                        layer = decoder.layers[i];original = layer.forward_inference
                        originals.append((layer,original))
                        def stored(*, index=i, fn=original, **kw):
                            mode=kw['mode'];k=counts.get((index,mode),0);counts[index,mode]=k+1
                            temporary=kw['past_key_values'];query=kw['packed_query_indexes']
                            past_keys=temporary.key_cache[index].clone()
                            past_values=temporary.value_cache[index].clone()
                            # Only private per-branch caches are instrumented. Native P stays read-only.
                            out=fn(**dict(kw,update_past_key_values=True))
                            actual[index,k,mode]=dict(past_keys=past_keys,past_values=past_values,
                                keys=temporary.key_cache[index][query].clone(),values=temporary.value_cache[index][query].clone())
                            return out
                        layer.forward_inference=stored
                    runtime.joint_observer=observe
                    observed=bundle.model._forward_flow(**inputs)
                finally:
                    for layer,original in originals:layer.forward_inference=original
                    runtime.joint_observer=None
                expected_count=(cfg.end_layer-cfg.start_layer)*cfg.micro_steps+len(decoder.layers)-cfg.end_layer
                checks = dict(observer_velocity_parity=torch.equal(plain,observed),
                    exact_branch_event_count=len(rows)==expected_count and len(actual)==0,
                    native_branch_contracts=all(all(c.values()) for c in contracts),
                    conditional_velocity_finite=bool(torch.isfinite(plain).all()))
                runtime.config=replace(cfg,mode='BASE',micro_steps=1)
                native=bundle.model._forward_flow(**inputs)
                runtime.config=cfg
                checks['vs_base']=tensor_metrics(native,plain)
                cfg_inputs=dict(inputs,cfg_text_scale=4.)
                loop_cfg=bundle.model._forward_flow(**cfg_inputs)
                # Assert that the empty-cache branch really executes the original decoder.
                original_decoder=runtime.original;null_checks=[]
                def native_bypass(**kw):
                    out=original_decoder(**kw)
                    if kw.get('mode')=='gen' and not int(kw['key_values_lens'].sum()):
                        repeat=original_decoder(**kw)
                        null_checks.append(torch.equal(out.packed_query_sequence,repeat.packed_query_sequence))
                    return out
                runtime.original=native_bypass
                try:
                    repeat_cfg=bundle.model._forward_flow(**cfg_inputs)
                finally:runtime.original=original_decoder
                checks['cfg_repeatable']=torch.equal(loop_cfg,repeat_cfg)
                checks['cfg_velocity_finite']=bool(torch.isfinite(loop_cfg).all())
                checks['null_text_native_bypass']=null_checks==[True]
                runtime.config=replace(cfg,mode='BASE',micro_steps=1)
                native_cfg=bundle.model._forward_flow(**cfg_inputs)
                runtime.config=cfg
                checks['cfg_vs_base']=tensor_metrics(native_cfg,loop_cfg)
                checks['base_velocities_finite']=checks['vs_base']['finite'] and checks['cfg_vs_base']['finite']
                runtime.progress=cfg.progress_start/2 if cfg.progress_start>0 else (cfg.progress_end+1)/2
                if runtime.progress>cfg.progress_end or runtime.progress<cfg.progress_start:
                    checks['outside_window_native_parity']=torch.equal(native_cfg,bundle.model._forward_flow(**cfg_inputs))
                runtime.progress=cfg.progress_start
                checks['passed']=all(v for v in checks.values() if isinstance(v,bool))
                results['timesteps'][str(t)]=dict(checks=checks,layers=rows)
            results['prompt_cache_unchanged']=all(cache.key_cache[i] is k and cache.value_cache[i] is v
                and torch.equal(k,kcopy) and torch.equal(v,vcopy) for i,(k,kcopy,v,vcopy) in frozen.items())
            results['native_hidden_seeds_unchanged']=all(torch.equal(seed.layer_hidden[i],h) for i,h in hidden.items())
            results['noise_sha256']=hashes
            results['passed']=(results['prompt_cache_unchanged'] and results['native_hidden_seeds_unchanged']
                and all(x['checks']['passed'] for x in results['timesteps'].values()))
    finally:
        runtime.config=cfg;runtime.progress=saved_progress;runtime.joint_observer=saved_observer
    return results
