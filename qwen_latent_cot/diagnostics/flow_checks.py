"""E0 on actual native BAGEL: numerical layout and live input-gradient contracts."""
import hashlib
from dataclasses import replace
import torch
from .flow_training import conditional_gen, flow_mse
from .loop_training import memory_loop
from .visual_objectives import semantic_loss
from .record import FrozenGuard, write_json
from ..bagel.accelerator import synchronize, reset_peak_memory_stats, max_memory_allocated


def parameter_hash(*modules):
    digest = hashlib.sha256()
    for module_index, module in enumerate(modules):
        for name, parameter in module.named_parameters():
            digest.update(f'{module_index}:{name}:{tuple(parameter.shape)}:{parameter.dtype}'.encode())
            flat = parameter.detach().reshape(-1)
            for offset in range(0, len(flat), 4*1024*1024):
                digest.update(flat[offset:offset+4*1024*1024].cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def check(student, rows, output):
    """Engineering-only updates of adapters; never saves them as a trained model."""
    runtime = student.runtime
    if len(rows) < 2: raise ValueError('E0 requires at least two distinct fixed image states')
    if any(p.requires_grad for p in student.writer.parameters()):
        raise ValueError('E0 first checks require the original text writer frozen')
    guard = FrozenGuard(runtime.model, runtime.bundle.vae, student.text_teacher, student.writer)
    print('E0: hashing frozen native parameters...', flush=True)
    before = parameter_hash(runtime.model, runtime.bundle.vae)
    reset_peak_memory_stats(runtime.device); records = []
    originals = {name: value.detach().clone() for name, value in student.visual.state_dict().items()}
    import time
    synchronize(runtime.device); start = time.perf_counter()
    for index, t in enumerate((.25, .75, .9)):
        print(f'E0: native/zero/layout contracts t={t}', flush=True)
        state = runtime.training_state(rows[index % len(rows)], t, index)
        with torch.no_grad():
            a = runtime.velocity(state.x_t, t, state.flow)[0]
            b = runtime.velocity(state.x_t, t, state.flow)[0]
            c = conditional_gen(runtime, state)
            repeat_error = float((a.float()-b.float()).abs().max())
            eps = torch.finfo(a.dtype).eps
            rtol = max(2*eps, runtime.config.get('numeric_rtol', .02))
            atol = max(2*repeat_error, 2*eps*float(a.float().square().mean().sqrt()), runtime.config.get('numeric_atol', .002))
            torch.testing.assert_close(a, c.velocity, rtol=rtol, atol=atol)
            zero = student.write(c.source, t)
            old = student.write(c.source, t, visual=False)
            for kv, previous in zip(zero.layers, old.layers):
                torch.testing.assert_close(kv.keys, previous.keys, rtol=0, atol=0)
                torch.testing.assert_close(kv.values, previous.values, rtol=0, atol=0)
            text_source = runtime.text('a red object')[0]
            off = student.write(text_source, None, visual=False)
            direct = runtime.memory(text_source)
            for kv, previous in zip(off.layers, direct.layers):
                torch.testing.assert_close(kv.keys, previous.keys, rtol=0, atol=0)
            fixed = student.writer.fixed_bank(runtime.seeds)
            masked = conditional_gen(runtime, state, fixed, masked=True, capture=False)
            shifted_flow = dict(state.flow, packed_position_ids=state.flow['packed_position_ids']+16)
            shifted = conditional_gen(runtime, replace(state, flow=shifted_flow), capture=False)
            torch.testing.assert_close(masked.velocity, shifted.velocity, rtol=rtol, atol=atol)
        records.append(dict(t=t, sample_id=rows[index % len(rows)]['sample_id'],
                            native_repeat_max_error=repeat_error, rtol=rtol, atol=atol,
                            conditional_native_max_error=float((a.float()-c.velocity.float()).abs().max()),
                            masked_shifted_max_error=float((masked.velocity.float()-shifted.velocity.float()).abs().max()),
                            zero_visual_equals_C1_reader=True, visual_off_equals_text_writer=True))
    state = runtime.training_state(rows[0], .75, 43)
    cache = state.flow['past_key_values']
    cache_before = [(cache.key_cache[i].clone(), cache.value_cache[i].clone()) for i in range(len(runtime.decoder.layers))]
    optimizer = torch.optim.AdamW(student.visual.parameters(), lr=1e-4)
    gradients = []
    for step in range(2):
        print(f'E0: complete R2 GEN backward step={step+1}', flush=True)
        optimizer.zero_grad(set_to_none=True)
        loop = memory_loop(student, state, 2, checkpoint_activations=runtime.config.get('activation_checkpointing', True))
        loss = torch.stack([flow_mse(exit, state) for exit in loop.exits]).mean()
        loss.backward()
        audit = []
        for index, layer in enumerate(student.visual.layers):
            norms = {name: float(p.grad.float().norm()) if p.grad is not None else None
                     for name, p in layer.named_parameters()}
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in layer.parameters()):
                raise RuntimeError('missing/non-finite visual gradient through complete GEN')
            if norms['up.weight'] <= 0 or (step > 0 and norms['down.weight'] <= 0):
                raise RuntimeError('complete GEN gradient did not reach both visual projections')
            audit.append(dict(layer=index, norms=norms))
        optimizer.step(); guard.check()
        gradients.append(dict(step=step+1, flow_loss=float(loss.detach()), groups=audit))
    with torch.no_grad():
        fixed = memory_loop(student, state, 2, feedback='fixed')
        diagnostic = memory_loop(student, state, 1, diagnostics=True)
        torch.testing.assert_close(fixed.exits[0].velocity, fixed.exits[1].velocity, rtol=rtol, atol=atol)
        torch.testing.assert_close(fixed.exits[0].velocity, diagnostic.exits[0].velocity, rtol=rtol, atol=atol)
    for index, (k, v) in enumerate(cache_before):
        torch.testing.assert_close(k, cache.key_cache[index], rtol=0, atol=0)
        torch.testing.assert_close(v, cache.value_cache[index], rtol=0, atol=0)
    print('E0: rechecking frozen parameter hashes...', flush=True)
    after = parameter_hash(runtime.model, runtime.bundle.vae)
    if before != after: raise RuntimeError('native model parameter SHA256 changed')
    student.visual.load_state_dict(originals, strict=True)
    student.visual.zero_grad(set_to_none=True)
    guard.check(); synchronize(runtime.device)
    result = dict(status='E0_passed', native_parameters_hash_before=before, native_parameters_hash_after=after,
                  contracts=records, gradients=gradients, cache_unchanged=True,
                  same_state_R2=True, fixed_M1_equals_R1=True, diagnostic_flag_output_unchanged=True,
                  peak_memory_bytes=max_memory_allocated(runtime.device), seconds=time.perf_counter()-start,
                  device=str(runtime.device), dtype=str(runtime.decoder.embed_tokens.weight.dtype),
                  scope='numerical and gradient contracts only; no semantic/final-image quality claim')
    write_json(output/'e0.json', result)
    return result
