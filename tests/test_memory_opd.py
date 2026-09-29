import pytest
import torch

from qwen_latent_cot.bagel.memory_init import initialize_memory_from_prompt_hidden
from qwen_latent_cot.bagel.memory_reader import install_memory_readers
from qwen_latent_cot.bagel.opd import opd_velocity_loss
from test_memory_grounding import tiny_bagel, flow_kwargs


def test_prompt_content_uniform_initializer_is_detached_and_deterministic():
    hidden = torch.arange(1 * 7 * 8, dtype=torch.float32).reshape(1, 7, 8).requires_grad_()
    mask = torch.ones(1, 7, dtype=torch.bool)
    content = torch.tensor([[False, True, True, False, True, True, False]])
    result = initialize_memory_from_prompt_hidden(hidden_cache={12: hidden},
        prompt_mask=mask, content_mask=content, layer_index=12, num_slots=3)
    assert result.shape == (3, 8) and not result.requires_grad
    assert torch.equal(result, hidden.detach()[0, [1, 4, 5]])
    assert torch.equal(result, initialize_memory_from_prompt_hidden(hidden_cache={12: hidden},
        prompt_mask=mask, content_mask=content, layer_index=12, num_slots=3))


def test_short_prompt_repeats_without_random_scale_and_empty_content_rejected():
    hidden = torch.randn(1, 3, 8, dtype=torch.float32)
    content = torch.tensor([[False, True, False]])
    memory = initialize_memory_from_prompt_hidden(hidden_cache=hidden,
        prompt_mask=torch.ones_like(content), content_mask=content,
        layer_index=12, num_slots=8)
    assert (memory - hidden[0, 1]).abs().max() <= 1.1e-5
    with pytest.raises(ValueError, match="no eligible"):
        initialize_memory_from_prompt_hidden(hidden_cache=hidden,
            prompt_mask=torch.zeros_like(content), layer_index=12, num_slots=8)


def test_reader_zero_effect_native_parity_and_isolated_gradient():
    model = tiny_bagel()
    names = install_memory_readers(model, start=1, end=2, rank=2, alpha=2)
    assert names and all("memory_reader.output_adapter" in name for name in names)
    kwargs = flow_kwargs(memory=False)
    memory = torch.randn(3, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        native = model._forward_flow(**kwargs)
        student = model._forward_flow(**kwargs, opd_memory_hidden=(memory,),
                                      opd_reader_start=1, opd_reader_end=2)
    assert torch.equal(native, student)
    target = native.float() + 1
    out = model._forward_flow(**kwargs, opd_memory_hidden=(memory,),
                              opd_reader_start=1, opd_reader_end=2)
    loss = opd_velocity_loss(out, target)
    loss.loss.backward()
    grads = {name: p.grad for name, p in model.named_parameters()}
    assert all(grads[name] is not None for name in names)
    assert all(grads[name] is None for name in grads if name not in names)
    assert any(grads[name].abs().sum() > 0 for name in names if name.endswith("B.weight"))
    assert loss.teacher_rms > 0


def test_opd_loss_does_not_train_teacher():
    student = torch.tensor([1., 2.], requires_grad=True)
    teacher = torch.tensor([3., 4.], requires_grad=True)
    result = opd_velocity_loss(student, teacher)
    assert result.loss == 4
    result.loss.backward()
    assert student.grad is not None and teacher.grad is None


def test_bagel_opd_api_reads_frozen_memory_and_retains_native_zero_parity(monkeypatch):
    model = tiny_bagel()
    install_memory_readers(model, start=1, end=2, rank=2, alpha=2)
    flow = flow_kwargs(memory=False)
    read = flow_kwargs(memory=True)
    read.pop("loop_memory")
    condition = dict(
        flow_kwargs=flow, read_kwargs=read,
        prompt_hidden={1: torch.randn(1, 4, 8, dtype=torch.bfloat16)},
        prompt_mask=torch.ones(1, 4, dtype=torch.bool),
        content_mask=torch.tensor([[False, True, True, False]]),
        num_slots=2,
    )
    body_inputs=[]
    layer=model.language_model.model.layers[1]
    original=layer.forward_inference
    def observe(*args,**kwargs):
        if kwargs.get("packed_memory_token_indexes") is not None:
            body_inputs.append(kwargs["packed_query_sequence"].detach().clone())
        return original(*args,**kwargs)
    monkeypatch.setattr(layer,"forward_inference",observe)
    with torch.no_grad():
        native = model._forward_flow(**flow)
    student, memory = model.forward_memory_opd_velocity(
        x_t=flow["x_t"], timestep=flow["timestep"], condition=condition,
        memory_body_start=1, memory_body_end=2, return_memory=True)
    assert torch.equal(native, student)
    assert len(memory) == 1 and memory[0].shape == (2, 8)
    assert not memory[0].requires_grad
    indexes=read["packed_loop_token_indexes"]
    assert torch.equal(body_inputs[0][indexes],
        condition["prompt_hidden"][1][0,1:3])
    assert model.last_opd_memory_stats["Mread_rms"] > 0
    with pytest.raises(ValueError, match="initializer"):
        model.forward_memory_opd_velocity(x_t=flow["x_t"], timestep=flow["timestep"],
            condition=condition, memory_init_strategy="random", memory_body_start=1,
            memory_body_end=2)


def test_eval_mode_opd_activation_checkpoint_preserves_reader_gradient():
    model=tiny_bagel()
    install_memory_readers(model,start=1,end=2,rank=2,alpha=2)
    kwargs=flow_kwargs(memory=False)
    memory=torch.randn(3,8,dtype=torch.bfloat16)
    decoder=model.language_model.model
    def gradients(checkpoint):
        decoder.gradient_checkpointing=checkpoint
        model.zero_grad(set_to_none=True)
        value=model._forward_flow(**kwargs,opd_memory_hidden=(memory,),
            opd_reader_start=1,opd_reader_end=2)
        value.float().square().mean().backward()
        return {name:p.grad.detach().clone() for name,p in model.named_parameters()
                if p.requires_grad}
    without=gradients(False)
    with_checkpoint=gradients(True)
    assert without.keys()==with_checkpoint.keys()
    assert all(torch.allclose(without[name],with_checkpoint[name],atol=1e-5,rtol=1e-4)
               for name in without)


def test_reader_receives_corresponding_frozen_read_layer_entry_and_native_norm(monkeypatch):
    model = tiny_bagel()
    install_memory_readers(model, start=1, end=3, rank=2, alpha=2)
    flow = flow_kwargs(memory=False)
    read = flow_kwargs(memory=True)
    read.pop("loop_memory")
    read["memory_loop_end"] = 3
    condition = dict(flow_kwargs=flow, read_kwargs=read,
        prompt_hidden={1: torch.randn(1, 4, 8, dtype=torch.bfloat16)},
        prompt_mask=torch.ones(1, 4, dtype=torch.bool),
        content_mask=torch.tensor([[False, True, True, False]]), num_slots=2)
    initial = initialize_memory_from_prompt_hidden(
        hidden_cache=condition["prompt_hidden"], prompt_mask=condition["prompt_mask"],
        content_mask=condition["content_mask"], layer_index=1, num_slots=2)
    with torch.no_grad():
        frozen = model.forward_memory_read(**{**read, "x_t": flow["x_t"],
            "timestep": flow["timestep"]}, embed_memory=torch.zeros_like(initial),
            memory_body_in=initial, adapter_mode="off", capture_layer_memory=True)
    assert frozen.layer_memory_read is not None
    seen = {}
    for layer_idx in (1, 2):
        layer = model.language_model.model.layers[layer_idx]
        reader_forward = layer.memory_reader.forward
        def observe(gen, memory, *, index=layer_idx, original=reader_forward):
            seen[index] = memory.detach().clone()
            return original(gen, memory)
        monkeypatch.setattr(layer.memory_reader, "forward", observe)
        projected = []
        layer.self_attn.k_proj.register_forward_pre_hook(
            lambda module, args, *, output=projected: output.append(args[0].detach().clone()))
        seen[(layer_idx, "projected")] = projected
    _, memories = model.forward_memory_opd_velocity(
        x_t=flow["x_t"], timestep=flow["timestep"], condition=condition,
        memory_body_start=1, memory_body_end=3, return_memory=True)
    assert len(memories) == 2
    assert not torch.equal(memories[0], memories[1])
    for offset, layer_idx in enumerate((1, 2)):
        layer = model.language_model.model.layers[layer_idx]
        assert torch.equal(memories[offset], frozen.layer_memory_read[offset])
        assert torch.equal(seen[layer_idx], memories[offset])
        assert torch.equal(seen[(layer_idx, "projected")][-1],
                           layer.input_layernorm(memories[offset]))
        assert not memories[offset].requires_grad
    with pytest.raises(ValueError, match="one .* entry"):
        model._forward_flow(**flow, opd_memory_hidden=(memories[-1],),
                            opd_reader_start=1, opd_reader_end=3)
