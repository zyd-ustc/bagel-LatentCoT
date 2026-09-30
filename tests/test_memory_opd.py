import pytest
import torch

from qwen_latent_cot.bagel.memory_init import initialize_memory_from_prompt_hidden
from qwen_latent_cot.bagel.memory_reader import install_memory_readers
from qwen_latent_cot.bagel.opd import opd_velocity_loss
from test_memory_grounding import tiny_bagel
from test_memory_read_bank import one_sample_flow, random_bank, reader_condition


def test_prompt_content_uniform_initializer_is_detached_and_deterministic():
    hidden = torch.arange(1*7*8,dtype=torch.float32).reshape(1,7,8).requires_grad_()
    mask = torch.ones(1,7,dtype=torch.bool)
    content = torch.tensor([[False,True,True,False,True,True,False]])
    result = initialize_memory_from_prompt_hidden(hidden_cache={12:hidden},
        prompt_mask=mask,content_mask=content,layer_index=12,num_slots=3)
    assert result.shape==(3,8) and not result.requires_grad
    assert torch.equal(result,hidden.detach()[0,[1,4,5]])
    assert torch.equal(result,initialize_memory_from_prompt_hidden(hidden_cache={12:hidden},
        prompt_mask=mask,content_mask=content,layer_index=12,num_slots=3))


def test_short_prompt_repeats_without_random_scale_and_empty_content_rejected():
    hidden=torch.randn(1,3,8)
    content=torch.tensor([[False,True,False]])
    memory=initialize_memory_from_prompt_hidden(hidden_cache=hidden,
        prompt_mask=torch.ones_like(content),content_mask=content,layer_index=12,num_slots=8)
    assert (memory-hidden[0,1]).abs().max()<=1.1e-5
    with pytest.raises(ValueError,match="no eligible"):
        initialize_memory_from_prompt_hidden(hidden_cache=hidden,
            prompt_mask=torch.zeros_like(content),layer_index=12,num_slots=8)


def test_warmed_reader_zero_gate_native_parity_and_gate_only_gradient():
    model=tiny_bagel()
    names=install_memory_readers(model,start=1,end=3,rank=2,alpha=2,stage="opd")
    assert len(names)==2 and all(name.endswith("memory_reader.injection_gate") for name in names)
    for layer in model.language_model.model.layers[1:3]:
        with torch.no_grad():
            layer.memory_reader.output_adapter.B.weight.normal_()  # simulate warmed adapter
    kwargs=one_sample_flow()
    bank=random_bank()
    with torch.no_grad():
        native=model._forward_flow(**kwargs)
        student=model._forward_flow(**kwargs,opd_memory_hidden=bank,
            opd_reader_start=1,opd_reader_end=3)
    assert torch.equal(native,student)
    output=model._forward_flow(**kwargs,opd_memory_hidden=bank,
        opd_reader_start=1,opd_reader_end=3)
    opd_velocity_loss(output,native.float()+1).loss.backward()
    grads={name:p.grad for name,p in model.named_parameters()}
    assert all(grads[name] is not None for name in names)
    assert any(grads[name].abs().sum()>0 for name in names)
    assert all(grads[name] is None for name in grads if name not in names)


def test_opd_loss_does_not_train_teacher():
    student=torch.tensor([1.,2.],requires_grad=True)
    teacher=torch.tensor([3.,4.],requires_grad=True)
    result=opd_velocity_loss(student,teacher)
    assert result.loss==4
    result.loss.backward()
    assert student.grad is not None and teacher.grad is None


def test_bagel_opd_api_uses_frozen_bank_and_accepts_scalar_timestep():
    model=tiny_bagel()
    install_memory_readers(model,start=1,end=3,rank=2,alpha=2,stage="opd")
    flow=one_sample_flow()
    condition=reader_condition(flow)
    with torch.no_grad():
        native=model._forward_flow(**flow)
    student,bank=model.forward_memory_opd_velocity(x_t=flow["x_t"],timestep=.5,
        condition=condition,memory_body_start=1,memory_body_end=3,return_memory=True)
    assert torch.equal(native,student)
    assert set(bank.states)=={1,2}
    assert all(state.key.shape==(2,1,4) for state in bank.states.values())
    assert model.last_opd_memory_stats["Mread_rms"]>0
    with pytest.raises(ValueError,match="initializer"):
        model.forward_memory_opd_velocity(x_t=flow["x_t"],timestep=.5,condition=condition,
            memory_init_strategy="random",memory_body_start=1,memory_body_end=3)


def test_eval_mode_checkpoint_preserves_all_gate_gradients_after_gates_open():
    model=tiny_bagel()
    install_memory_readers(model,start=1,end=3,rank=2,alpha=2,stage="opd")
    kwargs=one_sample_flow()
    bank=random_bank()
    decoder=model.language_model.model
    with torch.no_grad():
        decoder.layers[1].memory_reader.injection_gate.fill_(.3)
        decoder.layers[2].memory_reader.injection_gate.fill_(.2)
    def gradients(checkpoint):
        decoder.gradient_checkpointing=checkpoint
        model.zero_grad(set_to_none=True)
        value=model._forward_flow(**kwargs,opd_memory_hidden=bank,
            opd_reader_start=1,opd_reader_end=3)
        value.float().square().mean().backward()
        return {name:p.grad.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
    without=gradients(False)
    with_checkpoint=gradients(True)
    assert without.keys()==with_checkpoint.keys()
    assert all(torch.allclose(without[name],with_checkpoint[name],atol=1e-5,rtol=1e-4)
               for name in without)


def test_reader_receives_exact_corresponding_precomputed_native_kv(monkeypatch):
    model=tiny_bagel()
    install_memory_readers(model,start=1,end=3,rank=2,alpha=2,stage="opd")
    flow=one_sample_flow()
    bank=model.forward_memory_read_bank(x_t=flow["x_t"],timestep=.5,
        condition=reader_condition(flow),memory_body_start=1,memory_body_end=3)
    seen={}
    for index in (1,2):
        reader=model.language_model.model.layers[index].memory_reader
        original=reader.forward
        def observe(*,_index=index,_original=original,**kwargs):
            seen[_index]=(kwargs["memory_key"],kwargs["memory_value"])
            return _original(**kwargs)
        monkeypatch.setattr(reader,"forward",observe)
    model._forward_flow(**flow,opd_memory_hidden=bank,opd_reader_start=1,opd_reader_end=3)
    for index in (1,2):
        assert seen[index][0] is bank.states[index].key
        assert seen[index][1] is bank.states[index].value
    with pytest.raises(ValueError,match="exactly"):
        model._forward_flow(**flow,opd_memory_hidden=bank,opd_reader_start=1,opd_reader_end=2)
    with pytest.raises(ValueError,match="MemoryReadBank"):
        model._forward_flow(**flow,opd_memory_hidden=(bank.states[1].hidden_entry,),
            opd_reader_start=1,opd_reader_end=3)
