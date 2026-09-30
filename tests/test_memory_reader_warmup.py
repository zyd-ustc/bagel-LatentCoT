import pytest
import torch

from qwen_latent_cot.bagel.memory_reader import bank_attention, install_memory_readers
from qwen_latent_cot.bagel.reader_warmup import memory_reader_reconstruction_loss
from qwen_latent_cot.bagel.modeling.bagel import qwen2_navit as navit
from test_memory_grounding import tiny_bagel
from test_memory_read_bank import one_sample_flow, random_bank


def test_warmup_native_parity_after_adapter_changes_and_gradient_isolation():
    model=tiny_bagel()
    names=install_memory_readers(model,start=1,end=3,rank=2,alpha=2,stage="warmup")
    for layer in model.language_model.model.layers[1:3]:
        with torch.no_grad():
            layer.memory_reader.output_adapter.B.weight.normal_()
    flow=one_sample_flow()
    bank=random_bank()
    with torch.no_grad():
        native=model._forward_flow(**flow)
    output=model.forward_memory_reader_warmup(flow_kwargs=flow,memory_read_bank=bank,
        memory_body_start=1,memory_body_end=3)
    assert torch.equal(native,output.velocity)
    assert set(output.layer_losses)=={1,2}
    assert torch.equal(output.loss,torch.stack(list(output.layer_losses.values())).mean())
    assert all(not target.requires_grad for target in output.prompt_targets.values())
    output.loss.backward()
    grads={name:p.grad for name,p in model.named_parameters()}
    assert names and all("memory_reader.output_adapter." in name for name in names)
    assert all(grads[name] is not None and grads[name].abs().sum()>0 for name in names)
    assert all(grads[name] is None for name in grads if name not in names)
    assert all(state.key.grad is None and state.value.grad is None for state in bank.states.values())


def test_warmup_loss_does_not_backpropagate_to_rollout_or_prompt_cache():
    model=tiny_bagel()
    install_memory_readers(model,start=1,end=3,rank=2,alpha=2)
    flow=one_sample_flow()
    flow["x_t"].requires_grad_()
    for value in flow["past_key_values"].key_cache.values():
        value.requires_grad_()
    output=model.forward_memory_reader_warmup(flow_kwargs=flow,memory_read_bank=random_bank(),
        memory_body_start=1,memory_body_end=3)
    output.loss.backward()
    assert flow["x_t"].grad is None
    assert all(value.grad is None for value in flow["past_key_values"].key_cache.values())


def test_readout_query_is_exact_native_post_norm_rope_query_for_both_banks(monkeypatch):
    model=tiny_bagel()
    install_memory_readers(model,start=1,end=3,rank=2,alpha=2)
    flow=one_sample_flow()
    bank=random_bank()
    actual_queries=[]
    original=navit._sdpa_varlen_inference
    def capture(**kwargs):
        actual_queries.append(kwargs["query"].detach().clone())
        return original(**kwargs)
    monkeypatch.setattr(navit,"_sdpa_varlen_inference",capture)
    queries={}
    for layer in (1,2):
        reader=model.language_model.model.layers[layer].memory_reader
        original_forward=reader.forward
        original_target=reader.prompt_target
        def read(*,_layer=layer,_original=original_forward,**kwargs):
            queries[_layer,"memory"]=kwargs["gen_query"].detach().clone()
            return _original(**kwargs)
        def target(*,_layer=layer,_original=original_target,**kwargs):
            queries[_layer,"prompt"]=kwargs["gen_query"].detach().clone()
            return _original(**kwargs)
        monkeypatch.setattr(reader,"forward",read)
        monkeypatch.setattr(reader,"prompt_target",target)
    model.forward_memory_reader_warmup(flow_kwargs=flow,memory_read_bank=bank,
        memory_body_start=1,memory_body_end=3)
    for layer in (1,2):
        expected=actual_queries[layer][flow["packed_vae_token_indexes"]]
        assert torch.equal(queries[layer,"memory"],expected)
        assert torch.equal(queries[layer,"prompt"],expected)


def test_bank_attention_native_gqa_repeat_matches_manual_softmax():
    q=torch.randn(3,4,2)
    k=torch.randn(5,2,2)
    v=torch.randn(5,2,2)
    output,weights=bank_attention(q,k,v,num_heads=4,num_kv_heads=2,head_dim=2)
    expected=[]
    for head in range(4):
        probability=(q[:,head]@k[:,head//2].T/(2**.5)).softmax(-1)
        expected.append(probability@v[:,head//2])
    assert torch.allclose(output,torch.stack(expected,dim=1).flatten(1),atol=1e-6)
    assert torch.allclose(weights.sum(-1),torch.ones(4,3))


def test_target_detach_and_readout_frozen_o_plus_zero_initialized_adapter():
    model=tiny_bagel()
    install_memory_readers(model,start=1,end=3,rank=2,alpha=2)
    reader=model.language_model.model.layers[1].memory_reader
    query=torch.randn(2,2,4).bfloat16()
    state=random_bank().states[1]
    readout=reader(gen_query=query,memory_key=state.key,memory_value=state.value)
    attended,_=bank_attention(query,state.key,state.value,num_heads=2,num_kv_heads=1,head_dim=4)
    expected=reader.native_gen_o_proj(attended.bfloat16())
    assert torch.equal(readout,expected) and torch.count_nonzero(readout)>0
    target=torch.randn_like(readout,requires_grad=True)
    memory_reader_reconstruction_loss(readout,target).backward()
    assert target.grad is None
    assert reader.output_adapter.B.weight.grad.abs().sum()>0
    assert reader.output_adapter.A.weight.grad.abs().sum()==0


def test_warmup_rejects_combined_injection_and_batch_two():
    model=tiny_bagel()
    install_memory_readers(model,start=1,end=3,rank=2,alpha=2)
    flow=one_sample_flow()
    bank=random_bank()
    with pytest.raises(ValueError,match="exclusive"):
        model._forward_flow(**flow,opd_memory_hidden=bank,reader_warmup_bank=bank,
            reader_warmup_sink={},opd_reader_start=1,opd_reader_end=3)
    flow["packed_seqlens"]=torch.tensor([2,2],dtype=torch.int)
    with pytest.raises(ValueError,match="native K=0"):
        model.forward_memory_reader_warmup(flow_kwargs=flow,memory_read_bank=bank,
            memory_body_start=1,memory_body_end=3)
