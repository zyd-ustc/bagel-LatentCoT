"""Backend contracts: cached causal alignment, packed isolation and MoT parity."""
from copy import deepcopy
import importlib.util
import pytest
import torch
from helpers import fixture
from qwen_latent_cot.bagel.attention import flash_attn_varlen_func
from qwen_latent_cot.bagel.accelerator import set_device,autocast_for,seeded_context


def npu_device():
    if importlib.util.find_spec('torch_npu') is None:pytest.skip('Ascend runtime unavailable')
    import torch_npu
    if not torch.npu.is_available():pytest.skip('Ascend hardware unavailable')
    return set_device('npu:0')


@pytest.mark.parametrize('device_type',['cpu','npu'])
def test_cached_single_query_reads_all_previous_keys(device_type):
    device=torch.device('cpu') if device_type=='cpu' else npu_device()
    q=torch.zeros(1,4,16,dtype=torch.bfloat16,device=device)
    k=torch.zeros(5,2,16,dtype=torch.bfloat16,device=device)
    v=torch.arange(5,device=device).view(5,1,1).expand(5,2,16).to(torch.bfloat16)
    out=flash_attn_varlen_func(q,k,v,torch.tensor([0,1],device=device,dtype=torch.int32),
        torch.tensor([0,5],device=device,dtype=torch.int32),1,5,causal=True)
    assert torch.equal(out.cpu(),torch.full((1,4,16),2.,dtype=torch.bfloat16))


@pytest.mark.parametrize('device_type',['cpu','npu'])
def test_visual_posterior_replay_preserves_external_rng(device_type):
    device=torch.device('cpu') if device_type=='cpu' else npu_device()
    cpu_state=torch.get_rng_state().clone()
    backend_state=torch.npu.get_rng_state(device).clone() if device_type=='npu' else None
    with seeded_context(device,72):a=torch.randn(8,device=device);ca=torch.randn(8)
    with seeded_context(device,72):b=torch.randn(8,device=device);cb=torch.randn(8)
    assert torch.equal(a,b) and torch.equal(ca,cb) and torch.equal(cpu_state,torch.get_rng_state())
    if backend_state is not None:assert torch.equal(backend_state,torch.npu.get_rng_state(device))


@pytest.mark.parametrize('causal',[False,True])
def test_npu_variable_length_gqa_matches_cpu_and_preserves_sample_isolation(causal):
    device=npu_device();torch.manual_seed(732)
    q=torch.randn(8,4,16).bfloat16();k=torch.randn(19,2,16).bfloat16();v=torch.randn_like(k)
    cq=torch.tensor([0,1,5,5,8],dtype=torch.int32);ck=torch.tensor([0,7,16,16,19],dtype=torch.int32)
    ref=flash_attn_varlen_func(q,k,v,cq,ck,4,9,causal=causal)
    inputs=[x.to(device) for x in (q,k,v,cq,ck)]
    out=flash_attn_varlen_func(*inputs,4,9,causal=causal)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.cpu().float(),ref.float(),atol=.02,rtol=.02)
    assert torch.equal(out,flash_attn_varlen_func(*inputs,4,9,causal=causal))
    changed=inputs[2].clone();changed[7:]+=10
    edited=flash_attn_varlen_func(inputs[0],inputs[1],changed,inputs[3],inputs[4],4,9,causal=causal)
    assert torch.equal(out[:1],edited[:1]) and not torch.equal(out[1:],edited[1:])


def test_npu_full_mot_decoder_matches_cpu_and_does_not_modify_cache():
    device=npu_device();model,kwargs=fixture(device='cpu',batch=2,depth=4)
    cpu=model.language_model.model.forward_inference(**kwargs).packed_query_sequence
    model.language_model.to(device)
    args={key:value.to(device) if isinstance(value,torch.Tensor) else deepcopy(value) for key,value in kwargs.items()}
    cache=args['past_key_values']
    for field in ('key_cache','value_cache'):
        setattr(cache,field,{i:value.to(device) for i,value in getattr(cache,field).items()})
    frozen=deepcopy(cache)
    with torch.inference_mode(),autocast_for(device):
        out=model.language_model.model.forward_inference(**args).packed_query_sequence
        repeat=model.language_model.model.forward_inference(**args).packed_query_sequence
    assert torch.isfinite(out).all() and torch.equal(out,repeat)
    assert torch.allclose(out.cpu().float(),cpu.float(),atol=.03,rtol=.03)
    assert all(torch.equal(cache.key_cache[i],frozen.key_cache[i]) and
        torch.equal(cache.value_cache[i],frozen.value_cache[i]) for i in cache.key_cache)
