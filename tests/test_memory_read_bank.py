import pytest
import torch

from qwen_latent_cot.bagel.memory_read_bank import LayerMemoryState, MemoryReadBank
from qwen_latent_cot.bagel.modeling.bagel import qwen2_navit as navit
from test_memory_grounding import tiny_bagel


def one_sample_flow(*, memory=False, cache=None):
    """B=1, two GEN rows, nonzero RoPE, and an actual prompt cache."""
    width = 6 if memory else 4
    gen = torch.tensor([3, 4] if memory else [1, 2])
    if cache is None:
        cache = navit.NaiveCache(3)
        generator = torch.Generator().manual_seed(789)
        for layer in range(3):
            cache.key_cache[layer] = torch.randn(3, 1, 4, generator=generator).bfloat16()
            cache.value_cache[layer] = torch.randn(3, 1, 4, generator=generator).bfloat16()
    kwargs = dict(x_t=torch.randn(2,4).bfloat16(),
        timestep=torch.full((2,), .5, dtype=torch.bfloat16),
        packed_vae_token_indexes=gen, packed_vae_position_ids=torch.tensor([0,1]),
        packed_text_ids=torch.ones(2,dtype=torch.long),
        packed_text_indexes=torch.tensor([0,width-1]),
        packed_indexes=torch.arange(3,3+width), packed_position_ids=torch.arange(3,3+width),
        packed_seqlens=torch.tensor([width],dtype=torch.int),
        key_values_lens=torch.tensor([3],dtype=torch.int), past_key_values=cache,
        packed_key_value_indexes=torch.arange(3))
    if memory:
        kwargs.update(packed_loop_token_indexes=torch.tensor([1,2]),
            memory_loop_start=1,memory_loop_end=3)
    return kwargs


def reader_condition(flow):
    read = one_sample_flow(memory=True, cache=flow["past_key_values"])
    return dict(flow_kwargs=flow, read_kwargs=read,
        prompt_hidden={1:torch.randn(1,4,8).bfloat16()},
        prompt_mask=torch.ones(1,4,dtype=torch.bool),
        content_mask=torch.tensor([[False,True,True,False]]),num_slots=2)


def random_bank(start=1, end=3):
    return MemoryReadBank({layer:LayerMemoryState(layer,torch.randn(2,8).bfloat16(),
        torch.randn(2,1,4).bfloat16(),torch.randn(2,1,4).bfloat16())
        for layer in range(start,end)})


def test_strict_read_bank_exactly_matches_attention_kv_and_layer_entries(monkeypatch):
    model = tiny_bagel()
    flow = one_sample_flow()
    condition = reader_condition(flow)
    used, entries = [], {}
    original_attention = navit._sdpa_varlen_inference
    def capture_attention(**kwargs):
        used.append((kwargs["key"].detach().clone(),kwargs["value"].detach().clone()))
        return original_attention(**kwargs)
    monkeypatch.setattr(navit,"_sdpa_varlen_inference",capture_attention)
    for layer_idx in (1,2):
        layer = model.language_model.model.layers[layer_idx]
        original = layer.forward_inference
        def capture_entry(*args,_index=layer_idx,_original=original,**kwargs):
            entries[_index] = kwargs["packed_query_sequence"][kwargs["packed_memory_token_indexes"]].detach().clone()
            return _original(*args,**kwargs)
        monkeypatch.setattr(layer,"forward_inference",capture_entry)
    cache_snapshot = {layer:(key.clone(),flow["past_key_values"].value_cache[layer].clone())
        for layer,key in flow["past_key_values"].key_cache.items()}
    bank = model.forward_memory_read_bank(x_t=flow["x_t"],timestep=.5,
        condition=condition,memory_body_start=1,memory_body_end=3)
    assert set(bank.states)=={1,2} and len(used)==3  # prefix + body, STOP
    for layer,state in bank.states.items():
        # Three cached prompt rows precede the query; memory query rows are 1,2.
        assert torch.equal(state.key,used[layer][0][[4,5]])
        assert torch.equal(state.value,used[layer][1][[4,5]])
        assert torch.equal(state.hidden_entry,entries[layer])
        assert not any(value.requires_grad for value in (state.key,state.value,state.hidden_entry))
        assert torch.equal(flow["past_key_values"].key_cache[layer],cache_snapshot[layer][0])
        assert torch.equal(flow["past_key_values"].value_cache[layer],cache_snapshot[layer][1])
    assert torch.equal(bank.states[1].hidden_entry,condition["prompt_hidden"][1][0,1:3])
    assert not torch.equal(bank.states[1].key,bank.states[2].key)


def test_bank_rejects_misalignment_gradients_and_missing_layers():
    with pytest.raises(ValueError,match="detached"):
        LayerMemoryState(1,torch.randn(2,8,requires_grad=True),torch.randn(2,1,4),torch.randn(2,1,4))
    with pytest.raises(ValueError,match="indexed"):
        MemoryReadBank({2:random_bank().states[1]})
    with pytest.raises(ValueError,match="exactly"):
        random_bank().require_layers(1,2)
    source=random_bank()
    zero=source.zero_like()
    assert all(torch.count_nonzero(state.key)==0 and torch.count_nonzero(state.value)==0
               for state in zero.states.values())
    assert all(torch.count_nonzero(state.key)>0 for state in source.states.values())
