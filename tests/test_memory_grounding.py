from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from qwen_latent_cot.bagel.memory_grounding import (
    adapters_off, causal_reward_advantages, memory_dependency_loss,
    shuffle_across_batch, write_prompt_mask_probability,
)
from qwen_latent_cot.bagel.memory_training import validate_config, GroundingRuntime, ReplayItem
from qwen_latent_cot.bagel.loop import inject_loop_lora, configure_loop_trainable_routes
from qwen_latent_cot.bagel.modeling.bagel.bagel import Bagel
from qwen_latent_cot.bagel.modeling.bagel import qwen2_navit as navit
from test_memory_mechanism import tiny_model, real_kwargs


class TinyTime(nn.Module):
    def forward(self, t):
        return t[:, None].expand(-1, 8)


def tiny_bagel():
    model = Bagel.__new__(Bagel)
    nn.Module.__init__(model)
    decoder = tiny_model()
    model.language_model = navit.Qwen2ForCausalLM(decoder.config).to(torch.bfloat16)
    model.language_model.model = decoder
    model.hidden_size = 8
    model.use_moe = True
    model.config = SimpleNamespace(round0_memory_write_enabled=False)
    model.vae2llm = nn.Linear(4, 8).to(torch.bfloat16)
    model.llm2vae = nn.Linear(8, 4).to(torch.bfloat16)
    model.time_embedder = TinyTime()
    model.latent_pos_embed = nn.Embedding(4, 8).to(torch.bfloat16)
    model.loop_memory = nn.Parameter(torch.randn(1, 8, dtype=torch.bfloat16), requires_grad=False)
    inject_loop_lora(model, start_layer=1, end_layer=2, rank=2, alpha=2)
    for name, module in model.named_modules():
        if getattr(module, "is_loop_lora", False) and name.endswith(".q_proj"):
            module.write_enabled = False
    return model.eval()


def flow_kwargs(memory=True):
    source = real_kwargs(memory=memory)
    memory_idx = source["packed_memory_token_indexes"]
    text_idx = source["packed_text_indexes"]
    text_idx = text_idx[~torch.isin(text_idx, memory_idx)]
    return dict(x_t=torch.randn(2, 4, dtype=torch.bfloat16),
        timestep=torch.full((2,), .5, dtype=torch.bfloat16),
        packed_vae_token_indexes=source["packed_vae_token_indexes"],
        packed_vae_position_ids=torch.zeros(2, dtype=torch.long),
        packed_text_ids=torch.ones(len(text_idx), dtype=torch.long), packed_text_indexes=text_idx,
        packed_indexes=source["packed_query_indexes"], packed_position_ids=source["packed_query_position_ids"],
        packed_seqlens=source["query_lens"], key_values_lens=source["key_values_lens"],
        past_key_values=source["past_key_values"], packed_key_value_indexes=source["packed_key_value_indexes"],
        **(dict(packed_loop_token_indexes=memory_idx, loop_memory=torch.zeros(2,8,dtype=torch.bfloat16),
                memory_loop_start=1, memory_loop_end=2) if memory else {}))


def test_derangement_is_sample_level_deterministic_and_immutable():
    m = torch.arange(5*8*4.).reshape(5,8,4)
    snapshot = m.clone()
    shuffled, donors = shuffle_across_batch(m, generator=torch.Generator().manual_seed(42))
    assert torch.equal(m, snapshot)
    assert all(i != j for i,j in enumerate(donors))
    assert torch.equal(shuffled, m[donors])
    again, _ = shuffle_across_batch(m, generator=torch.Generator().manual_seed(42))
    assert torch.equal(shuffled, again)
    with pytest.raises(ValueError):
        shuffle_across_batch(m[:1])


def test_dependency_loss_per_sample_normalization_and_teacher_stopgrad():
    correct = torch.tensor([[[1., 1.]], [[10.,10.]]], requires_grad=True)
    teacher = correct.detach().clone().requires_grad_()
    shuffled = (correct.detach() + torch.tensor([[[1.]], [[10.]]])).requires_grad_()
    zero = torch.zeros_like(correct, requires_grad=True)
    out = memory_dependency_loss(correct_velocity=correct, shuffled_velocity=shuffled,
                                 zero_velocity=zero, teacher_velocity=teacher)
    assert out.error_correct == 0
    assert out.error_shuffled == 1
    assert out.dependency_gap_zero == 1
    out.loss.backward()
    assert correct.grad is not None and teacher.grad is None
    out = memory_dependency_loss(correct_velocity=zero, shuffled_velocity=zero,
                                 zero_velocity=zero, teacher_velocity=zero)
    assert torch.isfinite(out.loss)


def test_curriculum_has_bounded_linear_decay():
    assert write_prompt_mask_probability(0) == .5
    assert write_prompt_mask_probability(1500) == pytest.approx(.3)
    assert write_prompt_mask_probability(3000) == pytest.approx(.1)
    assert write_prompt_mask_probability(9000) == pytest.approx(.1)


def test_write_only_mask_and_explicit_override_leave_cache_and_read_intact(monkeypatch):
    model = tiny_model()
    kwargs = real_kwargs()
    original = kwargs["packed_query_sequence"].clone()
    keys = {i: v.clone() for i,v in kwargs["past_key_values"].key_cache.items()}
    values = {i: v.clone() for i,v in kwargs["past_key_values"].value_cache.items()}
    calls = []
    for i, layer in enumerate(model.layers):
        real = layer.forward_inference
        def capture(*args, _i=i, _real=real, **kw):
            calls.append((_i, kw["packed_query_sequence"].clone(), kw))
            return _real(*args, **kw)
        monkeypatch.setattr(layer, "forward_inference", capture)
    memory = torch.zeros(2,8, dtype=torch.bfloat16)
    before = memory.clone()
    out = model.forward_inference(**kwargs, memory_loop_start=1, memory_loop_end=2,
        memory_loop_repeat=3, write_memory_override=memory, mask_prompt_kv_during_write=True,
        collect_write_round_outputs=True)
    # prefix, Read, Write1, diagnostic suffix1, Write2, final suffix2.
    assert [i for i,_,_ in calls] == [0,1,1,2,1,2]
    assert [bool(k.get("mask_prompt_kv_for_nonmemory")) for _,_,k in calls] == [False,False,True,False,True,False]
    assert torch.equal(calls[2][1][[1,5]], memory)
    assert not torch.equal(calls[4][1][[1,5]], memory)  # only first Write is overridden
    nonmemory = torch.tensor([0,2,3,4,6,7])
    assert torch.equal(calls[1][1][nonmemory], calls[2][1][nonmemory])
    assert torch.equal(calls[1][1][nonmemory], calls[4][1][nonmemory])
    assert torch.equal(memory, before) and torch.equal(kwargs["packed_query_sequence"], original)
    assert all(torch.equal(v, keys[i]) for i,v in kwargs["past_key_values"].key_cache.items())
    assert all(torch.equal(v, values[i]) for i,v in kwargs["past_key_values"].value_cache.items())
    assert len(out.write_round_suffix_hiddens) == 2


@pytest.mark.parametrize("stage,route", [("reader","q_proj_moe_gen"), ("writer","q_proj")])
def test_real_bagel_gradient_routing(stage, route):
    model = tiny_bagel()
    configure_loop_trainable_routes(model, [route])
    kwargs = flow_kwargs()
    read_kwargs = {k:v for k,v in kwargs.items() if k != "loop_memory"}
    if stage == "reader":
        with torch.no_grad():
            m = model.forward_memory_read(**read_kwargs).memory_read
        assert not m.requires_grad
    else:
        m = model.forward_memory_read(**read_kwargs).memory_read
        m.retain_grad()
    out = model.forward_loop_supervised(**kwargs, num_write_rounds=1, write_memory_override=m,
                                        mask_prompt_kv_during_write=True)
    out.final_velocity.float().square().mean().backward()
    grads = [p.grad for n,p in model.named_parameters() if f".{route}.lora_" in n]
    assert grads and all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    assert model.loop_memory.grad is None
    assert all(p.grad is None for n,p in model.named_parameters() if f".{route}.lora_" not in n)
    if stage == "writer":
        assert m.grad is not None and m.grad.abs().sum() > 0


def test_native_adapter_off_matches_original_flow_exactly():
    model = tiny_bagel()
    kwargs = flow_kwargs(memory=False)
    for module in model.modules():
        if getattr(module, "is_loop_lora", False):
            module.lora_B.weight.data.fill_(1.)
    with torch.no_grad():
        expected = model._forward_flow(**kwargs)
        for module in model.modules():
            if getattr(module, "is_loop_lora", False):
                module.set_loop_mode("write")
        with adapters_off(model):
            actual = model._forward_flow(**kwargs)
        assert torch.equal(expected, actual)
    assert all(m.loop_mode == "write" for m in model.modules() if getattr(m,"is_loop_lora",False))


def test_attention_probe_observes_write_prompt_mask_without_masking_memory():
    model = tiny_model()
    trace = []
    with torch.no_grad():
        model.forward_inference(**real_kwargs(), memory_loop_start=1, memory_loop_end=2,
            memory_loop_repeat=2, mask_prompt_kv_during_write=True,
            attention_mass_sink=trace, attention_mass_layers=(0,1,2))
    read = [r for r in trace if r["stage"] == "read"]
    write = [r for r in trace if r["stage"] == "write"]
    assert read and write
    assert all(r["gen_to_memory"] == 0 and r["gen_to_prompt"] > 0 for r in read)
    assert all(r["gen_to_prompt"] == 0 and r["memory_to_prompt"] > 0 for r in write)


@pytest.mark.parametrize("overrides", [dict(loop_memory_persist=True), dict(trainable_routes=["k_proj"]),
    dict(batch_size=1), dict(num_write_rounds=3), dict(loop_update_mode="sma"), dict(cfg_text_scale=4)])
def test_config_fails_closed(overrides):
    with pytest.raises(ValueError):
        validate_config(dict(model_path="m",data_path="d",output_dir="o",**overrides), "reader")


def test_stage_gates_and_causal_reward():
    with pytest.raises(ValueError, match="preceding held-out"):
        validate_config(dict(model_path="m",data_path="d",output_dir="o",adapter_path="a"), "writer")
    c = torch.tensor([1.,2.,3.])
    s = torch.tensor([.5,2.,4.])
    objective, advantage, terms = causal_reward_advantages(c,s,c,c)
    assert torch.equal(objective, c+(c-s))
    assert advantage.mean().abs() < 1e-6


def test_dependency_runtime_replays_identical_states_and_mask_across_arms():
    class Fake:
        config = {}
        read_memory = staticmethod(lambda item, detach: torch.ones(8,4)*int(item.record["id"]))
        calls = []
        def write(self, item, memory, **kwargs):
            self.calls.append((item, memory.clone(), kwargs))
            return SimpleNamespace(final_velocity=item.sample+memory.mean())
    fake = Fake()
    items = [ReplayItem({"id":str(i)},None,None,None,torch.ones(2,4)*i,.3,torch.ones(2,4)) for i in (1,2)]
    loss, metrics = GroundingRuntime.dependency(fake,items,masks=[True,False],detach_read=True)
    assert torch.isfinite(loss)
    for i in range(2):
        group = fake.calls[i*3:(i+1)*3]
        assert all(call[0] is items[i] and call[2]["mask"] == (i==0) for call in group)
        assert torch.equal(group[0][1], torch.ones(8,4)*(i+1))
        assert torch.equal(group[1][1], torch.ones(8,4)*(2-i))
        assert torch.count_nonzero(group[2][1]) == 0
