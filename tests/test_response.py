"""Shared native GEN queries, common-softmax attention, and C-only gradients."""
from types import SimpleNamespace
import torch
from test_memory_topology import lm, setup_writer, source_for, native_call
from qwen_latent_cot.diagnostics.response import (
    capture_queries, attention_response, normalized_response_loss, response_loss)
from qwen_latent_cot.diagnostics.record import FrozenGuard


def test_common_kv_competes_in_the_same_softmax():
    q=torch.tensor([[[1.]]])
    common_k=torch.tensor([[[0.]]]); common_v=torch.tensor([[[2.]]])
    condition_k=torch.tensor([[[0.]]],requires_grad=True)
    condition_v=torch.tensor([[[6.]]],requires_grad=True)
    response=attention_response(q,common_k,common_v,condition_k,condition_v)
    torch.testing.assert_close(response,torch.tensor([[[4.]]]))
    response.sum().backward()
    assert condition_k.grad.abs().max()>0 and condition_v.grad.abs().max()>0
    # Condition-only attention would return 6, losing competition with common KV.


def test_teacher_is_detached_and_token_counts_need_not_match():
    torch.manual_seed(6)
    q=torch.randn(5,4,8); common_k=torch.randn(7,2,8); common_v=torch.randn_like(common_k)
    mk=torch.randn(16,2,8,requires_grad=True); mv=torch.randn_like(mk,requires_grad=True)
    teacher=torch.randn(5,4,8,requires_grad=True)
    student=attention_response(q,common_k,common_v,mk,mv)
    normalized_response_loss(student,teacher).backward()
    assert teacher.grad is None
    assert mk.grad.abs().max()>0 and mv.grad.abs().max()>0


def make_captured_probe(lm):
    source,cache=source_for(lm)
    runtime=SimpleNamespace(decoder=lm.model)
    flow={'past_key_values':cache}
    original=lm.model.forward_inference
    hidden=lm.model.embed_tokens(torch.arange(8))
    with capture_queries(runtime,flow,[0,2],64,3) as result:
        native_call(lm,hidden,torch.arange(5,13),(8,),cache,(5,),mode='gen')
    assert lm.model.forward_inference==original
    assert set(result)=={0,2}
    assert all(layer['q'].shape==(3,4,8) and layer['k'].shape==(13,2,8) for layer in result.values())
    assert all(not layer['q'].requires_grad and not layer['k'].requires_grad for layer in result.values())
    return dict(layers=result,prompt_positions=[5]),source




def test_expanded_query_pools_use_all_128_images_once():
    from qwen_latent_cot.diagnostics.response import training_images
    rows = list(range(128))
    assigned = [training_images(rows, rank, 16, 8) for rank in range(16)]
    assert all(len(pool) == 8 for pool in assigned)
    assert sorted(item for pool in assigned for item in pool) == rows
    assert training_images(list(range(12)), 15, 16) == [3]


def test_natural_caption_validation_keeps_missing_fact_scores(monkeypatch, tmp_path):
    from qwen_latent_cot.diagnostics import response
    runtime = SimpleNamespace(writer=SimpleNamespace(eval=lambda: None), text=lambda text: (text,),
        memory=lambda text: SimpleNamespace(slots=16), caption_prefix=lambda bank: None,
        config={'response_caption_weight': .1})
    monkeypatch.setattr(response, 'teacher_condition', lambda runtime, text: (None, []))
    monkeypatch.setattr(response, 'caption_result', lambda runtime, row, *args:
                        dict(nll=2., fact_accuracy=.5 if row['facts'] else None))
    monkeypatch.setattr(response, 'response_loss', lambda *args: (torch.tensor(.2), []))
    rows = [dict(sample_id='old', caption='old', facts=[{}]),
            dict(sample_id='new', caption='new', facts=[], dataset_group='natural_text')]
    result = response.validate(runtime, rows, [[dict(sample_id='q', t=.8, noise_seed=1)]], tmp_path)
    assert result['literal_fact_accuracy'] == .5
    assert result['literal_fact_scored_samples'] == 1
    assert result['groups']['natural_text']['samples'] == 1
