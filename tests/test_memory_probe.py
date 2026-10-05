from types import SimpleNamespace
import json
import pytest
import torch
from PIL import Image
from qwen_latent_cot.bagel.memory_probe import (
    NativeMemoryQA, ProbeCapture, probe_questions, canonical_answer,
    finalize_capture, validate_capture, snapshot_sources,
)
from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import Qwen2Config,Qwen2ForCausalLM
from qwen_latent_cot.bagel.internal_loop import LoopConfig,InternalLoopRuntime
from qwen_latent_cot.evaluation.io import sha256
from qwen_latent_cot.evaluation.probe_results import label_answer,summarize_probe
from test_internal_loop import fixture


class TinyTokenizer:
    def encode(self,text,add_special_tokens=False):
        return [2+ord(c)%29 for c in text]


def tiny_reader(device='cpu'):
    torch.manual_seed(81)
    cfg=Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,
        num_attention_heads=4,num_key_value_heads=2,num_hidden_layers=3,
        layer_module='Qwen2MoTDecoderLayer',pad_token_id=0)
    llm=Qwen2ForCausalLM(cfg).eval().requires_grad_(False).to(device=device,dtype=torch.bfloat16)
    model=torch.nn.Module();model.language_model=llm
    bundle=SimpleNamespace(model=model,tokenizer=TinyTokenizer(),token_ids={'bos_token_id':0,'eos_token_id':1})
    return NativeMemoryQA(bundle)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_native_qa_full_depth_sparse_kv_causal_packing_and_no_cache_write(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA contract')
    reader=tiny_reader(device);kv={0:(torch.randn(2,2,8,dtype=torch.bfloat16),torch.randn(2,2,8,dtype=torch.bfloat16)),
                                2:(torch.randn(3,2,8,dtype=torch.bfloat16),torch.randn(3,2,8,dtype=torch.bfloat16))}
    copies={i:(k.clone(),v.clone()) for i,(k,v) in kv.items()}
    observed=[];originals=[]
    for i,layer in enumerate(reader.decoder.layers):
        original=layer.forward_inference;originals.append(original)
        def wrapper(_original=original,_i=i,**kwargs):
            assert kwargs['mode']=='und' and kwargs['is_causal'] and not kwargs['update_past_key_values']
            expected=len(kv[_i][0]) if _i in kv else 0
            assert kwargs['key_values_lens'].tolist()==[expected]*3
            assert len(kwargs['packed_key_value_indexes'])==expected*3
            observed.append(_i)
            return _original(**kwargs)
        layer.forward_inference=wrapper
    packed=reader.score('Visible count?',['0','12','unknown'],kv,17)
    assert observed==[0,1,2]
    for layer,original in zip(reader.decoder.layers,originals):layer.forward_inference=original
    for candidate in packed['mean_token_log_likelihood']:
        single=reader.score('Visible count?',[candidate],kv,17)
        assert packed['mean_token_log_likelihood'][candidate]==pytest.approx(single['mean_token_log_likelihood'][candidate],abs=1e-5)
    assert sum(packed['choice_probability'].values())==pytest.approx(1)
    for i,(k,v) in kv.items():assert torch.equal(k,copies[i][0]) and torch.equal(v,copies[i][1])
    empty=reader.score('Visible count?',['0','12','unknown'],{},17)
    assert packed['mean_token_log_likelihood']!=empty['mean_token_log_likelihood']


def test_snapshot_noninterference_and_dynamic_vs_seed():
    model,kwargs=fixture(batch=1);cfg=LoopConfig(start_layer=1,end_layer=3)
    runtime=InternalLoopRuntime(model,cfg)
    try:
        expected=model.language_model.model.forward_inference(**kwargs).packed_query_sequence
        runtime.probe_capture=ProbeCapture([0])
        actual=model.language_model.model.forward_inference(**kwargs).packed_query_sequence
        assert torch.equal(actual,expected)
        snapshot=runtime.probe_capture.layers[0][1]
        assert not torch.equal(snapshot['dynamic_k'],snapshot['seed_k'])
        assert snapshot['question_position_start']==9 and snapshot['lengths']==[8]
    finally:runtime.close()


def test_capture_requires_all_steps_layers_and_binds_tensor_image_hashes(tmp_path):
    capture=ProbeCapture([8]);metadata={'start_layer':0,'end_layer':2,'num_timesteps':50}
    with pytest.raises(ValueError,match='steps'):capture.save(tmp_path,metadata)
    state={'dynamic_k':torch.ones(2,2,8),'dynamic_v':torch.ones(2,2,8),
        'seed_k':torch.zeros(2,2,8),'seed_v':torch.zeros(2,2,8),
        'source_indexes':torch.tensor([1,2]),'lengths':[2],'question_position_start':9}
    capture.record(8,0,state);state['dynamic_k'].zero_()
    assert capture.layers[8][0]['dynamic_k'].sum()>0
    capture.images[8]=Image.new('RGB',(28,28));capture.timesteps[8]=.7
    with pytest.raises(ValueError,match='per-layer'):capture.save(tmp_path,metadata)
    capture.record(8,1,state);capture.save(tmp_path,metadata)
    path=tmp_path/'capture.json';finalize_capture(path)
    snapshots=validate_capture(path,sha256(path))['snapshots']
    sources=snapshot_sources(snapshots[0]);assert set(sources)=={'DYNAMIC','SEED','EMPTY'}
    assert set(sources['DYNAMIC'])=={0,1} and sources['EMPTY']=={}
    assert not json.loads(path.read_text())['prompt_kv_exported']
    Image.new('RGB',(28,28),(255,0,0)).save(snapshots[0]['image_path'])
    with pytest.raises(ValueError,match='tensor/image'):validate_capture(path,sha256(path))


def test_questions_and_unknown_labels():
    row={'vqa_list':[['How many cats?','two'],['Any cats?','Yes'],['How many dogs?','one'],['Is cat left of dog?','No']],
         'skills':['count','object','count','position']}
    q=probe_questions(row,3)
    assert [x['question_id'] for x in q]==['0','1','3'] and q[0]['desired_answer']=='2'
    assert q[0]['candidates']==[str(i) for i in range(13)]+['unknown']
    assert canonical_answer(' THREE. ')=='3'
    assert label_answer({'answer':'two','confidence':.79},q[0]['candidates'])['observed_answer']=='unknown'
    assert label_answer({'answer':'two','confidence':.9},q[0]['candidates'])['observed_answer']=='2'
    with pytest.raises(ValueError):label_answer({'answer':'13','confidence':.9},q[0]['candidates'])
    with pytest.raises(ValueError):label_answer({'answer':'2','confidence':float('nan')},q[0]['candidates'])


def test_probe_uses_observed_content_separates_prompt_echo_and_unknown():
    rows=[];labels=[]
    for seed,actual in [(0,'1'),(1,'2'),(2,'unknown')]:
        common={'prompt_id':'p','seed':seed,'step':8,'question_id':'q','skill':'count',
                'desired_answer':'3','candidates':['1','2','3','unknown'],'image_sha256':str(seed)}
        labels.append({**common,'observed_answer':actual})
        for source in ['DYNAMIC','SEED','EMPTY','VIT_IMAGE']:
            prediction=actual if source in ['DYNAMIC','VIT_IMAGE'] else '3'
            prob={a:.01 for a in common['candidates']};prob[prediction]=.97
            rows.append({**common,'source':source,'prediction':prediction,'choice_probability':prob})
    summary=summarize_probe(rows,labels,resamples=20)
    assert summary['known_labels']==2 and summary['unknown_labels']==1
    assert summary['label_coverage']['step_8']=={'known':2,'unknown':1}
    allmetrics=summary['by_scope']['all']
    assert allmetrics['DYNAMIC']['accuracy_vs_observed']==1
    assert allmetrics['SEED']['prompt_echo_rate_on_mismatch']==1
    assert summary['dynamic_vs_seed']['accuracy_delta']['mean']==1
    assert summary['different_observed_states_same_prompt']['DYNAMIC']['both_states_correct_rate']==1
    assert summary['quality_claim'] is False
    with pytest.raises(ValueError,match='requires'):summarize_probe(rows[:-1],labels,resamples=20)
    with pytest.raises(ValueError,match='duplicate'):summarize_probe(rows+rows[:1],labels,resamples=20)


def test_guided_x0_capture_does_not_change_sampling_trajectory():
    from qwen_latent_cot.bagel.inferencer import T2IGenerator
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1),requires_grad=False)
        def _forward_flow(self,**kwargs):return kwargs['x_t']*.25+1
        def generate_image(self,**kwargs):
            x=torch.ones(1,2)
            for t in [.9,.5,.1]:
                velocity=self._forward_flow(x_t=x,timestep=torch.tensor([t]))
                x=x-.1*velocity
            return [x]
    model=Model();bundle=SimpleNamespace(model=model,vae=None)
    runtime=SimpleNamespace(diagnostics=[],probe_capture=None,progress=0)
    generator=T2IGenerator(bundle,runtime)
    generator.prepare=lambda *a:({},['same_noise'])
    decoded=[]
    def decode(latent,shape):
        decoded.append(latent.clone());return Image.new('RGB',(28,28))
    generator.decode=decode
    original=model._forward_flow
    generator.generate(['x'],[(28,28)],[0],num_timesteps=4)
    expected_final=decoded[-1];decoded.clear()
    runtime.probe_capture=ProbeCapture([0,1])
    generator.generate(['x'],[(28,28)],[0],num_timesteps=4)
    assert torch.equal(decoded[-1],expected_final)
    assert torch.allclose(decoded[0],torch.ones(1,2)-.9*1.25)
    assert set(runtime.probe_capture.images)=={0,1} and model._forward_flow==original
