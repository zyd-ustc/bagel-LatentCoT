"""Native UND smoke: full-depth GEN reading, selected gradients and resume."""
from types import SimpleNamespace
import json
import torch
from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import Qwen2Config, Qwen2ForCausalLM
from qwen_latent_cot.diagnostics.memory import MemoryWriter, neutral_seeds, project, rope_for, rotate, KV, consume_bank
from qwen_latent_cot.diagnostics.native import teacher_forcing
from qwen_latent_cot.diagnostics.visual import VisualStudent, validate_config
from qwen_latent_cot.diagnostics.visual_record import save_visual_checkpoint, load_visual_checkpoint
from qwen_latent_cot.diagnostics.record import FrozenGuard


def test_native_und_smoke(tmp_path):
    torch.manual_seed(7)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                        num_attention_heads=4, num_key_value_heads=2, layer_module='Qwen2MoTDecoderLayer')
    lm = Qwen2ForCausalLM(config).to(torch.bfloat16).eval().requires_grad_(False)
    cfg = json.load(open('configs/visual_memory_pilot.json'))
    cfg['source_layers'] = list(range(3))
    decoder = lm.model; writer = MemoryWriter(decoder, 'C', 16)
    ids = torch.arange(16)
    from qwen_latent_cot.diagnostics.numerics import autocast_for
    with autocast_for(torch.device('cpu')):
        seeds = neutral_seeds(decoder, ids)
    runtime = SimpleNamespace(decoder=decoder, writer=writer, model=SimpleNamespace(language_model=lm),
        config=cfg, device=torch.device('cpu'), seeds=seeds, seed_ids=ids.tolist(),
        native_files={}, data_hashes={'corpus':'smoke'})
    student = VisualStudent(runtime); validate_config(runtime, 'C2-observe')
    assert not list(writer.parameters())
    assert all('moe_gen' not in name and 'mlp' not in name for name in student.native_names)
    guard = FrozenGuard(lm, exclude_trainable=True)
    source=[]
    with torch.no_grad(), autocast_for(runtime.device):
        for i, layer in enumerate(decoder.layers):
            h=torch.randn(8,32).to(torch.bfloat16)
            _,k,v=project(layer.self_attn,h,'gen',torch.tensor([],dtype=torch.long),torch.arange(8))
            source.append(KV(i,rotate(k,rope_for(decoder,h,torch.arange(8))),v,(8,)))
    optimizer=torch.optim.AdamW(student.parameters(),lr=1e-3)
    before=student.und[0].detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        bank=student.write(tuple(source),.15)
        with autocast_for(runtime.device):
            loss,_=teacher_forcing(SimpleNamespace(model=SimpleNamespace(language_model=lm)),[1,2],[3,4,5],consume_bank(decoder,bank,[0]),16)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())
        assert all(any(p.grad.abs().max()>0 for p in group) for group in student.parameter_groups().values())
        optimizer.step();guard.check()
    assert not torch.equal(before,student.und[0])
    with torch.no_grad():
        a=student.write(tuple(source),.15)
        changed=list(source);kv=changed[-1];changed[-1]=KV(kv.layer,kv.keys,kv.values+3,kv.lengths)
        b=student.write(tuple(changed),.15)
        assert not torch.equal(a.layers[-1].values,b.layers[-1].values)
    path=tmp_path/'last.pt'
    save_visual_checkpoint(path,student,2,'C2-observe',optimizer=optimizer,schedule={'step':2})
    expected=student.und[0].detach().clone()
    with torch.no_grad():student.und[0].zero_()
    result=load_visual_checkpoint(path,student,optimizer=optimizer,resume=True,expected_stage='C2-observe')
    torch.testing.assert_close(expected,student.und[0],rtol=0,atol=0)
    assert result['step']==2 and result['schedule']=={'step':2}


def test_c2_feedback_text_smoke(tmp_path):
    from test_memory_topology import lm as lm_fixture, runtime as runtime_fixture
    from qwen_latent_cot.diagnostics.flow_training import ExecutionCounter
    from qwen_latent_cot.diagnostics.visual_objectives import training_objective
    from qwen_latent_cot.diagnostics.curriculum import NoisePolicy
    from qwen_latent_cot.diagnostics.task_evaluation import evaluate_tasks
    from qwen_latent_cot.diagnostics.training_logging import validation_metrics
    from PIL import Image
    runtime=runtime_fixture.__wrapped__(lm_fixture.__wrapped__(),tmp_path)
    cfg=json.load(open('configs/visual_feedback.json'))
    cfg.update(resolution=32,source_layers=[0,1,2],sampler_steps=4,
               output_token_budgets=[2],max_caption_tokens=2)
    runtime.config=cfg;student=VisualStudent(runtime)
    rows=[]
    for index,color in enumerate(('red','blue')):
        image=tmp_path/f'{color}.png';Image.new('RGB',(32,32),color).save(image)
        rows.append(dict(sample_id=str(index),family_id='pair',pair_id='pair',source='smoke',
            task='feedback',image=str(image),prompt='a blue object',caption='Make it blue.' if index==0 else 'No change is needed.',
            target_caption='a blue object',facts=[],label_type='clean',label_kind='feedback',
            flow_supervision=False,annotation={'scope':'endpoint','origin':'source_annotation'},
            feedback=dict(observation='',gap='',correction='Make it blue.' if index==0 else '',satisfied=index==1,origin='source_annotation')))
    counter=ExecutionCounter(runtime)
    try:
        for row in rows:
            student.zero_grad(set_to_none=True)
            state=runtime.training_state(row,.15,0);before=counter.counts[0]
            losses,details=training_objective(student,row,state,'C2-feedback')
            losses['sem'].backward()
            assert counter.counts[0]-before==1 and details['gen_calls']==1
            assert losses['teacher']==0 and losses['flow']==0
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())
        output=tmp_path/'validation';output.mkdir()
        records=evaluate_tasks(student,rows,output,NoisePolicy(cfg),'C2-feedback')
        assert records[0]['t']==records[1]['t']
        assert all('source_swap' in r and 'feedback_nll' in r for r in records)
        assert not any(output.glob('*.png'))
        metrics=validation_metrics(records)
        assert metrics['val/feedback/balanced_nll']==(metrics['val/feedback/repair_nll']+metrics['val/feedback/hold_nll'])/2
    finally:counter.close()
