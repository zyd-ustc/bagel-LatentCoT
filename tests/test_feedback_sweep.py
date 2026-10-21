"""Actual tiny BAGEL execution for 1–4 feedback updates and frozen config."""
import importlib.util
from pathlib import Path
import pytest
import torch
from test_memory_topology import lm,runtime
from test_visual_loop import visual_runtime
from test_stage_a import stage_student


def test_feedback_override_default_parity_and_explicit_inference_only(stage_student):
    student,state=stage_student
    with torch.no_grad():
        default=student(state,guidance=True)
        explicit=student(state,guidance=True,feedback_updates=1)
    torch.testing.assert_close(default.velocity,explicit.velocity,rtol=0,atol=0)
    assert student.config['feedback_updates']==1 and len(student.bridge.vel_scale)==1
    with pytest.raises(ValueError,match='inference-only'):student(state,feedback_updates=3)
    with torch.no_grad(),pytest.raises(ValueError,match='positive integer'):student(state,feedback_updates=0)


def test_real_blocks_cfg_once_and_last_gate_reused(stage_student,monkeypatch):
    import qwen_latent_cot.interaction.loop as loop
    student,state=stage_student
    with torch.no_grad():student.bridge.vel_scale.copy_(torch.tensor([.8]))
    original=loop.conditional_gen;scales=[]
    def record(*args,**kwargs):
        if kwargs.get('velocity_scale') is not None:scales.append(float(kwargs['velocity_scale']))
        return original(*args,**kwargs)
    monkeypatch.setattr(loop,'conditional_gen',record)
    for n in (1,2,3,4):
        calls=[]
        layer=student.runtime.decoder.layers[0]
        handle=layer.register_forward_pre_hook(lambda m,a,kw:calls.append(kw['past_key_values']) if kw.get('mode')=='gen' else None,with_kwargs=True)
        scales.clear()
        try:
            with torch.no_grad():result=student(state,guidance=True,feedback_updates=n)
        finally:handle.remove()
        assert len(result.exits)==len(result.banks)==n
        assert sum(c is state.flow['past_key_values'] for c in calls)==n+1
        assert sum(c is state.flow['cfg_text_past_key_values'] for c in calls)==1
        assert scales==pytest.approx(([.8]*n))
        assert torch.isfinite(result.velocity).all()
        for value in result.exits[:-1]:
            assert len(value.source)==3
            assert value.source[0].positions==result.baseline.source[0].positions
        assert result.exits[-1].source==()
    assert student.config['feedback_updates']==1


def test_shared_sampler_reports_actual_passes(stage_student):
    from qwen_latent_cot.interaction.sampling import sample
    student,_=stage_student
    student.runtime.config['sampler_steps']=3
    for n in (1,2,3,4):
        _,records=sample(student,'a red token',17,feedback_updates=n)
        assert [r['loop_active'] for r in records]==[True,False]
        assert [r['passes'] for r in records]==[n+1,1]
    _,records=sample(student,'a red token',17,native=True)
    assert not any(r['loop_active'] for r in records)
    with pytest.raises(ValueError,match='Native'):sample(student,'a red token',17,native=True,feedback_updates=1)


def test_maximum_atom_selection_is_original_and_reproducible():
    spec=importlib.util.spec_from_file_location('sweep',Path('scripts/sample_feedback_sweep.py'))
    sweep=importlib.util.module_from_spec(spec);spec.loader.exec_module(sweep)
    rows=sweep.read_rows('data/geneval2_800.jsonl')
    selected,meta=sweep.select_hard(rows)
    assert len(selected)==32 and meta['pool_size']==100 and meta['atom_count']==10
    assert all(r['atom_count']==10 for r in selected)
    assert selected==sweep.select_hard(rows)[0]
    source={r['prompt_id']:r for r in rows}
    assert all(r==source[r['prompt_id']] for r in selected)
