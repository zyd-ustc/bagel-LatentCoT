"""Real tiny BAGEL: ablations change only their named path."""
import importlib.util
from pathlib import Path
import sys
import torch
from test_memory_topology import lm,runtime
from test_visual_loop import visual_runtime
from test_stage_a import stage_student


def test_default_parity_and_masked_reentry_controls(stage_student,monkeypatch):
    import qwen_latent_cot.interaction.loop as loop
    student,state=stage_student
    original=loop.conditional_gen;calls=[]
    def record(*args,**kwargs):
        if len(args)>2 and args[2] is not None: calls.append(kwargs)
        return original(*args,**kwargs)
    monkeypatch.setattr(loop,'conditional_gen',record)
    with torch.no_grad():
        default=student(state,guidance=True)
        explicit=student(state,guidance=True,memory_masked=False,reentry=True)
        torch.testing.assert_close(default.velocity,explicit.velocity,atol=0,rtol=0)
        calls.clear();student(state,guidance=True,feedback_updates=1,memory_masked=True)
        assert len(calls)==1 and calls[0]['masked'] and calls[0]['hidden_input'] is not None
        calls.clear();student(state,guidance=True,feedback_updates=1,reentry=False)
        assert len(calls)==1 and not calls[0]['masked'] and calls[0]['hidden_input'] is None


def test_parent_no_loop_keeps_parent_context(stage_student,monkeypatch):
    from qwen_latent_cot.interaction.sampling import sample
    student,_=stage_student;student.runtime.config['sampler_steps']=3
    def forbidden(*a,**k):raise AssertionError('Native context must not run for parent arm')
    monkeypatch.setattr(student,'native_context',forbidden)
    _,records=sample(student,'a red token',17,loop_enabled=False)
    assert all(not r['loop_active'] and r['passes']==1 for r in records)


def test_probe_matches_conditional_velocity_before_cfg(stage_student,monkeypatch):
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    spec=importlib.util.spec_from_file_location('ablation',Path('scripts/sample_loop_ablation.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    student,state=stage_student
    with torch.no_grad(),module.conditional_probe(student) as rows:
        output=student(state,guidance=False,feedback_updates=1)
    expected=(output.velocity.float()-output.baseline.velocity.float()).square().mean().sqrt()
    assert len(rows)==1
    assert abs(rows[0]['conditional_delta_rms']-float(expected))<1e-5


def test_masked_memory_skips_consumer_and_preserves_native_positions(stage_student):
    from dataclasses import replace
    from qwen_latent_cot.diagnostics.flow_training import conditional_gen
    student,state=stage_student
    update=student.runtime.decoder.layers[0].self_attn.q_proj_moe_gen.update
    with torch.no_grad(): update.b.fill_(.02)
    calls=[]
    handle=update.register_forward_hook(lambda *args:calls.append(1))
    try:
        with torch.no_grad():
            student(state,guidance=True,feedback_updates=1,memory_masked=True,reentry=False)
            assert not calls
            calls.clear()
            actual=student(state,guidance=True,feedback_updates=1,memory_masked=True,reentry=False,consumer_enabled=False)
            assert not calls
            expected=conditional_gen(student.runtime,state,guidance=True,capture=False,
                                     native_scope=student.native_context)
        torch.testing.assert_close(actual.velocity,expected.velocity,rtol=1e-5,atol=1e-6)
    finally:handle.remove()


def test_gamma_multiplier_scales_and_restores_on_failure(stage_student,monkeypatch):
    import pytest
    import qwen_latent_cot.interaction.loop as loop
    from qwen_latent_cot.interaction.sampling import sample
    student,_=stage_student;student.runtime.config['sampler_steps']=3
    before=student.bridge.vel_scale.detach().clone();scales=[];original=loop.conditional_gen
    def record(*args,**kwargs):
        if kwargs.get('velocity_scale') is not None:scales.append(float(kwargs['velocity_scale']))
        return original(*args,**kwargs)
    monkeypatch.setattr(loop,'conditional_gen',record)
    sample(student,'a red token',17,feedback_updates=1,gamma_multiplier=.2)
    assert scales==pytest.approx([float(before[0])*.2])
    torch.testing.assert_close(student.bridge.vel_scale,before,atol=0,rtol=0)
    def fail(node):raise RuntimeError('intentional test failure')
    with pytest.raises(RuntimeError,match='intentional'):
        sample(student,'a red token',17,feedback_updates=1,gamma_multiplier=.3,on_step=fail)
    torch.testing.assert_close(student.bridge.vel_scale,before,atol=0,rtol=0)
