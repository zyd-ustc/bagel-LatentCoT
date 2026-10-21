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
