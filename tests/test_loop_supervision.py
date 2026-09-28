import pytest
import torch

from qwen_latent_cot.bagel.loop_supervision import loop_supervision_loss
from qwen_latent_cot.bagel.loop import configure_loop_trainable_routes
from test_memory_grounding import tiny_bagel, flow_kwargs


@pytest.mark.parametrize("writes", [1,2,3])
def test_full_suffix_outputs_final_parity_and_gradients(writes):
    model = tiny_bagel()
    configure_loop_trainable_routes(model, ["q_proj", "q_proj_moe_gen"])
    kwargs = flow_kwargs()
    out = model.forward_loop_supervised(**kwargs, num_write_rounds=writes)
    assert len(out.write_round_velocities) == len(out.write_round_memories) == writes
    assert out.final_velocity is out.write_round_velocities[-1]
    with torch.no_grad():
        legacy = model._forward_flow_loop(**kwargs, memory_loop_repeat=1+writes)[0]
    assert torch.equal(out.final_velocity, legacy)
    loss = loop_supervision_loss(out.write_round_velocities, torch.zeros_like(out.final_velocity),
                                 weights=[1.]*writes)
    loss.loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_round_outputs_equal_independent_shallower_replays():
    model = tiny_bagel()
    kwargs = flow_kwargs()
    with torch.no_grad():
        out = model.forward_loop_supervised(**kwargs, num_write_rounds=3)
        for r in range(3):
            shallow = model.forward_loop_supervised(**kwargs, num_write_rounds=r+1)
            assert torch.equal(out.write_round_velocities[r], shallow.final_velocity)


def test_distillation_detaches_final_teacher_and_requires_gate():
    shallow = torch.ones(2,3, requires_grad=True)
    final = torch.zeros(2,3, requires_grad=True)
    with pytest.raises(ValueError, match="held-out"):
        loop_supervision_loss([shallow, final], final, weights=[.3,1.], loop_distill=True)
    loss = loop_supervision_loss([shallow, final], torch.zeros_like(final), weights=[.3,1.],
                                 loop_distill=True, final_round_validated=True)
    loss.distillation.backward()
    assert shallow.grad is not None and final.grad is None


def test_monotonic_loss_penalizes_only_worse_later_rounds():
    vs = [torch.full((1,2), value) for value in (3.,2.,1.)]
    good = loop_supervision_loss(vs, torch.zeros(1,2), weights=[.3,.5,1.])
    bad = loop_supervision_loss(vs[::-1], torch.zeros(1,2), weights=[.3,.5,1.])
    assert good.monotonic == 0
    assert bad.monotonic == 8
    assert good.deep_supervision == pytest.approx(5.7)


def test_reject_cfg_or_full_depth_in_supervision():
    model = tiny_bagel()
    with pytest.raises(ValueError, match="CFG"):
        model.forward_loop_supervised(**flow_kwargs(), cfg_text_scale=4.)
    with pytest.raises(ValueError, match="same_depth"):
        model.forward_loop_supervised(**flow_kwargs(), recycle_mode="full_depth")
