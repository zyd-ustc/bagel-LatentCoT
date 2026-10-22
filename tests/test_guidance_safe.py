"""Guidance-preserving target, actual sampling projection, metrics and exact resume."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from torch.nn import functional as F
from test_correction import tiny
from test_correction_data import fixture
from qwen_latent_cot.correction.objective import remove_anti_guidance,residual_target,applied_correction,guidance_coefficient,MODE
from qwen_latent_cot.correction.sampling import sample,times,active_steps
from qwen_latent_cot.correction.training import train,validate
from qwen_latent_cot.correction.data import Corpus
from qwen_latent_cot.correction import checkpoint


def setup():
    model=tiny(post_cfg=True);model.config['correction_objective']=MODE;model.runtime.config.update(model.config)
    model.runtime.bundle.vae.encode=lambda x:F.avg_pool2d(x[:,:2],8)
    return model


class GuidanceSafeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_projection_geometry_zero_and_positive(self):
        g=torch.tensor([[2.,0.,0.]]);d=torch.tensor([[-3.,4.,5.]])
        actual=remove_anti_guidance(d,g);torch.testing.assert_close(actual,torch.tensor([[0.,4.,5.]]))
        torch.testing.assert_close(remove_anti_guidance(-d,g),-d)
        torch.testing.assert_close(remove_anti_guidance(d,torch.zeros_like(g)),d)
        torch.testing.assert_close(remove_anti_guidance(actual,g),actual)

    def test_target_uses_conditional_not_guided_and_keeps_gradients(self):
        cfg=dict(correction_space='post_cfg',correction_objective=MODE)
        vc=torch.zeros(1,2,requires_grad=True);vg=torch.tensor([[3.,0.]],requires_grad=True)
        x0=torch.zeros(1,2);eps=torch.tensor([[1.,2.]])
        target=residual_target(vg,x0,eps,cfg,vc)
        torch.testing.assert_close(target,eps);self.assertFalse(target.requires_grad)
        # Legitimate negative parallel conditional error is excluded by the declared objective.
        other=residual_target(vg,x0,torch.tensor([[-1.,2.]]),cfg,vc)
        torch.testing.assert_close(other,torch.tensor([[0.,2.]]))
        raw=torch.tensor([[-1.,2.]],requires_grad=True)
        F.mse_loss(raw,other).backward();self.assertLess(float(raw.grad[0,0]),0)
        self.assertIsNone(vc.grad);self.assertIsNone(vg.grad)

    def test_actual_sampler_removes_anti_guidance(self):
        model=setup();rt=model.runtime;rt.image=lambda x:x.clone();n=4
        vc=torch.zeros(n,8,dtype=torch.bfloat16);base=vc.clone();base[:,0]=1
        rt.flow=lambda prompt:{'packed_vae_token_indexes':torch.arange(n)}
        rt.proposal=lambda x,t,flow,capture=True,with_conditional=False:(base,(),vc) if with_conditional else (base,())
        raw=torch.zeros(n,8);raw[:,0]=-.5;raw[:,1]=.25
        model.forward=lambda *args:raw
        native,_=sample(model,'p',17,native=True);actual,trace=sample(model,'p',17)
        extent=times(model.config)[0]-times(model.config)[active_steps(model.config)]
        expected=torch.zeros_like(raw);expected[:,1]=-.25*extent
        torch.testing.assert_close(actual-native,expected,atol=3e-7,rtol=1e-5)
        for row in trace:
            if row['correction_active']:
                self.assertAlmostEqual(row['guidance_projection_raw'],-.5)
                self.assertAlmostEqual(row['guidance_projection_applied'],0.)

    def test_zero_initialization_preserves_native(self):
        model=setup();native,_=sample(model,'red cube',17,native=True);corrected,_=sample(model,'red cube',17)
        self.assertEqual(native.tobytes(),corrected.tobytes())

    def test_exact_resume_and_three_arm_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root)
            def configured():
                m=setup();m.config.update(global_batch=2,steps=3,save_steps=[2,3],val_images=0,warmup_steps=0)
                m.runtime.config.update(m.config);return m
            full=configured();train(full,Corpus(root),root/'full',{'test':'same'})
            resumed=configured();saved=checkpoint.read(root/'full/step_000002.pt')
            train(resumed,Corpus(root),root/'resume',{'test':'same'},saved=saved)
            for name,p in full.trainable().items():torch.testing.assert_close(p,resumed.trainable()[name],rtol=0,atol=0)
            torch.testing.assert_close(full.calibration.sums,resumed.calibration.sums,rtol=0,atol=0)
            torch.testing.assert_close(full.calibration.counts,resumed.calibration.counts,rtol=0,atol=0)
            records=[json.loads(r) for r in (root/'full/train.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['applied_negative_fraction']==0 for r in records))
            # Short toy training may not cover each time: complete the diagnostic baseline, not its model.
            full.calibration.counts.clamp_min_(1)
            corpus=Corpus(root);validate(full,corpus.validation(1),root,3);corpus.close()
            val=json.loads((root/'validation_000003/records.json').read_text())
            self.assertTrue(all(r['guidance_projection_applied']>=-1e-5 and 'objective_mse' in r for r in val))
            self.assertTrue(all((root/f'validation_000003/000_{arm}.png').exists() for arm in ('native','global','corrected')))


if __name__=='__main__':unittest.main()
