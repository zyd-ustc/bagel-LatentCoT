"""Post-CFG residual contract and train-only constant-field baseline."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from test_correction import tiny
from test_correction_data import fixture
from qwen_latent_cot.correction.calibration import GlobalCalibration
from qwen_latent_cot.correction.data import Corpus
from qwen_latent_cot.correction.model import proposal_from,correction_loss
from qwen_latent_cot.correction.training import train,paired_state
from qwen_latent_cot.correction.sampling import sample,guided,times,active_steps
from qwen_latent_cot.correction import checkpoint


class PostCFGTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_zero_delta_and_guided_proposal(self):
        model=tiny(post_cfg=True);runtime=model.runtime;flow=runtime.flow('a red cube')
        x=torch.randn(len(flow['packed_vae_token_indexes']),8);t=.8
        conditional,_=runtime.predict(x,t,flow);reference=runtime.reference(x,t,flow)
        base,context=runtime.proposal(x,t,flow)
        torch.testing.assert_close(base,guided(conditional,reference,model.config),rtol=0,atol=0)
        self.assertEqual(len(context),len(model.layers))
        native,_=sample(model,'a red cube',17,native=True);corrected,_=sample(model,'a red cube',17)
        self.assertEqual(native.tobytes(),corrected.tobytes())
        old=tiny();old_image,_=sample(old,'a red cube',17,native=True)
        self.assertEqual(native.tobytes(),old_image.tobytes())

    def test_constant_correction_after_cfg_once(self):
        model=tiny(post_cfg=True);runtime=model.runtime
        # An identity image decoder exposes the actual final latent for Euler arithmetic.
        runtime.image=lambda x:x.clone()
        nodes=times(model.config);n=4
        base=torch.full((n,8),.375,dtype=torch.bfloat16)
        runtime.flow=lambda prompt:{'packed_vae_token_indexes':torch.arange(n)}
        runtime.proposal=lambda x,t,flow,capture=True:(base,())
        model.forward=lambda *args:torch.full((n,8),.125)
        native,_=sample(model,'prompt',17,native=True);corrected,_=sample(model,'prompt',17)
        expected=torch.full_like(native,-.125*(nodes[0]-nodes[active_steps(model.config)]))
        torch.testing.assert_close(corrected-native,expected,rtol=1e-5,atol=2e-7)

    def test_least_squares_train_mean(self):
        c=GlobalCalibration([1.,.5],2,'cpu');pending=torch.zeros(2,3)
        c.observe(pending,1.,torch.tensor([[1.,2.],[3.,4.]]));c.observe(pending,1.,torch.tensor([[4.,5.],[6.,7.]]))
        c.update(pending);torch.testing.assert_close(c(1.),torch.tensor([3.5,4.5]))
        self.assertEqual(c.counts.tolist(),[2.,0.])
        before=c.sums.clone();c(1.);self.assertTrue(torch.equal(c.sums,before))
        with self.assertRaises(ValueError):c(.5)

    def test_three_arm_evaluation_does_not_update_baseline(self):
        from qwen_latent_cot.correction.training import semantic_validation,validate
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root);model=tiny(post_cfg=True)
            model.runtime.bundle.vae.encode=lambda x:torch.nn.functional.avg_pool2d(x[:,:2],8)
            model.calibration.counts.fill_(1)
            model.calibration.sums.fill_(.01)
            prompts=root/'prompts.jsonl';prompts.write_text(json.dumps(dict(prompt='a red cube',annotations={'count':1}))+'\n')
            model.config['semantic_prompts']=str(prompts)
            before=model.calibration.sums.clone()
            semantic_validation(model,root,100)
            records=json.loads((root/'semantic_000100/records.json').read_text())
            self.assertEqual(set(records[0]['arms']),{'native','global','corrected'})
            for value in records[0]['arms'].values():self.assertTrue((root/value).is_file())
            corpus=Corpus(root);validate(model,corpus.validation(1),root,100)
            metrics=json.loads((root/'validation_000100/records.json').read_text())
            self.assertTrue(all('global_mse' in row for row in metrics))
            self.assertTrue(torch.equal(before,model.calibration.sums));corpus.close()

    def test_resume_global_baseline_and_guided_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root)
            def setup():
                model=tiny(post_cfg=True);model.config.update(global_batch=2,steps=3,save_steps=[2,3],val_images=0,warmup_steps=0)
                model.runtime.config.update(model.config)
                model.runtime.bundle.vae.encode=lambda x:torch.nn.functional.avg_pool2d(x[:,:2],8)
                return model
            full=setup();train(full,Corpus(root),root/'full',{'test':'same'})
            resumed=setup();saved=checkpoint.read(root/'full/step_000002.pt')
            train(resumed,Corpus(root),root/'resume',{'test':'same'},saved=saved)
            for name,value in full.trainable().items():torch.testing.assert_close(value,resumed.trainable()[name],rtol=0,atol=0)
            torch.testing.assert_close(full.calibration.sums,resumed.calibration.sums,rtol=0,atol=0)
            torch.testing.assert_close(full.calibration.counts,resumed.calibration.counts,rtol=0,atol=0)
            self.assertEqual(float(full.calibration.counts.sum()),6.)
            corpus=Corpus(root);row=corpus.validation(1)[0]
            x0,eps,x,flow,base,context=paired_state(full.runtime,row,.8,17)
            expected,_=full.runtime.proposal(x,.8,flow)
            torch.testing.assert_close(base,expected,rtol=0,atol=0)
            delta=full(proposal_from(x,.8,base),.8,flow,context)
            torch.testing.assert_close(correction_loss(delta,base,x0,eps),(base.float()+delta-(eps-x0.float())).square().mean())
            corpus.close()


if __name__=='__main__':unittest.main()
