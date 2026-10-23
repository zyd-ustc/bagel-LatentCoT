"""Cross-time contract on a real tiny BAGEL MoT, not mocked attention."""
import tempfile
from pathlib import Path
import json
import torch
from test_mot_edit import setup
from qwen_latent_cot.mot_edit.cross_time import CrossTimeEdit,sample
from qwen_latent_cot.mot_edit.cross_time_training import edit_pair
from qwen_latent_cot.mot_edit.roles import KV
from qwen_latent_cot.mot_edit.training import train
from qwen_latent_cot.mot_edit.data import prepare,Corpus
from qwen_latent_cot.mot_edit import checkpoint
import unittest


def model():
    m=setup();cfg=dict(m.config,pipeline='cross_time',rollback_index=0,read_index=2,caption_weight=1.)
    return CrossTimeEdit(m.runtime,cfg)


class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_shared_reader_has_last_layer_qo_gradient(self):
        m=model();m.caption_loss(torch.randn(4,8),'two red cubes').backward()
        for name in ('q_proj','o_proj'):
            self.assertGreater(float(m.observation.blocks[-1].projections[name].b.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in m.runtime.model.parameters()))
    def test_goal_preserved_native_parity_without_feedback(self):
        m=model();x=torch.randn(4,8);f=m.runtime.flow('two red cubes')
        bank=m.observation_bank(x);empty=tuple(KV(k.keys[:0],k.values[:0],k.positions[:0]) for k in bank)
        w=m.edit_velocity(x,.8,empty,f);vc=m.runtime.predict(x,.8,f,capture=False)[0]
        torch.testing.assert_close(w,vc.float(),rtol=2e-3,atol=2e-4)
    def test_persistent_steps_and_saved_state_rollback(self):
        m=model();m.set_stage('edit')
        image,trace=sample(m,'p',17,2)
        self.assertEqual(len([x for x in trace if x['event']=='feedback_step']),4)
        roll=[x for x in trace if x['event']=='rollback'];self.assertEqual(len(roll),2)
        self.assertTrue(all(x['anchor_error_rms']==0 for x in roll))
        self.assertTrue(all(x['read_t']<x['write_t'] for x in roll))
        self.assertEqual(trace[-1]['euler_updates'],7)
        from qwen_latent_cot.mot_edit.sampling import sample as native
        before,_=sample(m,'p',17,0)
        with torch.no_grad():
            for p in m.all_adapters().values():p.normal_()
        after,_=sample(m,'p',17,0)
        self.assertEqual(before.tobytes(),after.tobytes())
    def test_actual_replay_training_all_roles(self):
        m=model();m.set_stage('edit');x=torch.randn(4,8)
        result=edit_pair(m,x,m.runtime.flow('p'),17,1,1)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result))
        for role in ('observation','feedback','editor'):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in getattr(m,role).adapters().values() if p.grad is not None),0)
        self.assertEqual(result[4],0)
    def test_exact_resume_with_caption_retention(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);source=p/'pairs.jsonl';source.write_text(''.join(json.dumps(dict(uid=str(i),image=f'/image/{i}',prompt=f'caption {i}'))+'\n' for i in range(600)))
            prepare(source,p/'corpus');m=model();m.runtime.encode=lambda *a:torch.ones(4,8)
            train(m,Corpus(p/'corpus'),p/'full',{'test':1})
            saved=checkpoint.read(p/'full/step_000002.pt');n=model();n.runtime.encode=lambda *a:torch.ones(4,8)
            train(n,Corpus(p/'corpus'),p/'resumed',{'test':1},saved)
            self.assertEqual(saved['kind'],'bagel_mot_cross_time_v3')
            for name,v in m.all_adapters().items():torch.testing.assert_close(v,n.all_adapters()[name],atol=0,rtol=0)
            rows=[json.loads(l) for l in (p/'full/train.jsonl').read_text().splitlines()]
            self.assertIn('caption_loss',rows[-1]);self.assertGreater(rows[-1]['caption_loss'],0)

if __name__=='__main__':unittest.main()
