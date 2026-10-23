"""Real tiny MoT: head reuse, role isolation, state dependence, gradients and resume."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from types import SimpleNamespace
from test_correction import tiny
from qwen_latent_cot.mot_edit.model import MoTEdit,project_safe,apply_edit
from qwen_latent_cot.mot_edit.sampling import sample
from qwen_latent_cot.mot_edit import checkpoint
from qwen_latent_cot.mot_edit.data import prepare,Corpus,Sampler
from qwen_latent_cot.mot_edit.training import train


def setup():
    old=tiny();rt=old.runtime
    cfg=json.loads(Path('configs/mot_edit_8000.json').read_text())
    cfg.update(resolution=32,lora_rank=4,slots=4,sampler_steps=4,feedback_indices=[1],global_batch=1,
        steps=3,observation_steps=1,save_steps=[1,2,3],semantic_eval_steps=[],observation_eval_steps=[],warmup_steps=1)
    rt.config.update(cfg)
    return MoTEdit(rt,cfg)


class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_native_heads_roles_and_gradients(self):
        m=setup();rt=m.runtime;x=torch.randn(4,8);before={n:p.clone() for n,p in rt.model.named_parameters()}
        loss=m.caption_loss(x,'two red cubes');loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.observation.adapters().values() if p.grad is not None),0)
        self.assertTrue(all(p.grad is None for p in rt.model.parameters()))
        m.zero_grad(set_to_none=True);m.set_stage('edit');flow=rt.flow('two red cubes')
        bank=m.observation_bank(x);f=m.feedback_bank(bank,flow);w=m.edit_velocity(x,.55,f)
        w.square().mean().backward()
        for role in ('observation','feedback','editor'):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in getattr(m,role).adapters().values() if p.grad is not None),0)
        self.assertEqual(w.shape,x.shape)
        self.assertTrue(all(torch.equal(p,before[n]) for n,p in rt.model.named_parameters()))
        self.assertTrue(all(n.endswith(('.a','.b')) for n in m.trainable()))
    def test_actual_state_editor_source_and_cross_round_gradient(self):
        m=setup();m.set_stage('edit');rt=m.runtime;x=torch.randn(4,8);flow=rt.flow('p')
        vg,_,vc=rt.proposal(x,.55,flow,capture=False,with_conditional=True)
        p=x-.55*vg;bank=m.observation_bank(p)
        changed=m.observation_bank(p+2)
        self.assertFalse(torch.equal(bank[-1].values,changed[-1].values))
        seen=[]
        handle=rt.model.vae2llm.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
        state1,w1,vc1,vg1,d1=m.round(x,.55,flow)
        handle.remove()
        torch.testing.assert_close(seen[-1],x.to(seen[-1].dtype))
        torch.testing.assert_close(state1,x.float()-.55*d1)
        state2,w2,vc2,vg2,d2=m.round(state1,.55,flow)
        torch.testing.assert_close(state2,state1-.55*d2)
        self.assertGreaterEqual(float((d2.detach()*(vg2-vc2).float()).sum()),-1e-4)
        cross=torch.autograd.grad(w2.square().mean(),w1,retain_graph=True)[0]
        self.assertGreater(float(cross.abs().sum()),0)
        w2.square().mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.observation.adapters().values() if p.grad is not None),0)
    def test_native_k0_and_checkpointing(self):
        m=setup();rt=m.runtime
        # K0 output is independent of all edit adapters, even after arbitrary changes.
        original,_=sample(m,'p',17,0)
        from qwen_latent_cot.correction.sampling import sample as old_sample
        reference,_=old_sample(SimpleNamespace(runtime=rt,config=m.config),'p',17,native=True)
        self.assertEqual(original.tobytes(),reference.tobytes())
        with torch.no_grad():
            for p in m.all_adapters().values():p.normal_()
        actual,_=sample(m,'p',17,0);self.assertEqual(original.tobytes(),actual.tobytes())
        m=setup();x=torch.randn(4,8);flow=m.runtime.flow('p');m.set_stage('edit')
        a=m.round(x,.55,flow)[1];m.config['activation_checkpointing']=False;b=m.round(x,.55,flow)[1]
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_edit_commits_clean_displacement_to_actual_state(self):
        x=torch.tensor([[2.,3.]])
        vc=torch.tensor([[.1,.2]]);vg=torch.tensor([[.3,.2]])
        w=torch.tensor([[.5,.6]])
        state,delta=apply_edit(x,.5,vc,vg,w)
        torch.testing.assert_close(state,x-.5*delta)
        # The clean preview displacement equals the committed latent displacement.
        torch.testing.assert_close(state-x,(x-.5*(vg+delta))-(x-.5*vg))

    def test_projection_zero_and_negative(self):
        g=torch.tensor([[1.,0.]])
        torch.testing.assert_close(project_safe(torch.tensor([[-2.,3.]]),g),torch.tensor([[0.,3.]]))
        d=torch.randn(4,8);torch.testing.assert_close(project_safe(d,torch.zeros_like(d)),d)
    def test_train_stages_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'pairs.jsonl'
            source.write_text(''.join(json.dumps(dict(uid=str(i),image=f'/image/{i}',prompt=f'caption {i}'))+'\n' for i in range(600)))
            prepare(source,root/'corpus');m=setup();m.runtime.encode=lambda *a:torch.ones(4,8)
            train(m,Corpus(root/'corpus'),root/'full',{'source':'test'})
            saved=checkpoint.read(root/'full/step_000002.pt');restored=setup();restored.runtime.encode=lambda *a:torch.ones(4,8)
            train(restored,Corpus(root/'corpus'),root/'resumed',{'source':'test'},saved)
            for n,p in m.all_adapters().items():torch.testing.assert_close(p,restored.all_adapters()[n],rtol=0,atol=0)
            self.assertEqual(saved['stage'],'edit')
            self.assertEqual(set(saved['adapters']),set(m.all_adapters()))


if __name__=='__main__':unittest.main()
