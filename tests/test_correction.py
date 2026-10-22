"""Real tiny native MoT: spatial correction, gradients, CFG, resume, and isolation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import torch
from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import Qwen2Config,Qwen2ForCausalLM
from qwen_latent_cot.bagel.modeling.bagel.bagel import Bagel,BagelConfig
from qwen_latent_cot.correction.runtime import Runtime
from qwen_latent_cot.correction.model import SpatialCorrector,proposal_from,correction_loss
from qwen_latent_cot.correction.sampling import sample,guided
from qwen_latent_cot.correction.training import FrozenGuard
from qwen_latent_cot.correction import checkpoint


def tiny(checkpointing=False,post_cfg=False):
    torch.manual_seed(7)
    cfg=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
        num_attention_heads=4,num_key_value_heads=2,layer_module='Qwen2MoTDecoderLayer')
    lm=Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval().requires_grad_(False)
    model=Bagel(lm,None,BagelConfig(visual_und=False,llm_config=cfg,
        vae_config=SimpleNamespace(downsample=8,z_channels=2),max_latent_size=4)).to(torch.bfloat16).eval()
    torch.nn.init.normal_(model.llm2vae.weight,std=.01)
    class Tokenizer:
        def encode(self,text,add_special_tokens=True):return [4,5,6,7] if text else []
    class VAE(torch.nn.Module):
        def __init__(self):super().__init__();self.dummy=torch.nn.Parameter(torch.zeros(1,dtype=torch.bfloat16))
        def decode(self,x):return torch.nn.functional.interpolate(torch.cat((x,x[:,:1]),1),scale_factor=8)
    bundle=SimpleNamespace(model=model,vae=VAE(),tokenizer=Tokenizer(),token_ids=dict(bos_token_id=1,eos_token_id=2,start_of_image=1,end_of_image=3))
    config=json.loads(Path('configs/correction.json').read_text());config.update(resolution=32,sampler_steps=4,lora_rank=4,activation_checkpointing=checkpointing)
    if post_cfg:config['correction_space']='post_cfg'
    runtime=Runtime(bundle,config)
    return SpatialCorrector(runtime,config)


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model=tiny();self.runtime=self.model.runtime;self.flow=self.runtime.flow('a red cube')
        n=len(self.flow['packed_vae_token_indexes']);self.x=torch.randn(n,8);self.t=.8
        self.base,self.context=self.runtime.predict(self.x,self.t,self.flow)
        self.proposal=proposal_from(self.x,self.t,self.base)
    def test_initial_native_and_spatial_shape(self):
        delta=self.model(self.proposal,self.t,self.flow,self.context)
        self.assertEqual(delta.shape,self.base.shape);self.assertEqual(delta.count_nonzero(),0)
        self.assertTrue(all(not c.keys.requires_grad and not c.values.requires_grad for c in self.context))
        expected=len(self.flow['past_key_values'].key_cache[0])+len(self.x)
        self.assertTrue(all(len(c.keys)==expected for c in self.context))
    def test_only_head_first_then_und_and_input(self):
        guard=FrozenGuard(self.runtime.model,self.runtime.bundle.vae,self.model)
        before=[p.clone() for p in self.runtime.model.parameters()]
        optimizer=torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],lr=.01)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            delta=self.model(self.proposal,self.t,self.flow,self.context)
            correction_loss(delta,self.base,torch.ones_like(delta),torch.zeros_like(delta)).backward()
            norms={k:sum(float(p.grad.abs().sum()) for p in ps) for k,ps in self.model.group_parameters().items()}
            self.assertGreater(norms['head'],0)
            if step==0:self.assertEqual(norms['und'],0);self.assertEqual(norms['input'],0)
            else:self.assertGreater(norms['und'],0);self.assertGreater(norms['input'],0)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.trainable().values()))
            optimizer.step();guard.check()
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,self.runtime.model.parameters())))
        for a,b in zip(self.runtime.decoder.layers,self.model.layers):
            self.assertNotEqual(a.self_attn.q_proj.weight.data_ptr(),b.self_attn.q_proj.base.weight.data_ptr())
    def test_exact_flow_loss_identity_and_detached_base(self):
        delta=torch.randn_like(self.base,requires_grad=True);base=self.base.clone().requires_grad_()
        x0=torch.randn_like(delta);eps=torch.randn_like(delta)
        actual=correction_loss(delta,base,x0,eps)
        expected=((base.detach().float()+delta.float())-(eps.float()-x0.float())).square().mean()
        torch.testing.assert_close(actual,expected);actual.backward();self.assertIsNone(base.grad)
    def test_checkpointing_matches_forward_and_backward(self):
        with torch.no_grad():self.model.head.weight.normal_(std=.01)
        other=tiny(True);other.load_state_dict(self.model.state_dict())
        a=self.model(self.proposal,self.t,self.flow,self.context);b=other(self.proposal,self.t,self.flow,self.context)
        torch.testing.assert_close(a,b,rtol=0,atol=0);a.square().mean().backward();b.square().mean().backward()
        for name,p in self.model.trainable().items():torch.testing.assert_close(p.grad,other.trainable()[name].grad,rtol=0,atol=0)
    def test_sampling_zero_head_and_native_cfg(self):
        original,trace=sample(self.model,'a red cube',17,native=True)
        corrected,other=sample(self.model,'a red cube',17)
        self.assertEqual(original.tobytes(),corrected.tobytes())
        self.assertEqual(sum(n['und_passes'] for n in other),2)
        self.assertTrue(all(n['gen_passes']==2 for n in other))
        self.assertEqual(len(trace),3)
        conditional=torch.randn_like(self.x);reference=torch.randn_like(self.x)
        result=guided(conditional,reference,self.model.config)
        self.assertLessEqual(float(result.norm()),float(conditional.norm())+1e-5)
    def test_native_cfg_matches_original_bagel(self):
        with torch.autocast('cpu',dtype=torch.bfloat16):
            native=self.runtime.model._forward_flow(x_t=self.x,timestep=torch.full((len(self.x),),self.t),**self.flow,
            cfg_text_scale=4.,cfg_img_scale=1.,cfg_renorm_min=0.,cfg_renorm_type='global')
        reference=self.runtime.reference(self.x,self.t,self.flow)
        actual=guided(self.base,reference,self.model.config)
        torch.testing.assert_close(native,actual,rtol=0,atol=0)
    def test_feature_context_and_prompt_cache_immutable(self):
        before={i:k.clone() for i,k in self.flow['past_key_values'].key_cache.items()}
        with torch.no_grad():self.model.head.weight.normal_(std=.01)
        a=self.model(self.proposal,self.t,self.flow,self.context)
        changed=copy.deepcopy(self.context);changed[-1].values.add_(2)
        b=self.model(self.proposal,self.t,self.flow,changed)
        self.assertFalse(torch.equal(a,b))
        for i,value in before.items():self.assertTrue(torch.equal(value,self.flow['past_key_values'].key_cache[i]))
    def test_trainable_checkpoint_roundtrip_and_legacy_rejection(self):
        class Sampler:
            def state_dict(self):return {'cursor':2}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt';optimizer=torch.optim.AdamW(self.model.trainable().values())
            with torch.no_grad():self.model.head.bias.fill_(.25)
            checkpoint.save(path,self.model,2,optimizer,Sampler(),{'native':'test'},[checkpoint.rng_state(torch.device('cpu'))])
            saved=checkpoint.read(path);self.assertEqual(set(saved['trainable']),set(self.model.trainable()))
            with torch.no_grad():self.model.head.bias.zero_()
            checkpoint.apply(saved,self.model);self.assertTrue(torch.all(self.model.head.bias==.25))
            torch.save({'kind':'bagel_canvas_stage_a_v2_1'},path)
            with self.assertRaises(ValueError):checkpoint.read(path)


if __name__=='__main__':unittest.main()
