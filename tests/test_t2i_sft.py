"""Native MoT parity, GEN-only gradients, grouped data, exact resume and EMA export."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import zlib
import torch
from PIL import Image
from test_correction import tiny
from qwen_latent_cot.t2i.model import NativeT2I,sequence,gen_parameter
from qwen_latent_cot.t2i.data import build,Pairs,Sampler,dimensions
from qwen_latent_cot.t2i.training import train
from qwen_latent_cot.t2i import checkpoint as ck
from qwen_latent_cot.t2i.sampling import load_overlay,sample


def cfg():
    c=json.loads(Path('configs/t2i_sft.json').read_text())
    c.update(steps=2,global_batch=2,min_short_side=32,max_long_side=64,max_sequence_length=128,
             save_every=2,validate_every=2,val_images=1,activation_checkpointing=True)
    return c


def bundle():
    b=tiny().runtime.bundle;b.model_path='/tiny-native'
    def encode(x):return torch.nn.functional.avg_pool2d(x[:,:2],8)
    b.vae.encode=encode
    return b


def fixture(root,tokenizer,config):
    rows=[];original=[]
    def prompt(bucket):
        for i in range(10000):
            s=f'original {bucket} {i}';v=zlib.crc32(('17\n'+s).encode())%10000
            if (100<=v<200) if bucket=='val' else (v>=200):return s
        raise AssertionError('no split fixture')
    train_prompt=prompt('train');val_prompt=prompt('val')
    source=root/'source.jsonl'
    for i,p in enumerate([train_prompt,train_prompt,val_prompt]):
        image=root/f'{i}.png';Image.new('RGB',(64,64),(30+i*20,100,160)).save(image)
        original.append(dict(uid=str(i),raw_prompt=p))
        rows.append(dict(uid=str(i),image=str(image),prompt=f'expanded caption {i}',source_file=str(source),source_line=i+1,benchmark='fixture'))
    source.write_text(''.join(json.dumps(r)+'\n' for r in original))
    pairs=root/'input.jsonl';pairs.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    output=root/'corpus';build(pairs,output,tokenizer,config)
    return output


class T2ITests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_recipe_and_size(self):
        c=json.loads(Path('configs/t2i_sft.json').read_text())
        self.assertEqual((c['lr'],c['steps'],c['warmup_steps'],c['ema_decay']),(2.5e-5,15000,500,.995))
        self.assertEqual(dimensions(1248,832),(1024,672))
        with self.assertRaises(ValueError):dimensions(3200,640)
    def test_original_forward_parity_and_gen_gradients(self):
        b=bundle();m=NativeT2I(b.model,False).float();b.model.timestep_shift=4.
        inputs=sequence(b.model,b.tokenizer,b.token_ids,'red cube',64,64,'cpu',128)
        latent=torch.randn(2,8,8);logit=torch.tensor(.3)
        before={n:p.clone() for n,p in b.model.named_parameters() if not gen_parameter(n)}
        torch.manual_seed(31)
        with torch.autocast('cpu',dtype=torch.bfloat16):loss,pred,t=m(latent,inputs,logit)
        reference=copy.deepcopy(b.model).bfloat16()
        torch.manual_seed(31)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            native=reference(sequence_length=len(inputs['positions']),packed_text_ids=inputs['ids'],packed_text_indexes=inputs['text_indexes'],
                sample_lens=[len(inputs['positions'])],packed_position_ids=inputs['positions'],nested_attention_masks=[inputs['mask']],
                padded_latent=latent[None],patchified_vae_latent_shapes=[(4,4)],packed_latent_position_ids=inputs['image_positions'],
                packed_vae_token_indexes=inputs['image_indexes'],packed_timesteps=logit.expand(16),mse_loss_indexes=inputs['image_indexes'])
        # The wrapper performs FP32 target subtraction; original may autocast intermediates.
        torch.testing.assert_close(loss,native['mse'].float().mean(),rtol=.01,atol=1e-5)
        self.assertGreater(float(t),.5)
        loss.backward();optimizer=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3);optimizer.step()
        self.assertTrue(all(torch.equal(before[n],p) for n,p in b.model.named_parameters() if n in before))
        self.assertTrue(all(p.grad is not None for n,p in b.model.named_parameters() if gen_parameter(n)))
        # Prompt cannot attend to the image; image can attend to prompt and all image patches.
        self.assertTrue(torch.isneginf(inputs['mask'][0,inputs['image_indexes'][0]]))
        self.assertEqual(float(inputs['mask'][inputs['image_indexes'][0],-1]),0.)
    def test_checkpoint_recompute_and_no_truncation(self):
        b=bundle();m=NativeT2I(b.model,False).float();other=copy.deepcopy(m);other.activation_checkpointing=True
        inputs=sequence(b.model,b.tokenizer,b.token_ids,'test',32,32,'cpu',128);latent=torch.randn(2,4,4);noise=torch.randn(4,8)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            a=m(latent,inputs,torch.tensor(.2),noise)[0];b_loss=other(latent,inputs,torch.tensor(.2),noise)[0]
        a.backward();b_loss.backward();torch.testing.assert_close(a,b_loss)
        for p,q in zip(m.parameters(),other.parameters()):
            if p.requires_grad:torch.testing.assert_close(p.grad,q.grad)
        with self.assertRaises(ValueError):sequence(m.model,b.tokenizer,b.token_ids,'test',64,64,'cpu',10)
    def test_data_and_exact_resume_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c=cfg();b=bundle();corpus=fixture(root,b.tokenizer,c)
            data=Pairs(corpus);self.assertEqual(len(data.ids('train')),2);self.assertEqual(len(data.ids('val')),1)
            self.assertEqual(len({data.row(i)['family'] for i in data.ids('train')}),1);data.close()
            first=root/'first';train(b,corpus,first,c,torch.device('cpu'))
            c3=dict(c,steps=3);train(bundle(),corpus,first,c3,torch.device('cpu'),first/'step_000002')
            uninterrupted=root/'full';train(bundle(),corpus,uninterrupted,c3,torch.device('cpu'))
            a=torch.load(first/'step_000003/rank_00000.pt',weights_only=False);z=torch.load(uninterrupted/'step_000003/rank_00000.pt',weights_only=False)
            for n in a['weights']:torch.testing.assert_close(a['weights'][n],z['weights'][n],rtol=0,atol=0)
            for n in a['ema']:torch.testing.assert_close(a['ema'][n],z['ema'][n],rtol=0,atol=0)
            exported=ck.export(first/'step_000003',root/'export');new=bundle();load_overlay(new,exported)
            self.assertEqual(sample(new,'red cube',17,32,32,steps=3).size,(32,32))
            for n,p in new.model.named_parameters():
                if gen_parameter(n):torch.testing.assert_close(p,a['ema']['model.'+n].to(p.dtype),rtol=0,atol=0)
            bad=dict(c3,lr=1e-3)
            with self.assertRaises(ValueError):train(bundle(),corpus,first,bad,torch.device('cpu'),first/'step_000003')
    def test_sampler_resume(self):
        s=Sampler(list(range(11)),17);first=s.take(11);self.assertEqual(len(set(first)),11)
        saved=s.state_dict();expected=s.take(15);other=Sampler(list(range(11)),17);other.load_state_dict(saved)
        self.assertEqual(expected,other.take(15))


if __name__=='__main__':unittest.main()
