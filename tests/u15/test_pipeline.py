"""Real upstream tiny architecture on CPU/NPU; no release-weight quality claims."""
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image
from qwen_latent_cot.u15.runtime import upstream,Runtime
from qwen_latent_cot.u15.readout import Corrector
from qwen_latent_cot.u15.objective import project,residual_target
from qwen_latent_cot.u15.sampling import sample
from qwen_latent_cot.u15.training import train
from qwen_latent_cot.u15 import checkpoint
from qwen_latent_cot.bagel.accelerator import set_device

SOURCE=os.environ.get('U15_SOURCE')
DEVICE=os.environ.get('U15_TEST_DEVICE','cpu')


class Tokenizer:
    eos_token_id=2
    def encode(self,text,add_special_tokens=False):return [3+ord(c)%60 for c in text[:40]]
    def __call__(self,text,return_tensors='pt'):return {'input_ids':torch.tensor([self.encode(text)])}
    def decode(self,ids,skip_special_tokens=True):return ' '.join(map(str,ids))


def tiny():
    native=upstream(SOURCE)
    from sensenova_u1.models.neo_unify.configuration_neo_chat import NEOLLMConfig
    llm=NEOLLMConfig(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=512,
        bos_token_id=1,eos_token_id=2,pad_token_id=0,tie_word_embeddings=False)
    vision=native.NEOVisionConfig(patch_size=16,hidden_size=8,llm_hidden_size=16,downsample_ratio=.5)
    cfg=native.NEOChatConfig(vision_config=vision,llm_config=llm,template='neo1_0',
        downsample_ratio=.5,fm_head_layers=2,use_pixel_head=True,concat_time_token_num=0,
        noise_scale=.5,noise_scale_mode='constant',noise_scale_base_image_seq_len=256,
        add_noise_scale_embedding=True,noise_scale_max_value=2.,time_schedule='standard',
        time_shift_type='linear',base_shift=.5,max_shift=1.15,base_image_seq_len=256,max_image_seq_len=4096)
    torch.manual_seed(9);device=set_device(DEVICE)
    model=native.NEOChatModel(cfg).to(device=device,dtype=torch.float32 if device.type=='cpu' else torch.bfloat16)
    settings=json.loads((Path(__file__).resolve().parents[2]/'configs/u15/correction.json').read_text())
    settings.update(width=64,height=64,num_steps=4,active_steps=2,lora_rank=2,head_channels=8,
        readout_heads=2,steps=2,global_batch=2,save_every=1,validate_every=1,val_images=1,warmup_steps=0)
    rt=Runtime(model,Tokenizer(),settings);rt.identity=dict(model='synthetic-tiny',revision='c35c83b')
    return rt,settings


class Algebra(unittest.TestCase):
    def test_per_image_projection_and_target(self):
        a=torch.tensor([[-2.,1.],[3.,2.]]).reshape(2,1,1,2);g=torch.tensor([[1.,0.],[-1.,0.]]).reshape_as(a)
        result=project(a,g)
        torch.testing.assert_close(result,torch.tensor([[0.,1.],[0.,2.]]).reshape_as(a))
        torch.testing.assert_close(project(a,torch.zeros_like(g)),a)
        clean=torch.ones_like(a);noise=torch.zeros_like(a)
        torch.testing.assert_close(residual_target(clean,noise,.5,a,a+g),project(clean-a,g))


class DataFilter(unittest.TestCase):
    def test_excludes_original_prompt_family_and_preserves_default(self):
        import zlib
        from qwen_latent_cot.t2i.data import build,Pairs
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.jsonl';pairs=root/'pairs.jsonl'
            val=next('family '+str(i) for i in range(10000) if 100<=zlib.crc32(('17\nfamily '+str(i)).encode())%10000<200)
            train_name=next('family '+str(i) for i in range(10000) if zlib.crc32(('17\nfamily '+str(i)).encode())%10000>=200)
            rows=[]
            for i,name in enumerate((train_name,val,'excluded original')):
                path=root/f'{i}.png';Image.new('RGB',(64,64),'red').save(path)
                rows.append(dict(uid=str(i),prompt='rewritten '+name,raw_prompt=name,image=str(path),source_file=str(source),source_line=i+1,benchmark='natural'))
            text=''.join(json.dumps(x)+'\n' for x in rows);source.write_text(text);pairs.write_text(text)
            cfg=dict(min_short_side=64,max_long_side=64,max_sequence_length=40960)
            manifest=build(pairs,root/'filtered',Tokenizer(),cfg,excluded_prompts=['excluded original'])
            self.assertEqual(manifest['removed']['evaluation_prompt_family'],1)
            original=build(pairs,root/'original',Tokenizer(),cfg)
            self.assertEqual(sum(original['counts'].values()),3)
            self.assertEqual(sum(manifest['counts'].values()),2)


@unittest.skipUnless(SOURCE,'Set U15_SOURCE to the pinned upstream checkout')
class Pipeline(unittest.TestCase):
    def test_native_zero_correction_and_reader_backward(self):
        rt,cfg=tiny();prefix=rt.prefix('a red ball');t=rt.times()[1]
        z=torch.zeros(1,3,64,64,device=rt.device,dtype=next(rt.model.parameters()).dtype)
        with torch.no_grad():p=rt.predict(z,t,prefix,capture=True)
        self.assertEqual(tuple(p.observation[-1].shape),(1,4,16))
        self.assertEqual(tuple(p.clean_prediction.shape),(1,3,64,64))
        corrector=Corrector(rt,cfg);corrector.initialize_global(cfg['active_steps'])
        loss=corrector.reader.caption_loss(p.observation,t,'red ball');loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(corrector.reader.input.weight.grad.abs().sum()),0)
        self.assertTrue(all(x.grad is None for x in rt.model.parameters()))
        goal=prefix.hidden
        correction=corrector(p.observation,t,goal)
        self.assertEqual(tuple(correction.shape),(1,3,64,64));self.assertEqual(float(correction.abs().sum()),0)
        a,trace=sample(rt,'a red ball',7,corrector,'corrected')
        b,_=sample(rt,'a red ball',7)
        np.testing.assert_array_equal(np.array(a),np.array(b));self.assertEqual(sum(x['und_passes'] for x in trace),2)
        with torch.no_grad():native=rt.model.t2i_generate(rt.tokenizer,'a red ball',cfg_scale=cfg['cfg_scale'],
            timestep_shift=cfg['timestep_shift'],cfg_norm=cfg['cfg_norm'],image_size=(64,64),num_steps=4,
            t_eps=cfg['t_eps'],seed=7)
        native=((native[0].float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy()
        np.testing.assert_array_equal(np.array(b),native)
        # Teacher answer cannot change the image-prefix states.
        ids=torch.tensor([[3,4,5]],device=rt.device)
        first,_=corrector.reader(p.observation,t,ids);second,_=corrector.reader(p.observation,t,ids+1)
        torch.testing.assert_close(first,second)

    def test_native_release_loader_roundtrip(self):
        from tokenizers import Tokenizer as Backend
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        rt,cfg=tiny();rt.model.cpu().float();rt.device=torch.device('cpu')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rt.model.save_pretrained(root)
            # Release config schema uses scalar patchifier fields, as HF U1.5 does.
            config=json.loads((root/'config.json').read_text())
            config['vision_config']['llm_hidden_size']=16;config['vision_config']['downsample_ratio']=.5
            config['llm_config']['architectures']=['Qwen3ForCausalLM']
            (root/'config.json').write_text(json.dumps(config))
            tok=PreTrainedTokenizerFast(tokenizer_object=Backend(WordLevel({'[UNK]':0,'[BOS]':1,'[EOS]':2},unk_token='[UNK]')),unk_token='[UNK]',eos_token='[EOS]')
            tok.save_pretrained(root)
            loaded=Runtime.load(root,SOURCE,cfg,'cpu')
            for n,p in rt.model.named_parameters():torch.testing.assert_close(p,dict(loaded.model.named_parameters())[n])
            # Compare deterministic model computation, including nonpersistent RoPE.
            rt.tokenizer=loaded.tokenizer
            x=torch.randn(1,3,64,64)
            with torch.no_grad():
                a=rt.predict(x,rt.times()[1],rt.prefix('red ball')).velocity
                b=loaded.predict(x,loaded.times()[1],loaded.prefix('red ball')).velocity
            torch.testing.assert_close(a,b)

    def test_train_export_resume_all_stages(self):
        rt,cfg=tiny()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);corpus=root/'corpus';corpus.mkdir()
            Image.new('RGB',(64,64),'red').save(corpus/'image.png')
            db=sqlite3.connect(corpus/'pairs.sqlite')
            db.execute('CREATE TABLE pairs(id INTEGER PRIMARY KEY,uid TEXT,image TEXT,prompt TEXT,width INTEGER,height INTEGER,split TEXT)')
            for i in range(3):db.execute('INSERT INTO pairs VALUES(?,?,?,?,?,?,?)',(i,str(i),str(corpus/'image.png'),'red ball',64,64,'val' if i==2 else 'train'))
            db.commit();db.close();(corpus/'manifest.json').write_text(json.dumps(dict(format='bagel_t2i_pairs_v1',identity='test')))
            base={n:p.detach().cpu().clone() for n,p in rt.model.named_parameters()}
            train(rt,cfg,'caption',corpus,root/'caption')
            checkpoint.export(root/'caption/step_000002',root/'reader.pt')
            for n,p in rt.model.named_parameters():torch.testing.assert_close(p.cpu(),base[n])
            from qwen_latent_cot.t2i.checkpoint import rng,restore_rng
            initial=rng(rt.device)
            train(rt,cfg,'correct',corpus,root/'correct',root/'reader.pt')
            result=checkpoint.export(root/'correct/step_000002',root/'corrector.pt')
            c=Corrector(rt,cfg);c.initialize_global(2);checkpoint.load_portable(root/'corrector.pt',c,rt,'correct')
            self.assertTrue(bool((c.global_counts>0).all()));self.assertEqual(result['step'],2)
            # Exact resume reproduces the next update including sampler and RNG.
            extended=dict(cfg,steps=3)
            train(rt,extended,'correct',corpus,root/'correct',root/'reader.pt',root/'correct/step_000002')
            self.assertTrue((root/'correct/step_000003/manifest.json').exists())
            resumed=torch.load(root/'correct/step_000003/rank_00000.pt',map_location='cpu',weights_only=False)
            restore_rng(initial,rt.device)
            train(rt,extended,'correct',corpus,root/'uninterrupted',root/'reader.pt')
            uninterrupted=torch.load(root/'uninterrupted/step_000003/rank_00000.pt',map_location='cpu',weights_only=False)
            for name,value in resumed['weights'].items():
                torch.testing.assert_close(value,uninterrupted['weights'][name],rtol=1e-4,atol=1e-5)
            # Native RGB head and GEN expert train, original UND remains frozen.
            rt.model.cpu();rt.device=torch.device('cpu')
            train(rt,cfg,'t2i',corpus,root/'sft')
            result=checkpoint.export(root/'sft/step_000002',root/'gen.pt')
            self.assertTrue(any('fm_head' in k for k in result['weights']))
            self.assertTrue(all('_mot_gen' in k or k.startswith('fm_modules.') for k in result['weights']))


if __name__=='__main__':unittest.main()
