"""Existing corpus schema, family split preservation, and training continuation."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from PIL import Image
import torch
from test_correction import tiny
from qwen_latent_cot.correction.data import Corpus,Sampler
from qwen_latent_cot.correction.training import train
from qwen_latent_cot.correction import checkpoint


def fixture(root):
    root=Path(root);(root/'records').mkdir();(root/'views').mkdir()
    references=[]
    with (root/'records/rows.jsonl').open('wb') as stream:
        for i in range(4):
            image=root/f'{i}.png';Image.new('RGB',(32,32),(i*50,70,120)).save(image)
            references.append(struct.pack('<IQB',0,stream.tell(),2))
            value=dict(record_id=str(i),family_id=f'family{i}',source='fixture',split='train' if i<3 else 'val',
                target_caption=f'a full caption {i}',target_image=str(image),original={'path':str(root/'original.jsonl')})
            stream.write((json.dumps(value)+'\n').encode())
    (root/'views/train.idx').write_bytes(b''.join(references[:3]));(root/'views/val.idx').write_bytes(references[3])
    (root/'manifest.json').write_text(json.dumps(dict(format='annotation_index_v1',fingerprint='fixture',registry={'roots':{}},
        shards=[{'path':'rows.jsonl'}],views=[dict(path='train.idx',task='generation',split='train',count=3),dict(path='val.idx',task='generation',split='val',count=1)])))


class DataTests(unittest.TestCase):
    def test_original_split_and_sampler_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture(directory);corpus=Corpus(directory);sampler=Sampler(corpus,17)
            self.assertEqual(corpus.validation(1)[0]['family_id'],'family3')
            first=sampler.take(3);self.assertEqual(len({r['family_id'] for r in first}),3)
            saved=sampler.state_dict();expected=sampler.take(7)
            other=Sampler(corpus,17);other.load_state_dict(saved)
            self.assertEqual(expected,other.take(7));self.assertTrue(all(r['split']=='train' and r['caption'].startswith('a full caption') for r in expected))
            corpus.close()
    def test_training_resume_matches_uninterrupted(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root)
            def setup():
                model=tiny();model.config.update(global_batch=2,steps=3,save_steps=[2,3],val_images=0,warmup_steps=0)
                model.runtime.config.update(model.config)
                model.runtime.bundle.vae.encode=lambda x:torch.nn.functional.avg_pool2d(x[:,:2],8)
                return model
            model=setup();train(model,Corpus(root),root/'whole',{'test':'same'})
            expected={k:p.detach().clone() for k,p in model.trainable().items()}
            saved=checkpoint.read(root/'whole/step_000002.pt')
            resumed=setup();train(resumed,Corpus(root),root/'resumed',{'test':'same'},saved=saved)
            for name,value in expected.items():torch.testing.assert_close(value,resumed.trainable()[name],rtol=0,atol=0)
            actual=json.loads((root/'resumed/train.jsonl').read_text().splitlines()[-1])
            reference=json.loads((root/'whole/train.jsonl').read_text().splitlines()[-1])
            self.assertEqual(actual['loss'],reference['loss'])


if __name__=='__main__':unittest.main()
