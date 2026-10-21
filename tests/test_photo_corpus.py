"""Filtering preserves the original split and rejects broken/relative training pairs."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from PIL import Image
from qwen_latent_cot.correction.data import Corpus

spec=importlib.util.spec_from_file_location('photo_prepare',Path('scripts/prepare_photo_corpus.py'))
photo=importlib.util.module_from_spec(spec);spec.loader.exec_module(photo)


class PhotoCorpusTests(unittest.TestCase):
    def test_selection_decode_duplicates_and_split_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir();(source/'records').mkdir();(source/'views').mkdir()
            for name in ('good','shared'):Image.new('RGB',(128,128),'blue').save(source/f'{name}.png')
            (source/'bad.png').write_bytes(b'broken image')
            coco='unimoe2.5_coco_fixed';flickr='unimoe2.5_flickr30k_fixed'
            inputs=[('train',coco,'good','a blue object',{}),('train',coco,'good','a blue object',{}),
                    ('train',coco,'bad','a broken object',{}),('train',coco,'good','as before',{}),
                    ('train',coco,'good','an edit',{'edit_instruction':'add'}),
                    ('train',coco,'shared','a shared photo',{}),('val',flickr,'shared','a shared photo',{}),
                    ('val',flickr,'good','a validation photo',{})]
            # good is deliberately also cross-split; add one separate eligible train/val image.
            for name in ('train','val'):Image.new('RGB',(128,128),'red').save(source/f'{name}.png')
            inputs += [('train',coco,'train','a red photo',{}),('val',flickr,'val','another red photo',{})]
            views={}
            with (source/'records/rows.jsonl').open('wb') as stream:
                for i,(split,src,image,caption,extra) in enumerate(inputs):
                    offset=stream.tell();row=dict(record_id=str(i),family_id=image,source=src,split=split,
                        target_caption=caption,target_image=str(source/f'{image}.png'),original={'path':str(source/'origin.jsonl')},**extra)
                    stream.write((json.dumps(row)+'\n').encode());views.setdefault((split,src),[]).append(struct.pack('<IQB',0,offset,2))
            infos=[]
            for i,((split,src),refs) in enumerate(views.items()):
                name=f'{i}.idx';(source/'views'/name).write_bytes(b''.join(refs));infos.append(dict(path=name,task='generation',source=src,split=split,count=len(refs)))
            (source/'manifest.json').write_text(json.dumps(dict(format='annotation_index_v1',fingerprint='fixture',registry={'roots':{}},shards=[dict(path='rows.jsonl')],views=infos)))
            report=photo.build(source,root/'filtered',workers=2)
            self.assertEqual(report['removed']['duplicate_pair'],1)
            self.assertEqual(report['removed']['relative_caption'],1)
            self.assertEqual(report['removed']['edit_condition'],1)
            self.assertEqual(report['removed']['unreadable_image'],1)
            self.assertEqual(report['removed']['cross_split_conflict'],2)
            corpus=Corpus(root/'filtered')
            self.assertEqual(sum(len(v) for v in corpus.groups('train')),1)
            self.assertEqual(sum(len(v) for v in corpus.groups('val')),3);corpus.close()


if __name__=='__main__':unittest.main()
