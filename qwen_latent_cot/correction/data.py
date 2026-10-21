"""Natural caption/image pairs from existing generation indices and family splits."""
from collections import OrderedDict
import json
import math
import mmap
import os
from pathlib import Path
import random
import struct

REF = struct.Struct('<IQB')


class View:
    def __init__(self, corpus, info):
        self.corpus, self.info = corpus, info
        self.stream = (corpus.root/'views'/info['path']).open('rb')
        self.index = mmap.mmap(self.stream.fileno(),0,access=mmap.ACCESS_READ)
    def __len__(self): return self.info['count']
    def __getitem__(self,index):
        if not 0<=index<len(self): raise IndexError(index)
        shard,offset,role = REF.unpack_from(self.index,index*REF.size)
        if role!=2: raise ValueError('Correction requires a paired generation record')
        value = self.corpus.record(shard,offset)
        caption = value['target_caption'].strip()
        image = str(value['target_image'])
        for old,new in sorted(self.corpus.manifest['registry'].get('roots',{}).items(),key=lambda p:-len(p[0])):
            if image.startswith(old): image=str(new)+image[len(old):];break
        image = os.path.abspath(os.path.join(os.path.dirname(value['original']['path']),image))
        if not caption or not value['target_image']: raise ValueError('Missing caption/image')
        return dict(caption=caption,image=image,sample_id=value['record_id'],family_id=value['family_id'],
                    source=value['source'],split=value['split'])
    def close(self): self.index.close();self.stream.close()


class Corpus:
    def __init__(self,root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if self.manifest['format'] not in ('annotation_index_v1','annotation_index_v2'):
            raise ValueError('Unsupported paired-generation corpus')
        self.views = [View(self,v) for v in self.manifest['views'] if v['task']=='generation' and v['count']]
        self.handles = OrderedDict()
    def record(self,shard,offset):
        if shard not in self.handles:
            self.handles[shard]=(self.root/'records'/self.manifest['shards'][shard]['path']).open('rb')
            if len(self.handles)>8:self.handles.popitem(last=False)[1].close()
        handle=self.handles[shard];self.handles.move_to_end(shard);handle.seek(offset)
        return json.loads(handle.readline())
    def groups(self,split): return [v for v in self.views if v.info['split']==split]
    def validation(self,count):
        result=[];families=set();views=self.groups('val')
        for position in range(count):
            for view in views:
                if position>=len(view):continue
                row=view[position]
                if row['family_id'] in families:continue
                result.append(row);families.add(row['family_id'])
                if len(result)==count:return result
        if len(result)<count:raise ValueError('Insufficient distinct validation families')
        return result
    def close(self):
        for v in self.views:v.close()
        for h in self.handles.values():h.close()


class Sampler:
    """Sqrt(source-size) coverage and constant-size, exactly resumable affine shuffles."""
    def __init__(self,corpus,seed):
        self.corpus=corpus;self.views=corpus.groups('train');self.rng=random.Random(seed)
        if not self.views:raise ValueError('No train generation views')
        self.cursors=[0]*len(self.views);self.epochs=[0]*len(self.views)
        self.transforms=[self.permutation(len(v)) for v in self.views]
    def permutation(self,n):
        a=self.rng.randrange(1,max(2,n))
        while math.gcd(a,n)!=1:a=(a+1)%n or 1
        return a,self.rng.randrange(n)
    def take(self,count):
        rows=[]
        for _ in range(count):
            i=self.rng.choices(range(len(self.views)),weights=[len(v)**.5 for v in self.views],k=1)[0]
            view=self.views[i];a,b=self.transforms[i]
            rows.append(view[(a*self.cursors[i]+b)%len(view)]);self.cursors[i]+=1
            if self.cursors[i]==len(view):
                self.cursors[i]=0;self.epochs[i]+=1;self.transforms[i]=self.permutation(len(view))
        return rows
    def state_dict(self):
        return dict(fingerprint=self.corpus.manifest['fingerprint'],views=[v.info['path'] for v in self.views],
                    rng=self.rng.getstate(),cursors=list(self.cursors),epochs=list(self.epochs),transforms=list(self.transforms))
    def load_state_dict(self,state):
        if state['fingerprint']!=self.corpus.manifest['fingerprint'] or state['views']!=[v.info['path'] for v in self.views]:
            raise ValueError('Resume corpus inventory differs')
        self.rng.setstate(state['rng']);self.cursors=state['cursors'];self.epochs=state['epochs'];self.transforms=state['transforms']
