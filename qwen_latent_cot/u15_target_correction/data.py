"""Tier A paired images, Tier B real histories and separately authored edit targets."""
import json,random
from pathlib import Path
import torch
from .images import load_pair
from .field import protocol
from . import TRACE_KIND


class Corpus:
    def __init__(self,path,rt,cfg):
        self.path=Path(path).resolve();self.rt=rt;self.cfg=cfg
        self.manifest=json.loads(self.path.read_text())
        self.rows=self.manifest['records']
        families={}
        for row in self.rows:
            if row['tier'] not in ('A','B','zero') or not row['approved']:
                raise ValueError('Expected approved A/B/zero record')
            if families.setdefault(row['family'],row['split'])!=row['split']:
                raise ValueError('Prompt family crosses splits')
    def row(self,i,seed):
        row=self.rows[i];rt=self.rt;cfg=self.cfg
        target=str(self.path.parent/row['target_image']) if row['tier']!='zero' else None
        generator=torch.Generator(device=rt.device).manual_seed(seed)
        if row['tier']=='A':
            source,target,valid,_=load_pair(dict(source_image=str(self.path.parent/row['source_image']),target_image=target),rt.width,rt.height,rt.device)
            tau=cfg['deep_time'] if seed%2==0 else cfg['shallow_time']
            noise=torch.randn(source.shape,device=rt.device,generator=generator)*rt.noise_scale()
            anchor=tau*source+(1-tau)*noise
            preview=source
            condition=row['raw_prompt']+'\n[Semantic Corrections]:\n'+row['instruction']
        else:
            trace=torch.load(self.path.parent/row['trace'],map_location='cpu',weights_only=True)
            if trace['kind']!=TRACE_KIND or trace['record']['protocol']!=protocol(rt,cfg):
                raise ValueError('Tier B trajectory protocol differs')
            if trace['record']['raw_prompt']!=row['raw_prompt']:
                raise ValueError('Target prompt differs from trace')
            tau=cfg[row['arm']+'_time']
            anchor=trace['history'][str(tau)].to(rt.device)
            preview=trace['preview'].to(rt.device)
            source=trace['final'].to(rt.device)
            # Teacher edits the native final, never the preview. No resizing of real traces.
            from PIL import Image
            import numpy as np
            if row['tier']=='zero':
                target=source
            else:
                with Image.open(target) as image:
                    if image.size!=(rt.width,rt.height):raise ValueError('Tier B target must use native canvas dimensions')
                    target=torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1)[None].to(rt.device).float()/127.5-1
            valid=trace['valid'].to(rt.device)
            condition=trace['record']['condition']
        return dict(reference=source,target=target,valid=valid,anchor=anchor,preview=preview,tau=tau,
                    raw_prompt=row['raw_prompt'],condition=condition,zero=row['tier']=='zero',tier=row['tier'])


class Sampler:
    def __init__(self,corpus,cfg):
        self.rng=random.Random(cfg['seed'])
        self.groups={k:[i for i,r in enumerate(corpus.rows) if r['tier']==k and r['split']=='train'] for k in cfg['tier_weights']}
        self.weights=cfg['tier_weights']
        if any(not self.groups[k] for k,w in self.weights.items() if w>0):
            raise ValueError('Missing required training tier; no implicit replacement')
    def take(self,n):
        kinds=self.rng.choices(list(self.weights),weights=list(self.weights.values()),k=n)
        return [(self.rng.choice(self.groups[k]),self.rng.randrange(2**31),(self.rng.randrange(65536)+.5)/65536) for k in kinds]
    def state_dict(self):return dict(rng=self.rng.getstate())
    def load_state_dict(self,state):self.rng.setstate(state['rng'])
