"""Task/noise supervision policy; no learned gate and no synthetic high-noise truth."""
from dataclasses import dataclass
import random
from .schedule import native_times


@dataclass(frozen=True)
class Supervision:
    sem: bool = False
    teacher: bool = False
    flow: bool = False

    def asdict(self): return dict(sem=self.sem, teacher=self.teacher, flow=self.flow)


class NoisePolicy:
    def __init__(self, config, step=1):
        self.config, self.step = config, step
        selected = dict(observe_max_t=.25, endpoint_max_t=.25)
        for phase in config.get('noise_curriculum', []):
            if step >= phase['start_step']: selected.update(phase)
        calibrated=config.get('noise_calibration',dict(observe_max_t=.25,endpoint_max_t=.25))
        self.observe_max=min(selected['observe_max_t'],calibrated['observe_max_t'])
        self.endpoint_max=min(selected['endpoint_max_t'],calibrated['endpoint_max_t'])
        self.native=native_times(config)

    def masks(self, row, t, *, pass_index=0, fresh=False, teacher_available=False):
        task=row['task']; scope=row.get('annotation',{}).get('scope','endpoint')
        if task=='observe':
            valid=0<t<=self.observe_max if scope=='endpoint' else t<1
            return Supervision(sem=valid and bool(row.get('caption')))
        flow=bool(row.get('flow_supervision')) and not row.get('state')
        if task=='generation': return Supervision(flow=flow)
        endpoint=scope=='endpoint'
        valid=(0<t<=self.endpoint_max) if endpoint else (t<1 or self.config.get('pure_noise_trajectory',False))
        # A later pass is a different proposal even though x_t/t remain fixed.
        valid=valid and (pass_index==0 or fresh)
        feedback=row.get('feedback')
        text=bool(feedback and (feedback.get('satisfied') or feedback.get('correction')))
        return Supervision(sem=valid and text, teacher=valid and text and teacher_available, flow=flow)

    def time(self, row, rng):
        if row.get('state') or row.get('state_hash'): return float(row['t'])
        if row['task']=='observe': maximum=self.observe_max
        elif row['task']=='feedback': maximum=self.endpoint_max
        else:
            nodes=self.native[:-1]
            bands=[nodes[:20],nodes[20:35],nodes[35:]]
            bands=[b for b in bands if b]
            band=rng.choices(bands,weights=([.6,.2,.2] if len(bands)==3 else None),k=1)[0]
            return rng.choice(band)
        if row['task']=='feedback' and self.config.get('online_feedback'):
            nodes=[t for t in self.native[:-1] if 0<t<=maximum]
            if not nodes:raise ValueError('native grid has no node inside calibrated feedback range')
            return rng.choice(nodes)
        lower=.05
        if maximum<lower: raise ValueError('noise range has no valid observation/endpoint state')
        if maximum>.25 and rng.random()>=.6:return rng.uniform(.25,maximum)
        return rng.uniform(lower,min(.25,maximum))


class CoverageSampler:
    """Weighted source selection, O(number-of-views) resume state, bounded rows.

    Each view uses a coprime affine permutation: no full-corpus shuffle list.
    Endpoint A/B adjacent references form one episode with shared time/seed.
    """
    def __init__(self, corpus, config, stage):
        import math
        self.corpus,self.config,self.stage=corpus,config,stage
        requested={'observe':1.} if stage=='C2-observe' else dict(config['task_mix'])
        weights=config['loss_weights']
        for task in list(requested):
            active=weights['sem'] if task=='observe' else weights['sem']+weights['teacher'] if task in ('endpoint','trajectory') else weights['flow']
            if not active:requested[task]=0
        self.views=corpus.groups('train',{k:v for k,v in requested.items() if v>0})
        if not self.views:raise ValueError('no available train task views')
        self.by_task={}
        for i,view in enumerate(self.views):self.by_task.setdefault(view.info['task'],[]).append(i)
        if 'generation' in self.by_task:
            for task in ('trajectory','conditional'):
                if task not in self.by_task:requested['generation']+=requested.get(task,0.)
        self.weights={task:requested[task] for task in self.by_task if requested[task]>0}
        total=sum(self.weights.values());self.weights={k:v/total for k,v in self.weights.items()}
        self.task_counts={k:0 for k in self.weights}
        if not self.weights:raise ValueError('no enabled task views')
        self.rng=random.Random(config['seed']);self.cursors=[0]*len(self.views);self.epochs=[0]*len(self.views)
        self.transforms=[];self.pending=None;self.draws=0
        for view in self.views:
            n=len(view);a=self.rng.randrange(1,max(2,n))
            while math.gcd(a,n)!=1:a=(a+1)%n or 1
            self.transforms.append((a,self.rng.randrange(n)))

    def next(self, step):
        policy=NoisePolicy(self.config,step)
        if self.pending is not None:
            row,t,seed=self.pending;self.pending=None;self.draws+=1;self.task_counts['endpoint']+=1
            return row,t,seed
        task=max(self.weights,key=lambda k:self.weights[k]*(self.draws+1)-self.task_counts[k])
        indexes=self.by_task[task]
        index=self.rng.choices(indexes,weights=[len(self.views[i])**.5 for i in indexes],k=1)[0]
        view=self.views[index];n=len(view)
        cursor=self.cursors[index];a,b=self.transforms[index];position=(a*cursor+b)%n
        row=view[position];self.cursors[index]+=1
        if self.cursors[index]==n:
            import math
            self.cursors[index]=0;self.epochs[index]+=1
            a=self.rng.randrange(1,max(2,n))
            while math.gcd(a,n)!=1:a=(a+1)%n or 1
            self.transforms[index]=(a,self.rng.randrange(n))
        t=policy.time(row,self.rng);seed=row.get('noise_seed',self.rng.randrange(2**63-1))
        # Pair references share a base record; avoid selecting unrelated neighbours.
        if task=='endpoint' and not row.get('state'):
            for other_index in (position-1,position+1):
                if 0<=other_index<n:
                    other=view[other_index]
                    if other['pair_id']==row['pair_id'] and other['feedback']['satisfied']!=row['feedback']['satisfied']:
                        self.pending=(other,t,seed);break
        self.draws+=1;self.task_counts[task]+=1
        return row,t,seed

    def state_dict(self):
        return dict(fingerprint=self.corpus.fingerprint,views=[v.info['path'] for v in self.views],
                    rng=self.rng.getstate(),cursors=list(self.cursors),epochs=list(self.epochs),
                    transforms=list(self.transforms),pending=self.pending,draws=self.draws,task_counts=dict(self.task_counts))

    def load_state_dict(self,state):
        if state['fingerprint']!=self.corpus.fingerprint or state['views']!=[v.info['path'] for v in self.views]:
            raise ValueError('resume corpus views changed')
        self.rng.setstate(state['rng']);self.cursors=state['cursors'];self.epochs=state['epochs']
        self.transforms=state['transforms'];self.pending=state['pending'];self.draws=state['draws'];self.task_counts=state['task_counts']
