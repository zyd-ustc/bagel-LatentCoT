"""Bounded task validation. NLL is a reading diagnostic, not image quality."""
import random
import torch
from .flow_training import conditional_gen
from .visual_objectives import semantic_result
from .record import write_json


@torch.no_grad()
def evaluate_tasks(student, rows, output, policy, stage, provider=None):
    runtime=student.runtime;records=[];rng=random.Random(runtime.config['seed']);paired={}
    for row in rows:
        seed=row.get('noise_seed',runtime.config['noise_seeds'][0])
        if row['task']=='feedback' and not row.get('state'):
            pair=row.get('pair_id',row['sample_id'])
            if pair not in paired:paired[pair]=policy.time(row,rng)
            t=paired[pair]
        else:t=policy.time(row,rng)
        state=runtime.training_state(row,t,seed);baseline=conditional_gen(runtime,state)
        item=dict(sample_id=row['sample_id'],family_id=row['family_id'],task=row['task'],t=t,state=state.metadata)
        if row['task']=='observe':
            bank=student.write(baseline.source,t);off=student.write(baseline.source,t,source_off=True)
            item.update(observe_nll=semantic_result(runtime,row,bank)['nll'],
                        source_off=semantic_result(runtime,row,off),caption=semantic_result(runtime,row,bank))
        elif stage=='C2-feedback':
            bank=student.write(baseline.source,t,prompt=state.prompt_kv)
            off=student.write(baseline.source,t,prompt=state.prompt_kv,source_off=True)
            correct=semantic_result(runtime,row,bank)
            item.update(feedback_nll=correct['nll'],caption=correct,
                        source_off=semantic_result(runtime,row,off),
                        expected_feedback=row['feedback'],target_text=row['caption'],
                        prompt=row['prompt'],source=row['source'],image=row.get('image'))
            donor=next((r for r in rows if r['task']=='feedback' and r['sample_id']!=row['sample_id']
                        and r['prompt']==row['prompt'] and r.get('image')!=row.get('image')
                        and not r.get('state')),None)
            if donor:
                other=runtime.training_state(donor,t,seed)
                source=conditional_gen(runtime,other).source
                swapped=student.write(source,t,prompt=state.prompt_kv)
                item.update(source_swap=semantic_result(runtime,row,swapped),
                            swap_sample_id=donor['sample_id'],swap_target_text=donor['caption'])
        else:
            raise ValueError('Generation validation uses interaction full-loop SFT')
        records.append(item)
    write_json(output/'metrics.json',records);return records
