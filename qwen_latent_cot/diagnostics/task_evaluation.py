"""Bounded task validation. NLL is a reading diagnostic, not image quality."""
import random
import torch
from .flow_training import conditional_gen
from .loop_training import memory_loop
from .visual_objectives import semantic_result, training_objective
from .schedule import native_suffix, nearest_node
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
            values,details=training_objective(student,row,state,stage,policy=policy,feedback_provider=provider)
            item.update(losses={k:float(v) for k,v in values.items()},supervision=details)
            # Controlled semantic states may be continuous. Image review uses a
            # separate actual native node; never attach their labels to that node.
            image_t=t if row.get('state') else nearest_node(runtime.config,t)[1]
            image_state=state if image_t==t else runtime.training_state(row,image_t,seed)
            item['image_state']=image_state.metadata
            loop=memory_loop(student,image_state,runtime.config.get('feedback_updates',1))
            off=student.write(loop.baseline.source,image_t,prompt=image_state.prompt_kv,source_off=True)
            banks={'memory':loop.banks[-1],'source_off':off}
            donor=next((r for r in rows if r['sample_id']!=row['sample_id'] and r['prompt']==row['prompt'] and r.get('image')!=row.get('image') and not r.get('state')),None)
            if donor:
                other=runtime.training_state(donor,image_t,seed);source=conditional_gen(runtime,other).source
                banks['source_swap']=student.write(source,image_t,prompt=image_state.prompt_kv)
                item['swap_sample_id']=donor['sample_id']
            predictions={'base':runtime.velocity(image_state.x_t,image_t,image_state.flow)[1]}
            predictions.update({arm:runtime.gen_arm(image_state.x_t,image_t,image_state.flow,bank,image_state.lengths,image_state.positions)[1] for arm,bank in banks.items()})
            images={}
            for arm,velocity in predictions.items():
                latent=runtime.continue_once(image_state.x_t,image_t,velocity,image_state.flow,native_suffix(runtime.config,image_t))
                name=f'{row["sample_id"]}_{arm}.png';runtime.image(latent).save(output/name);images[arm]=name
            item.update(images=images,quality_status='unverified; paired final-image review required')
        records.append(item)
    write_json(output/'metrics.json',records);return records
