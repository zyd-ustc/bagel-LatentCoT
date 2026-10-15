"""One indexed-task trainer for Observe, Feedback and same-state memory loops."""
import time
import torch
from . import distributed as parallel
from .curriculum import CoverageSampler, NoisePolicy
from .visual import validate_config
from .visual_objectives import training_objective, online_feedback_provider
from .visual_record import save_visual_checkpoint, load_visual_checkpoint
from .flow_training import ExecutionCounter
from .record import FrozenGuard, append_json, write_json
from ..bagel.accelerator import synchronize, reset_peak_memory_stats, max_memory_allocated


def train(student, corpus, stage, output, parent=None, *, resume=None, skip_validation=False):
    runtime=student.runtime; cfg=runtime.config; validate_config(runtime,stage)
    sampler=CoverageSampler(corpus,cfg,stage)
    batch=cfg['effective_batch']; world=parallel.world_size(); worker=parallel.rank()
    if batch%world: raise ValueError('global effective batch must be divisible by world size')
    if cfg['loss_weights']['teacher'] and student.text_teacher is None:
        raise ValueError('teacher loss requires an explicit text teacher; set teacher weight to zero for teacher-free training')
    provider=online_feedback_provider(runtime) if cfg.get('online_feedback') else None
    if stage=='C3-loop' and 'endpoint' in sampler.weights and provider is None:
        raise ValueError('C3 endpoint repair requires fresh online feedback')
    val=corpus.select('val', cfg.get('validation_max_families',64),
                     {'observe'} if stage=='C2-observe' else sampler.weights,
                     max_rows=cfg.get('validation_max_rows',128))
    if not val and not skip_validation: raise ValueError('no bounded validation task views')
    parameters=[p for p in student.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    start_step=0; best=None
    runtime.data_hashes=dict(corpus=corpus.fingerprint)
    if resume:
        saved=load_visual_checkpoint(resume,student,optimizer=optimizer,resume=True,expected_stage=stage)
        sampler.load_state_dict(saved['schedule']);start_step=saved['step'];best=saved['best']
    parallel.broadcast_parameters(student)
    guard=FrozenGuard(runtime.model,runtime.bundle.vae,student.text_teacher,*([student.writer] if not cfg['train_writer'] else []))
    counter=ExecutionCounter(runtime);reset_peak_memory_stats(runtime.device);start=time.perf_counter();names=('sem','teacher','flow')
    if worker==0:
        save_visual_checkpoint(output/'init.pt',student,start_step,stage,parent)
        write_json(output/'training_plan.json',dict(stage=stage,corpus=corpus.fingerprint,global_batch=batch,
            available_task_weights=sampler.weights,source_layers=cfg['source_layers'],online_feedback=bool(provider),
            validation_samples=[r['sample_id'] for r in val],selection='Observe NLL only; Feedback requires final-image review'))
    try:
        for step in range(start_step+1,cfg['steps']+1):
            student.train();optimizer.zero_grad(set_to_none=True);sums=[0.]*8;samples=[];policy=NoisePolicy(cfg,step)
            # Every rank advances the same small sampler; no corpus broadcast or per-rank random schedule.
            for micro in range(batch):
                row,t,seed=sampler.next(step)
                if micro%world!=worker:continue
                state=runtime.training_state(row,t,seed);before=list(counter.counts)
                values,details=training_objective(student,row,state,stage,policy=policy,feedback_provider=provider)
                loss=sum(cfg['loss_weights'][n]*values[n] for n in names)
                if not torch.isfinite(loss):raise FloatingPointError('non-finite training loss')
                if not loss.requires_grad:raise RuntimeError('task has no enabled loss reaching memory')
                (loss/batch).backward()
                for i,n in enumerate(names):sums[i]+=float(values[n].detach())
                sums[3]+=float(loss.detach());sums[4]+=1
                for i,(a,b) in enumerate(zip(counter.counts,before)):sums[5+i]+=a-b
                samples.append(dict(sample_id=row['sample_id'],task=row['task'],source=row['source'],
                    global_micro=micro,rank=worker,state=state.metadata,**details))
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):raise FloatingPointError('non-finite memory gradient')
            parallel.sum_gradients(student,allow_unused=True)
            groups={name:float(sum((p.grad.float().square().sum() for p in group if p.grad is not None),torch.zeros((),device=runtime.device)).sqrt())
                    for name,group in student.parameter_groups().items() if group}
            if not any(groups.values()):raise RuntimeError('memory has no global gradient')
            norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();guard.check();synchronize(runtime.device)
            totals=parallel.sum_statistics(sums,runtime.device);count=totals[4]
            sampled=sorted(parallel.gather_samples(samples),key=lambda r:r['global_micro'])
            if worker==0:
                append_json(output/'training.jsonl',dict(step=step,stage=stage,loss=totals[3]/count,
                    losses={n:totals[i]/count for i,n in enumerate(names)},group_gradient_norms=groups,gradient_norm=float(norm),
                    sampled_states=sampled,world_size=world,global_effective_batch=batch,
                    actual_gen_entries=int(totals[5]),gen_layer_entries=int(totals[6]),und_layer_entries=int(totals[7]),
                    elapsed_seconds=time.perf_counter()-start,peak_memory_bytes=max_memory_allocated(runtime.device)))
                if step==start_step+1 or step%10==0:print(f'Step {step}: loss={totals[3]/count:.6f}',flush=True)
            if step%cfg['eval_every']==0 or step==cfg['steps']:
                parallel.barrier();score=None
                if not skip_validation:
                    from .task_evaluation import evaluate_tasks
                    directory=output/f'validation_{step:06d}';local=directory/f'rank_{worker:02d}';local.mkdir(parents=True,exist_ok=True)
                    families=sorted({r['family_id'] for r in val});assigned=set(families[worker::world])
                    records=evaluate_tasks(student,[r for r in val if r['family_id'] in assigned],local,policy,stage,provider)
                    stats=parallel.sum_statistics([sum(r['observe_nll'] for r in records if 'observe_nll' in r),sum('observe_nll' in r for r in records)],runtime.device)
                    if stage=='C2-observe' and stats[1]:score=stats[0]/stats[1]
                    records=parallel.gather_samples(records);guard.check()
                    if worker==0:
                        write_json(directory/'metrics_index.json',records)
                        write_json(directory/'selection.json',dict(observe_nll=score,quality_status='unverified; review full suffix images'))
                        if score is not None and (best is None or score<best):
                            best=score;save_visual_checkpoint(output/'best.pt',student,step,stage,parent,optimizer,sampler.state_dict(),best)
                if worker==0:
                    for path in (output/f'step_{step:06d}.pt',output/'last.pt'):
                        save_visual_checkpoint(path,student,step,stage,parent,optimizer,sampler.state_dict(),best)
                parallel.barrier()
    finally:counter.close()
    if worker==0:write_json(output/'training_summary.json',dict(stage=stage,completed_steps=cfg['steps'],resumed_from_step=start_step,
            best_observation_nll=best,corpus=corpus.fingerprint,frozen_guard='passed',quality_status='unverified'))
