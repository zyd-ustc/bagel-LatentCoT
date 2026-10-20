"""Indexed Observe/Feedback text training, separate from generation full-loop SFT."""
import time
import torch
from . import distributed as parallel
from .curriculum import CoverageSampler, NoisePolicy
from .visual import validate_config
from .visual_objectives import training_objective
from .visual_record import save_visual_checkpoint, load_visual_checkpoint
from .flow_training import ExecutionCounter
from .record import FrozenGuard, append_json, write_json
from .training_logging import configure_logging, start_wandb, training_metrics, validation_metrics
from ..bagel.accelerator import synchronize, reset_peak_memory_stats, max_memory_allocated


def train(student, corpus, stage, output, parent=None, *, resume=None, skip_validation=False, wandb_options=None):
    if stage not in ('C2-observe', 'C2-feedback'):
        raise ValueError('Generation training uses interaction full-loop SFT')
    runtime=student.runtime; cfg=runtime.config; validate_config(runtime,stage)
    excluded=set(cfg.get('excluded_sources',[]))
    active=[]
    for view in corpus.views:
        if view.info['source'] in excluded:view.close()
        else:active.append(view)
    corpus.views=active
    sampler=CoverageSampler(corpus,cfg,stage)
    batch=cfg['effective_batch']; world=parallel.world_size(); worker=parallel.rank()
    if batch%world: raise ValueError('global effective batch must be divisible by world size')
    if cfg['loss_weights']['teacher'] and student.text_teacher is None:
        raise ValueError('teacher loss requires an explicit text teacher; set teacher weight to zero for teacher-free training')
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
        sampler.load_state_dict(saved['schedule']);start_step=saved['step']
        same_validation=all(saved['config'].get(key)==cfg.get(key) for key in ('validation_max_families','validation_max_rows'))
        best=saved['best'] if same_validation else None
    parallel.broadcast_parameters(student)
    guard=FrozenGuard(runtime.model,runtime.bundle.vae,student.text_teacher,exclude_trainable=True)
    counter=ExecutionCounter(runtime);reset_peak_memory_stats(runtime.device);start=time.perf_counter();names=('sem','teacher','flow')
    logger=configure_logging(output);tracker=None;completed_step=start_step;status='failed';failure=None
    try:
        tracker=start_wandb(output,cfg,dict(stage=stage,corpus_fingerprint=corpus.fingerprint,
            world_size=world,parent_checkpoint=str(parent) if parent else None,
            resume_checkpoint=str(resume) if resume else None,resumed_from_step=start_step,
            text_teacher_hash=student.text_teacher_hash,
            trainable_parameters=sum(p.numel() for p in parameters)),wandb_options)
        if worker==0:
            save_visual_checkpoint(output/'init.pt',student,start_step,stage,parent)
            write_json(output/'training_plan.json',dict(stage=stage,corpus=corpus.fingerprint,global_batch=batch,
                available_task_weights=sampler.weights,excluded_sources=sorted(excluded),source_layers=cfg['source_layers'],und_train=cfg['und_train'],
                trainable_parameters={name:p.numel() for name,p in student.native_parameters().items()},
                validation_samples=[r['sample_id'] for r in val],selection=cfg['selection']))
            logger.info('Training %s: steps=%d..%d batch=%d world=%d tasks=%s tracking=%s',
                stage,start_step+1,cfg['steps'],batch,world,sampler.weights,
                wandb_options['mode'] if wandb_options else 'disabled')
        for step in range(start_step+1,cfg['steps']+1):
            step_start=time.perf_counter()
            student.train();optimizer.zero_grad(set_to_none=True);sums=[0.]*8;samples=[];policy=NoisePolicy(cfg,step)
            # Every rank advances the same small sampler; no corpus broadcast or per-rank random schedule.
            for micro in range(batch):
                row,t,seed=sampler.next(step)
                if micro%world!=worker:continue
                state=runtime.training_state(row,t,seed);before=list(counter.counts)
                values,details=training_objective(student,row,state,stage,policy=policy)
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
            optimizer.step();runtime._neutral_context=None;completed_step=step;guard.check();synchronize(runtime.device)
            totals=parallel.sum_statistics(sums,runtime.device);count=totals[4]
            sampled=sorted(parallel.gather_samples(samples),key=lambda r:r['global_micro'])
            step_seconds,peak_memory=parallel.max_statistics(
                [time.perf_counter()-step_start,max_memory_allocated(runtime.device)],runtime.device)
            if worker==0:
                record=dict(step=step,stage=stage,loss=totals[3]/count,
                    losses={n:totals[i]/count for i,n in enumerate(names)},group_gradient_norms=groups,gradient_norm=float(norm),
                    sampled_states=sampled,world_size=world,global_effective_batch=batch,
                    actual_gen_entries=int(totals[5]),gen_layer_entries=int(totals[6]),und_layer_entries=int(totals[7]),
                    step_seconds=step_seconds,elapsed_seconds=time.perf_counter()-start,peak_memory_bytes=int(peak_memory))
                append_json(output/'training.jsonl',record)
                metrics=training_metrics(record,optimizer,sampler,policy)
                metrics.update({'trainer/step':step,'train/progress':step/cfg['steps'],'train/global_samples':step*batch})
                if tracker:tracker.log(metrics)
                if step==start_step+1 or step%cfg.get('log_every',10)==0 or step==cfg['steps']:
                    logger.info('%s step=%d/%d loss=%.6f sem=%.6f teacher=%.6f flow=%.6f grad=%.4f lr=%.3g seconds=%.2f peak_GiB=%.2f',
                        stage,step,cfg['steps'],record['loss'],record['losses']['sem'],record['losses']['teacher'],
                        record['losses']['flow'],record['gradient_norm'],optimizer.param_groups[0]['lr'],step_seconds,peak_memory/1024**3)
            if step%cfg['eval_every']==0 or step==cfg['steps']:
                parallel.barrier();score=None
                if not skip_validation:
                    logger.info('Validation starts: step=%d samples=%d',step,len(val))
                    from .task_evaluation import evaluate_tasks
                    directory=output/f'validation_{step:06d}';local=directory/f'rank_{worker:02d}';local.mkdir(parents=True,exist_ok=True)
                    families=sorted({r['family_id'] for r in val});assigned=set(families[worker::world])
                    records=evaluate_tasks(student,[r for r in val if r['family_id'] in assigned],local,policy,stage)
                    stats=parallel.sum_statistics([sum(r['observe_nll'] for r in records if 'observe_nll' in r),sum('observe_nll' in r for r in records)],runtime.device)
                    if stage=='C2-observe' and stats[1]:score=stats[0]/stats[1]
                    if stage=='C2-feedback':
                        sums=[]
                        for satisfied in (False,True):
                            group=[r for r in records if 'feedback_nll' in r and r['expected_feedback']['satisfied']==satisfied]
                            sums.extend((sum(r['feedback_nll'] for r in group),len(group)))
                        reduced=parallel.sum_statistics(sums,runtime.device)
                        means=[reduced[i]/reduced[i+1] for i in (0,2) if reduced[i+1]]
                        if means:score=sum(means)/len(means)
                    records=parallel.gather_samples(records);guard.check()
                    if worker==0:
                        metrics=validation_metrics(records)
                        metrics['trainer/step']=step
                        append_json(output/'validation.jsonl',dict(step=step,stage=stage,metrics=metrics))
                        if tracker:tracker.log(metrics)
                        logger.info('Validation step=%d metrics=%s',step,metrics)
                        write_json(directory/'metrics_index.json',records)
                        write_json(directory/'selection.json',dict(score=score,selection=cfg['selection']))
                        if score is not None and (best is None or score<best):
                            best=score;save_visual_checkpoint(output/'best.pt',student,step,stage,parent,optimizer,sampler.state_dict(),best)
                if worker==0:
                    for path in (output/f'step_{step:06d}.pt',output/'last.pt'):
                        save_visual_checkpoint(path,student,step,stage,parent,optimizer,sampler.state_dict(),best)
                    logger.info('Checkpoint saved: step=%d last=%s',step,output/'last.pt')
                    keep=cfg.get('checkpoint_keep',2)
                    for old in sorted(output.glob('step_*.pt'))[:-keep]:
                        old.unlink()
                parallel.barrier()
        status='completed'
    except BaseException as error:
        failure=dict(type=type(error).__name__,message=str(error))
        logger.exception('Training failed at step %d',completed_step)
        raise
    finally:
        counter.close()
        if worker==0:
            summary=dict(status=status,stage=stage,completed_steps=completed_step,target_steps=cfg['steps'],
                resumed_from_step=start_step,corpus=corpus.fingerprint,elapsed_seconds=time.perf_counter()-start,
                checkpoint_selection=cfg['selection'],validation_skipped=skip_validation)
            if stage=='C2-observe':summary['best_observation_nll']=best
            if stage=='C2-feedback':summary['best_balanced_feedback_nll']=best
            if failure:summary['failure']=failure
            write_json(output/'training_summary.json',summary)
            logger.info('Training %s: completed_steps=%d target_steps=%d',status,completed_step,cfg['steps'])
            if tracker:
                tracker.summary.update(summary)
                tracker.finish(exit_code=0 if status=='completed' else 1)
