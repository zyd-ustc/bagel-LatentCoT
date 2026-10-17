"""Local training logs and one W&B run on the global rank zero."""
from collections import Counter, defaultdict
import logging
import os
from . import distributed as parallel
from .record import write_json


def configure_logging(output):
    directory = output/'logs'; directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f'bagel.training.{output.resolve()}.{parallel.rank()}')
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear(); logger.propagate = False; logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handlers = [logging.FileHandler(directory/f'rank_{parallel.rank():02d}.log', encoding='utf-8')]
    if parallel.rank() == 0:
        handlers.append(logging.StreamHandler())
    for handler in handlers:
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def start_wandb(output, config, metadata, options):
    if parallel.rank() != 0 or not options or options['mode'] == 'disabled':
        return None
    import wandb
    # Every launch has its own run, including training resumed from a checkpoint.
    for key in ('WANDB_RUN_ID', 'WANDB_RESUME', 'WANDB_SWEEP_ID', 'WANDB_RUN_GROUP', 'WANDB_DIR'):
        os.environ.pop(key, None)
    run = wandb.init(project=options['project'], entity=options.get('entity'),
        name=options.get('name') or output.name, resume='never' if options['mode']=='online' else None,
        mode=options['mode'], dir=str(output.resolve()), job_type=metadata['stage'],
        config=dict(config, **metadata), save_code=False,
        settings=wandb.Settings(disable_git=True))
    run.define_metric('trainer/step')
    run.define_metric('*', step_metric='trainer/step')
    write_json(output/'wandb.json', dict(id=run.id, project=run.project, entity=run.entity,
        mode=options['mode'], url=run.url if options['mode']=='online' else None))
    return run


def training_metrics(record, optimizer, sampler, policy):
    metrics = {'train/loss':record['loss'], 'train/gradient_norm':record['gradient_norm'],
        'train/learning_rate':optimizer.param_groups[0]['lr'],
        'performance/step_seconds':record['step_seconds'],
        'performance/samples_per_second':record['global_effective_batch']/record['step_seconds'],
        'performance/elapsed_seconds':record['elapsed_seconds'],
        'memory/peak_allocated_gib':record['peak_memory_bytes']/1024**3,
        'compute/gen_entries':record['actual_gen_entries'],
        'compute/gen_layer_entries':record['gen_layer_entries'],
        'compute/und_layer_entries':record['und_layer_entries'],
        'data/observe_max_t':policy.observe_max, 'data/endpoint_max_t':policy.endpoint_max}
    metrics.update({f'train/loss_{key}':value for key,value in record['losses'].items()})
    metrics.update({f'train/gradient_{key}':value for key,value in record['group_gradient_norms'].items()})
    metrics.update({f'data/sampled_{key}':value for key,value in sampler.task_counts.items()})
    tasks = Counter(row['task'] for row in record['sampled_states'])
    metrics.update({f'train/task_fraction_{key}':value/record['global_effective_batch'] for key,value in tasks.items()})
    times = [row['state']['t'] for row in record['sampled_states']]
    metrics.update({'train/t_mean':sum(times)/len(times), 'train/t_min':min(times), 'train/t_max':max(times)})
    return metrics


def validation_metrics(records):
    values = defaultdict(list); counts = Counter()
    for row in records:
        task = row['task']; counts[task] += 1
        if 'observe_nll' in row:
            values['val/observe/nll'].append(row['observe_nll'])
            values['val/observe/source_off_nll'].append(row['source_off']['nll'])
            for name,key in (('fact_accuracy','caption'), ('source_off_fact_accuracy','source_off')):
                score = row[key]['fact_accuracy']
                if score is not None: values['val/observe/'+name].append(score)
        if 'feedback_nll' in row:
            satisfied=row['expected_feedback']['satisfied']
            label='hold' if satisfied else 'repair'
            values['val/feedback/nll'].append(row['feedback_nll'])
            values['val/feedback/'+label+'_nll'].append(row['feedback_nll'])
            values['val/feedback/source_off_nll'].append(row['source_off']['nll'])
            values['val/feedback/source_gain_nll'].append(row['source_off']['nll']-row['feedback_nll'])
            canonical=row['caption']['text'].strip().lower().rstrip('.')=='no change is needed'
            values['val/feedback/'+('canonical_hold_rate' if satisfied else 'false_hold_rate')].append(float(canonical))
            if 'source_swap' in row:
                values['val/feedback/source_swap_gain_nll'].append(row['source_swap']['nll']-row['feedback_nll'])
        for name,value in row.get('losses',{}).items():
            values[f'val/{task}/loss_{name}'].append(value)
    metrics = {key:sum(items)/len(items) for key,items in values.items()}
    metrics.update({f'val/count_{key}':value for key,value in counts.items()})
    if 'val/observe/nll' in metrics:
        metrics['val/observe/source_gain_nll'] = metrics['val/observe/source_off_nll']-metrics['val/observe/nll']
    labels=[metrics['val/feedback/'+kind+'_nll'] for kind in ('repair','hold')
            if 'val/feedback/'+kind+'_nll' in metrics]
    if labels:metrics['val/feedback/balanced_nll']=sum(labels)/len(labels)
    return metrics
