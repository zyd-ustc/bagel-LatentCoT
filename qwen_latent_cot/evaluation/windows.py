"""Full-depth observation comparison and explicit historical configurations."""
from dataclasses import asdict
from pathlib import Path
import json
from qwen_latent_cot.bagel.internal_loop import LoopConfig, MODE


def load_comparison(path,native_depth):
    return validate_config(json.loads(Path(path).read_text()),native_depth)


def validate_config(c,native_depth):
    if c['schema']!=3 or native_depth!=c['expected_num_hidden_layers']:
        raise ValueError('incompatible comparison schema or native depth')
    expected_window={'start_layer':0,'end_layer':native_depth if c['experiment']=='observation_memory' else 8}
    if c['layer_window']!=expected_window:
        raise ValueError('layer window differs from selected experiment')
    s=c['sampling']
    if s['num_timesteps']!=50 or s['timestep_shift']!=3. or s['cfg_renorm_type']!='global':
        raise ValueError('comparison requires the bound 50-point shift3 native schedule')
    if c['experiment']=='loop_grid':
        if c['depths']!=[1,2,3,4] or c['windows']!=[
            {'name':'EARLY_10','step_start':0,'step_end':10},
            {'name':'EARLY_20','step_start':0,'step_end':20}]:
            raise ValueError('require Early10/Early20 x R1/2/3/4')
    elif c['experiment']=='feedback_pilot':
        if c['feedback_max_tokens']<1:raise ValueError('feedback token budget must be positive')
    elif c['experiment']=='observation_memory':
        if c['observation_steps']!=[9,19] or c['updates_per_image']!=1:
            raise ValueError('require exactly one update at step9 or step19 per arm')
        if c['probe_questions_per_image']<1 or c['probe_max_tokens']<1:
            raise ValueError('observation diagnostics require positive probe budgets')
        if c['edit_sampling']!={'cfg_text_scale':3.,'cfg_img_scale':1.5,'cfg_interval':[.4,1.]}:
            raise ValueError('require bound native edit CFG')
    else:raise ValueError('unknown experiment')
    if c['max_prompts']!=c['expected_prompts'] or c['max_prompts']<1:
        raise ValueError('explicit expected benchmark coverage required')
    if not c['seeds'] or len(set(c['seeds']))!=len(c['seeds']) or any(type(v) is not int for v in c['seeds']):
        raise ValueError('require distinct integer seeds')
    return c


def arm_configs(c):
    base=asdict(LoopConfig(mode='BASE',extra_rounds=0,**c['layer_window']))
    arms={'BASE':base}
    if c['experiment']=='feedback_pilot':
        return dict(arms,GENERIC_EDIT=dict(base),FEEDBACK_EDIT=dict(base))
    if c['experiment']=='observation_memory':
        for step in c['observation_steps']:
            for name,observe in (('STATIC',False),('OBSERVED',True)):
                arms[f'STEP_{step:02d}_{name}']=dict(base,observation_step=step,observe_image=observe)
        return arms
    for w in c['windows']:
        for r in c['depths']:
            arms[f"{w['name']}_R{r}"]=asdict(LoopConfig(mode=MODE,extra_rounds=r,**c['layer_window'],
                progress_start=w['step_start']/48,progress_end=(w['step_end']-1)/48))
    return arms


def window_metadata(c):
    out={}
    def t(i):
        x=1-i/49
        return 3*x/(1+2*x)
    for arm,cfg in arm_configs(c).items():
        if c['experiment']=='observation_memory':
            steps=[cfg['observation_step']] if arm!='BASE' else []
        else:steps=[i for i in range(49) if cfg['mode']!='BASE' and cfg['progress_start']<=i/48<=cfg['progress_end']]
        out[arm]={'step_start':min(steps) if steps else 0,'step_end':max(steps)+1 if steps else 0,
            'loop_calls':len(steps),'loop_step_indexes':steps,'extra_rounds':(1 if steps else 0) if c['experiment']=='observation_memory' else cfg['extra_rounds'],
            't_first':t(steps[0]) if steps else None,'t_last':t(steps[-1]) if steps else None,
            't_after_window':t(steps[-1]+1) if steps else None}
    return out


def comparison_pairs(c):
    if c['experiment']=='feedback_pilot':return [('FEEDBACK_EDIT','GENERIC_EDIT')]
    if c['experiment']=='observation_memory':return [(f'STEP_{step:02d}_OBSERVED',f'STEP_{step:02d}_STATIC') for step in c['observation_steps']]
    pairs=[(f'EARLY_20_R{r}',f'EARLY_10_R{r}') for r in c['depths']]
    return pairs+[(f'{w["name"]}_R{r}',f'{w["name"]}_R{r-1}') for w in c['windows'] for r in c['depths'][1:]]


def reference_arm(c):
    if c['experiment']=='observation_memory':return 'STEP_09_STATIC'
    return 'GENERIC_EDIT' if c['experiment']=='feedback_pilot' else 'EARLY_10_R2'
