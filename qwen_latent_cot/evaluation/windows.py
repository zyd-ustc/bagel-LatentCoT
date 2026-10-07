"""Fixed [0,8) loop grid and a separate native feedback reference experiment."""
from dataclasses import asdict
from pathlib import Path
import json
from qwen_latent_cot.bagel.internal_loop import LoopConfig, MODE


def load_comparison(path,native_depth):
    return validate_config(json.loads(Path(path).read_text()),native_depth)


def validate_config(c,native_depth):
    if c['schema']!=3 or native_depth!=c['expected_num_hidden_layers']:
        raise ValueError('incompatible comparison schema or native depth')
    if c['layer_window']!={'start_layer':0,'end_layer':8}:
        raise ValueError('selected model window is [0,8)')
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
        steps=[i for i in range(49) if cfg['mode']!='BASE' and cfg['progress_start']<=i/48<=cfg['progress_end']]
        out[arm]={'step_start':min(steps) if steps else 0,'step_end':max(steps)+1 if steps else 0,
            'loop_calls':len(steps),'loop_step_indexes':steps,'extra_rounds':cfg['extra_rounds'],
            't_first':t(steps[0]) if steps else None,'t_last':t(steps[-1]) if steps else None,
            't_after_window':t(steps[-1]+1) if steps else None}
    return out


def comparison_pairs(c):
    if c['experiment']=='feedback_pilot':return [('FEEDBACK_EDIT','GENERIC_EDIT')]
    pairs=[(f'EARLY_20_R{r}',f'EARLY_10_R{r}') for r in c['depths']]
    return pairs+[(f'{w["name"]}_R{r}',f'{w["name"]}_R{r-1}') for w in c['windows'] for r in c['depths'][1:]]


def reference_arm(c):
    return 'GENERIC_EDIT' if c['experiment']=='feedback_pilot' else 'EARLY_10_R2'
