"""Full-depth observation comparison and explicit historical configurations."""
from dataclasses import asdict
from pathlib import Path
import json
from qwen_latent_cot.bagel.internal_loop import LoopConfig, MODE, JOINT_MICRO


def load_comparison(path,native_depth):
    return validate_config(json.loads(Path(path).read_text()),native_depth)


def validate_config(c,native_depth):
    if c['schema']!=3 or native_depth!=c['expected_num_hidden_layers']:
        raise ValueError('incompatible comparison schema or native depth')
    expected_window={'start_layer':0,'end_layer':native_depth if c['experiment']=='observation_memory' else 8}
    if c['experiment']=='joint_micro_loop':
        start,end=c['layer_window']['start_layer'],c['layer_window']['end_layer']
        if type(start) is not int or type(end) is not int or not 0<=start<end<=native_depth:
            raise ValueError('joint micro window must be a nonempty native layer interval')
    elif c['layer_window']!=expected_window:
        raise ValueError('layer window differs from selected experiment')
    s=c['sampling']
    if s['num_timesteps']!=50 or s['timestep_shift']!=3. or s['cfg_renorm_type']!='global':
        raise ValueError('comparison requires the bound 50-point shift3 native schedule')
    if c['experiment']=='joint_micro_loop':
        counts=c['micro_steps']
        if (not counts or len(set(counts))!=len(counts)
                or any(type(k) is not int or k<1 for k in counts) or counts!=sorted(counts)):
            raise ValueError('require distinct, increasing positive integer micro_steps')
        names=[]
        for window in c['windows']:
            a,b=window['step_start'],window['step_end']
            if type(a) is not int or type(b) is not int or not 0<=a<b<=49:
                raise ValueError('joint time windows must select native denoiser calls')
            if not isinstance(window['name'],str) or not window['name'].replace('_','').isalnum():
                raise ValueError('invalid joint window name')
            names.append(window['name'])
        if not names or len(set(names))!=len(names):raise ValueError('require unique joint time windows')
        if type(c.get('legacy_early20_r2',False)) is not bool:raise ValueError('invalid legacy control flag')
    elif c['experiment']=='loop_grid':
        if c['depths']!=[1,2,3,4] or c['windows']!=[
            {'name':'EARLY_10','step_start':0,'step_end':10},
            {'name':'EARLY_20','step_start':0,'step_end':20}]:
            raise ValueError('require Early10/Early20 x R1/2/3/4')
    elif c['experiment']=='feedback_pilot':
        if c['feedback_max_tokens']<1:raise ValueError('feedback token budget must be positive')
    elif c['experiment']=='observation_memory':
        if 'conditioning_durations' in c:
            if c['observation_step']!=9 or c['conditioning_durations']!=[1,5,10,20] or c.get('legacy_early20_r2') is not True:
                raise ValueError('require step9 fixed cache x 1/5/10/20 calls and legacy Early20 R2')
        elif c['observation_steps']!=[9,19]:
            raise ValueError('require step9/19 for the single-call comparison')
        if c['updates_per_image']!=1:raise ValueError('require exactly one Memory writer per observed arm')
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
    if c['experiment']=='joint_micro_loop':
        if c.get('legacy_early20_r2'):
            arms['LEGACY_EARLY_20_R2']=asdict(LoopConfig(mode=MODE,extra_rounds=2,
                start_layer=0,end_layer=8,progress_end=19/48))
        for window in c['windows']:
            for steps in c['micro_steps']:
                arms[f"JOINT_{window['name']}_K{steps}"]=asdict(LoopConfig(mode=JOINT_MICRO,
                    extra_rounds=0,micro_steps=steps,**c['layer_window'],
                    progress_start=window['step_start']/48,progress_end=(window['step_end']-1)/48))
        return arms
    if c['experiment']=='feedback_pilot':
        return dict(arms,GENERIC_EDIT=dict(base),FEEDBACK_EDIT=dict(base))
    if c['experiment']=='observation_memory':
        if 'conditioning_durations' in c:
            arms['LEGACY_EARLY_20_R2']=asdict(LoopConfig(mode=MODE,extra_rounds=2,start_layer=0,end_layer=8,
                progress_start=0.,progress_end=19/48))
            step=c['observation_step']
            for duration in c['conditioning_durations']:
                for name,observe in (('STATIC',False),('OBSERVED',True)):
                    arms[f'STEP_{step:02d}_L{duration:02d}_{name}']=dict(base,observation_step=step,
                        conditioning_end_step=step+duration,observe_image=observe)
            return arms
        for step in c['observation_steps']:
            for name,observe in (('STATIC',False),('OBSERVED',True)):
                arms[f'STEP_{step:02d}_{name}']=dict(base,observation_step=step,observe_image=observe)
        return arms
    for w in c['windows']:
        for r in c['depths']:
            arms[f"{w['name']}_R{r}"]=asdict(LoopConfig(mode=MODE,extra_rounds=r,**c['layer_window'],
                memory_update=c.get('memory_update','legacy_layerwise'),
                progress_start=w['step_start']/48,progress_end=(w['step_end']-1)/48))
    return arms


def window_metadata(c):
    out={}
    def t(i):
        x=1-i/49
        return 3*x/(1+2*x)
    for arm,cfg in arm_configs(c).items():
        observed=c['experiment']=='observation_memory' and 'observation_step' in cfg
        if observed:
            steps=list(range(cfg['observation_step'],cfg.get('conditioning_end_step',cfg['observation_step']+1)))
        else:steps=[i for i in range(49) if cfg['mode']!='BASE' and cfg['progress_start']<=i/48<=cfg['progress_end']]
        out[arm]={'step_start':min(steps) if steps else 0,'step_end':max(steps)+1 if steps else 0,
            'loop_calls':len(steps),'loop_step_indexes':steps,'extra_rounds':1 if observed else cfg['extra_rounds'],
            'memory_writer_calls':1 if observed else len(steps)*cfg['extra_rounds'],
            'conditioning_kind':'fixed_observation_cache' if observed else cfg.get('memory_update','legacy_layerwise') if steps else 'base',
            'covered_delta_t':t(steps[0])-t(steps[-1]+1) if steps else 0.,
            't_first':t(steps[0]) if steps else None,'t_last':t(steps[-1]) if steps else None,
            't_after_window':t(steps[-1]+1) if steps else None}
        if cfg['mode']==JOINT_MICRO:
            width=cfg['end_layer']-cfg['start_layer'];suffix=c['expected_num_hidden_layers']-cfg['end_layer']
            body=width*cfg['micro_steps'];gen_calls=cfg['start_layer']+body+suffix
            und_calls=body+suffix
            out[arm].update(micro_steps=cfg['micro_steps'],step_scale=1/cfg['micro_steps'],
                conditioning_kind='joint_micro',memory_writer_calls=len(steps),
                memory_writer_count_unit='one_connected_traversal_with_micro_steps_per_active_call',
                gen_layer_calls_per_active_call=gen_calls,und_layer_calls_per_active_call=und_calls,
                total_gen_layer_calls=(49-len(steps))*c['expected_num_hidden_layers']+len(steps)*gen_calls,
                total_und_layer_calls=len(steps)*und_calls)
    return out


def comparison_pairs(c):
    if c['experiment']=='joint_micro_loop':
        pairs=[]
        for window in c['windows']:
            prefix=f"JOINT_{window['name']}_K"
            pairs += [(prefix+str(b),prefix+str(a)) for a,b in zip(c['micro_steps'],c['micro_steps'][1:])]
            if c.get('legacy_early20_r2'):
                pairs += [(prefix+str(k),'LEGACY_EARLY_20_R2') for k in c['micro_steps']]
        return pairs
    if c['experiment']=='feedback_pilot':return [('FEEDBACK_EDIT','GENERIC_EDIT')]
    if c['experiment']=='observation_memory':
        if 'conditioning_durations' in c:
            prefix=f"STEP_{c['observation_step']:02d}_L"
            pairs=[(f'{prefix}{n:02d}_OBSERVED',f'{prefix}{n:02d}_STATIC') for n in c['conditioning_durations']]
            pairs += [(f'{prefix}{b:02d}_OBSERVED',f'{prefix}{a:02d}_OBSERVED')
                      for a,b in zip(c['conditioning_durations'],c['conditioning_durations'][1:])]
            return pairs+[(f'{prefix}{n:02d}_OBSERVED','LEGACY_EARLY_20_R2') for n in c['conditioning_durations']]
        return [(f'STEP_{step:02d}_OBSERVED',f'STEP_{step:02d}_STATIC') for step in c['observation_steps']]
    pairs=[(f'EARLY_20_R{r}',f'EARLY_10_R{r}') for r in c['depths']]
    return pairs+[(f'{w["name"]}_R{r}',f'{w["name"]}_R{r-1}') for w in c['windows'] for r in c['depths'][1:]]


def reference_arm(c):
    if c['experiment']=='joint_micro_loop':
        return ('LEGACY_EARLY_20_R2' if c.get('legacy_early20_r2')
                else f"JOINT_{c['windows'][0]['name']}_K{c['micro_steps'][0]}")
    if c['experiment']=='observation_memory':return 'LEGACY_EARLY_20_R2' if 'conditioning_durations' in c else 'STEP_09_STATIC'
    return 'GENERIC_EDIT' if c['experiment']=='feedback_pilot' else 'EARLY_10_R2'
