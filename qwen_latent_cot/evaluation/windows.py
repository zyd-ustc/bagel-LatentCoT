"""Compare denoising time windows at the selected native layer window [0,8)."""
from dataclasses import asdict
from pathlib import Path
import json
from qwen_latent_cot.bagel.internal_loop import LoopConfig, MODE


def load_comparison(path, native_depth):
    return validate_config(json.loads(Path(path).read_text()), native_depth)


def validate_config(config, native_depth):
    if config['schema'] != 2 or config['comparison_axis'] != 'denoising_steps':
        raise ValueError('require the denoising-step comparison config')
    if native_depth != config['expected_num_hidden_layers']:
        raise ValueError('configured windows require the recorded native decoder depth')
    if config['extra_rounds'] != 2 or config['layer_window'] != {'start_layer':0,'end_layer':8}:
        raise ValueError('this comparison fixes R2 and the selected layer window [0,8)')
    sampling = config['sampling']
    if sampling['num_timesteps'] != 50 or sampling['timestep_shift'] != 3.0:
        raise ValueError('time-window counts are bound to the native 50-point shifted schedule')
    if sampling['cfg_renorm_type'] != 'global':
        raise ValueError('require global CFG renormalization')
    windows = config['windows']
    required={'EARLY_05':(0,5),'EARLY_10':(0,10),'EARLY_20':(0,20),'LATE_20':(29,49),'FULL':(0,49)}
    bounds = {w['name']:(w['step_start'],w['step_end']) for w in windows}
    if len(windows) != 5 or len({w['name'] for w in windows}) != 5:
        raise ValueError('require five uniquely named loop arms, plus shared Base')
    if bounds != required or any(type(v) is not int for w in windows for v in (w['step_start'],w['step_end'])):
        raise ValueError('require Early5/10/20, matched Late20 and Full')
    for w in windows:
        if w['name'] == 'BASE' or not w['name'].replace('_','').isalnum():
            raise ValueError('unsafe or reserved arm name')
    seeds = config['seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('require distinct integer seeds')
    if config['max_prompts'] < 1:
        raise ValueError('max_prompts must be positive')
    return config


def arm_configs(config):
    layer = config['layer_window']
    calls = config['sampling']['num_timesteps']-1
    arms = {'BASE':asdict(LoopConfig(mode='BASE',extra_rounds=0,**layer))}
    for w in config['windows']:
        # Runtime progress uses step/(calls-1), with inclusive endpoints.
        # Config step_end is exclusive: include exactly start ... end-1.
        arms[w['name']] = asdict(LoopConfig(mode=MODE,extra_rounds=2,**layer,
            progress_start=w['step_start']/(calls-1),
            progress_end=(w['step_end']-1)/(calls-1)))
    return arms


def window_metadata(config):
    sampling = config['sampling'];calls=sampling['num_timesteps']-1
    shift=sampling['timestep_shift']
    def timestep(step):
        raw=1-step/calls
        return shift*raw/(1+(shift-1)*raw)
    out={'BASE':{'step_start':0,'step_end':0,'loop_calls':0,'loop_step_indexes':[],
        't_first':None,'t_last':None,'t_after_window':None}}
    for w in config['windows']:
        a,b=w['step_start'],w['step_end']
        out[w['name']]={'step_start':a,'step_end':b,'loop_calls':b-a,
            'loop_step_indexes':list(range(a,b)),'t_first':timestep(a),
            't_last':timestep(b-1),'t_after_window':timestep(b),
            'timestep_scope':'analytic native shifted schedule; float32 rounding may differ'}
    return out
