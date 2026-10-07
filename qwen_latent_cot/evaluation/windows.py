"""Six fixed-width R2 windows on the same frozen BAGEL model."""
from dataclasses import asdict
from pathlib import Path
import json
from qwen_latent_cot.bagel.internal_loop import LoopConfig, MODE


def load_comparison(path, native_depth):
    return validate_config(json.loads(Path(path).read_text()), native_depth)


def validate_config(config, native_depth):
    if native_depth != config['expected_num_hidden_layers']:
        raise ValueError('configured windows require the recorded native decoder depth')
    if config['extra_rounds'] != 2:
        raise ValueError('this comparison fixes all Memory arms to R2')
    windows = config['windows']
    if len(windows) != 6 or len({w['name'] for w in windows}) != 6:
        raise ValueError('require six uniquely named windows')
    if len({(w['start_layer'], w['end_layer']) for w in windows}) != 6:
        raise ValueError('duplicate layer windows')
    for phase in ('early', 'middle', 'late'):
        if sum(w['phase'] == phase for w in windows) != 2:
            raise ValueError('require two windows per phase')
    for w in windows:
        if w['name'] == 'BASE' or not w['name'].replace('_','').isalnum():
            raise ValueError('unsafe or reserved window name')
        if w['end_layer'] - w['start_layer'] != 8 or w['end_layer'] > native_depth:
            raise ValueError('all windows must have eight valid decoder layers')
        LoopConfig(start_layer=w['start_layer'], end_layer=w['end_layer'])
    sampling = config['sampling']
    if sampling['num_timesteps'] < 2 or sampling['cfg_renorm_type'] != 'global':
        raise ValueError('require native schedule with global CFG renormalization')
    seeds = config['seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('require distinct integer seeds')
    if config['max_prompts'] < 1:
        raise ValueError('max_prompts must be positive')
    return config


def arm_configs(config):
    first = config['windows'][0]
    arms = {'BASE': asdict(LoopConfig(mode='BASE', extra_rounds=0,
        start_layer=first['start_layer'], end_layer=first['end_layer']))}
    for w in config['windows']:
        arms[w['name']] = asdict(LoopConfig(mode=MODE, extra_rounds=2,
            start_layer=w['start_layer'], end_layer=w['end_layer']))
    return arms
