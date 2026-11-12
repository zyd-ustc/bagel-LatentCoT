"""Configuration for the full 490k Stage 1 training recipe."""

import json
import math
from pathlib import Path

from . import CONDITION_PROTOCOL, KIND, PROTOCOL_VERSION


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    parent = cfg.pop('extends', None)
    if parent:
        base = json.loads((path.parent / parent).read_text())
        if 'extends' in base:
            raise ValueError('A shared core cannot extend another configuration')
        base.update(cfg)
        cfg = base
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg.get('kind') != KIND or cfg.get('protocol_version') != PROTOCOL_VERSION:
        raise ValueError('Isochronic state flow configuration required')
    if cfg.get('stage') != 1 or cfg.get('source_weights') != {'linear_pair': 1.0}:
        raise ValueError('Only Stage 1 linear-pair training is supported')
    if cfg.get('role_weights') != {'L_edit': 1.0} or cfg.get('keep_fraction') != 0:
        raise ValueError('The Stage 1 recipe requires only L_edit')
    if cfg.get('condition_protocol') != CONDITION_PROTOCOL:
        raise ValueError('Condition protocol must match the Stage 1 checkpoint')
    for key in ('width', 'height', 'num_steps', 'edit_steps', 'lora_rank',
                'global_batch', 'save_every', 'max_steps', 'updates_per_pair'):
        if not isinstance(cfg.get(key), int) or cfg[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if cfg['width'] % 8 or cfg['height'] % 8:
        raise ValueError('Canvas dimensions must be divisible by eight')
    if cfg['cfg_norm'] != 'none' or float(cfg['cfg_scale']) != 4 or float(cfg['image_cfg_scale']) != 1:
        raise ValueError('Native guidance must match the Stage 1 recipe')
    lo, hi = map(float, cfg['cfg_interval'])
    if not 0 <= lo < hi <= 1 or not 0 < float(cfg['t_eps']) < 1:
        raise ValueError('Invalid native sampling interval')
    times = list(map(float, cfg['state_times']))
    probs = list(map(float, cfg['state_time_probs']))
    if times != [float(cfg['read_time'])] or not 0 < times[0] < 1 or probs != [1.0]:
        raise ValueError('Stage 1 uses exactly one read time')
    if int(cfg['lookahead_steps']) != 2 or float(cfg['timestep_shift']) <= 0:
        raise ValueError('Invalid preview or timestep schedule')
    if float(cfg['lr']) <= 0 or not 0 < float(cfg['lr_final']) <= float(cfg['lr']):
        raise ValueError('Invalid learning-rate schedule')
    if int(cfg['warmup_steps']) < 0 or float(cfg['gradient_clip']) <= 0:
        raise ValueError('Invalid training schedule')
    for key in ('terminal', 'semantic_weights', 'recovery_prob'):
        if cfg.get(key):
            raise ValueError(key + ' is not part of the Stage 1 recipe')


def stage_updates(records, cfg, global_batch=None):
    records = int(records)
    batch = int(global_batch or cfg['global_batch'])
    if records < 1 or batch < 1:
        raise ValueError('Training needs records and a positive global batch')
    return min(int(cfg['max_steps']), math.ceil(int(cfg['updates_per_pair']) * records / batch))
