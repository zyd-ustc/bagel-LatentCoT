"""Strict versioned checkpoints including independent text teacher and exact resume state."""
import random
import numpy as np
import torch
from .record import load_checkpoint, FrozenGuard
from .data import file_hash

FORMAT_VERSION = 2
INTERFACE = ('visual_architecture', 'visual_rank', 'label_interface_version', 'label_instructions',
             'visual_source_policy', 'source_prompt_policy', 'allow_target_prompt',
             'response_position_reserve', 'response_teacher',
             'observation_response_teacher',
             'response_layers', 'response_query_count', 'response_energy_floor')
STAGES = ('C2-observe', 'C2-joint', 'C3-loop')


def verify_hash(path, expected):
    actual = file_hash(path)
    if not expected or actual != expected:
        raise ValueError('text teacher checkpoint SHA256 differs from locked provenance')
    return actual


def initialize_student(runtime, text_path, parent=None):
    from .visual import VisualStudent
    verify_hash(text_path, runtime.config['text_teacher_sha256'])
    load_checkpoint(text_path, runtime, ('C1', 'C1-read'))
    student = VisualStudent(runtime)
    student.text_teacher_hash = file_hash(text_path)
    if parent:
        load_visual_checkpoint(parent, student)
    return student


def save_visual_checkpoint(path, student, step, stage, parent, optimizer=None, schedule=None, best=None):
    runtime = student.runtime
    if stage not in STAGES:
        raise ValueError('unknown visual checkpoint stage')
    numpy_state = np.random.get_state()
    rng = dict(python=random.getstate(), torch=torch.get_rng_state(),
               numpy=dict(kind=numpy_state[0], keys=numpy_state[1].tolist(),
                          position=numpy_state[2], has_gauss=numpy_state[3], cached_gauss=numpy_state[4]))
    accelerator = getattr(torch, runtime.device.type, None)
    if runtime.device.type != 'cpu':
        rng['accelerator'] = accelerator.get_rng_state_all()
    saved = dict(format_version=FORMAT_VERSION, adapters=student.writer.state_dict(),
                 visual=student.visual.state_dict(), text_teacher=student.text_teacher.state_dict(),
                 text_teacher_hash=student.text_teacher_hash, stage=stage, step=step,
                 config=runtime.config, topology='C', slots=student.writer.slots, rank=student.writer.rank,
                 depth=len(runtime.decoder.layers), seed_ids=runtime.seed_ids,
                 native_files=runtime.native_files, parent=parent,
                 parent_hash=file_hash(parent) if parent else None,
                 trainable_parameter_names=[name for name, p in student.named_parameters() if p.requires_grad],
                 frozen_parameter_names=[name for name, _ in runtime.model.named_parameters()],
                 data_hashes=getattr(runtime, 'data_hashes', None),
                 optimizer=optimizer.state_dict() if optimizer is not None else None,
                 schedule=schedule, best=best, rng=rng)
    torch.save(saved, path)


def load_visual_checkpoint(path, student, *, optimizer=None, resume=False, expected_stage=None):
    runtime = student.runtime
    saved = load_checkpoint(path, runtime, STAGES)
    if saved.get('format_version') != FORMAT_VERSION:
        raise ValueError('visual checkpoint requires explicit format version 2')
    for key in INTERFACE:
        if saved['config'].get(key) != runtime.config.get(key):
            raise ValueError('visual interface mismatch: '+key)
    if saved.get('text_teacher_hash') != student.text_teacher_hash:
        raise ValueError('independent text teacher provenance changed')
    student.visual.load_state_dict(saved['visual'], strict=True)
    student.text_teacher.load_state_dict(saved['text_teacher'], strict=True)
    student.text_teacher.eval().requires_grad_(False)
    if resume:
        if saved['stage'] != expected_stage or not saved.get('optimizer') or not saved.get('schedule'):
            raise ValueError('resume requires matching stage, optimizer and global sampling schedule')
        for key in ('effective_batch', 'seed', 'loss_weights', 'learning_rate', 'weight_decay',
                    'cross_round_gradients', 'train_writer', 'replay_fraction'):
            if saved['config'][key] != runtime.config[key]:
                raise ValueError('exact resume configuration differs: '+key)
        if saved.get('data_hashes') != getattr(runtime, 'data_hashes', None):
            raise ValueError('resume visual/text dataset hashes changed')
        optimizer.load_state_dict(saved['optimizer'])
        rng = saved['rng']; random.setstate(rng['python']); torch.set_rng_state(rng['torch'].cpu())
        n = rng['numpy']; np.random.set_state((n['kind'], np.array(n['keys'], dtype=np.uint32),
                                              n['position'], n['has_gauss'], n['cached_gauss']))
        if 'accelerator' in rng:
            getattr(torch, runtime.device.type).set_rng_state_all([value.cpu() for value in rng['accelerator']])
    return saved
