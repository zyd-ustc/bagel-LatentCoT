"""Indexed-task checkpoints. Weight initialization is separate from exact resume."""
import random
import numpy as np
import torch
from .record import load_checkpoint, validate_checkpoint
from .data import file_hash

FORMAT_VERSION = 3
STAGES = ('C2-observe', 'C2-feedback', 'C3-loop')
RESUME_KEYS = ('effective_batch', 'seed', 'loss_weights', 'learning_rate', 'weight_decay',
               'gradient_clip', 'cross_round_gradients', 'train_writer', 'source_layers',
               'feedback_updates', 'online_feedback', 'noise_curriculum', 'noise_calibration',
               'task_mix', 'output_token_budgets', 'context_token_limit', 'label_instructions',
               'sampler_steps', 'timestep_shift', 'resolution', 'activation_checkpointing')


def initialize_student(runtime, writer_init=None, visual_init=None, *, teacher_path=None):
    from .visual import VisualStudent
    from .memory import MemoryWriter
    if writer_init: load_checkpoint(writer_init, runtime, ('C1', 'C1-read'))
    teacher = None
    if teacher_path:
        saved = torch.load(teacher_path, map_location=runtime.device, weights_only=True)
        validate_checkpoint(saved, runtime, ('C1', 'C1-read'))
        expected = runtime.config.get('text_teacher_sha256')
        if expected and file_hash(teacher_path) != expected: raise ValueError('text teacher SHA256 changed')
        teacher = MemoryWriter(runtime.decoder, 'C', runtime.writer.slots, runtime.writer.rank)
        teacher.load_state_dict(saved['adapters'], strict=True)
    student = VisualStudent(runtime, teacher)
    student.text_teacher_hash = file_hash(teacher_path) if teacher_path else None
    if visual_init:
        saved = torch.load(visual_init, map_location=runtime.device, weights_only=True)
        validate_checkpoint(saved, runtime, (*STAGES, 'C2-joint'))
        # Historical checkpoints are weights only. No optimizer or teacher migration.
        if saved['config'].get('visual_architecture') != student.visual.architecture:
            raise ValueError('visual architecture differs')
        student.visual.load_state_dict(saved['visual'], strict=True)
        if not writer_init: student.writer.load_state_dict(saved['adapters'], strict=True)
    return student


def save_visual_checkpoint(path, student, step, stage, parent=None, optimizer=None, schedule=None, best=None):
    runtime = student.runtime
    if stage not in STAGES: raise ValueError('unknown training stage')
    n = np.random.get_state()
    rng = dict(python=random.getstate(), torch=torch.get_rng_state(),
               numpy=dict(kind=n[0], keys=n[1].tolist(), position=n[2], has_gauss=n[3], cached_gauss=n[4]))
    if runtime.device.type != 'cpu':
        rng['accelerator_selected'] = getattr(torch, runtime.device.type).get_rng_state(runtime.device)
    saved = dict(format_version=FORMAT_VERSION, adapters=student.writer.state_dict(), visual=student.visual.state_dict(),
                 text_teacher=student.text_teacher.state_dict() if student.text_teacher is not None else None,
                 text_teacher_hash=student.text_teacher_hash, stage=stage, step=step, config=runtime.config,
                 topology='C', slots=student.writer.slots, rank=student.writer.rank, depth=len(runtime.decoder.layers),
                 seed_ids=runtime.seed_ids, native_files=runtime.native_files, parent=parent,
                 parent_hash=file_hash(parent) if parent else None,
                 data_hashes=getattr(runtime, 'data_hashes', None),
                 optimizer=optimizer.state_dict() if optimizer is not None else None,
                 schedule=schedule, best=best, rng=rng)
    torch.save(saved, path)


def load_visual_checkpoint(path, student, *, optimizer=None, resume=False, expected_stage=None):
    runtime = student.runtime
    saved = torch.load(path, map_location=runtime.device, weights_only=True)
    validate_checkpoint(saved, runtime, STAGES)
    if saved.get('format_version') != FORMAT_VERSION:
        raise ValueError('exact checkpoint loading requires format 3; use weight initialization for historical weights')
    if saved.get('text_teacher_hash') != student.text_teacher_hash:
        raise ValueError('independent text teacher provenance differs')
    teacher = student.text_teacher.state_dict() if student.text_teacher is not None else None
    stored = saved['text_teacher']
    if (teacher is None) != (stored is None) or (teacher is not None and
       (teacher.keys()!=stored.keys() or any(not torch.equal(value,stored[name]) for name,value in teacher.items()))):
        raise ValueError('frozen text teacher tensors differ')
    if resume:
        if saved['stage'] != expected_stage or saved['optimizer'] is None or saved['schedule'] is None:
            raise ValueError('resume requires matching stage, optimizer and indexed sampler state')
        for key in RESUME_KEYS:
            if saved['config'].get(key) != runtime.config.get(key): raise ValueError('exact resume configuration differs: '+key)
        if saved.get('data_hashes') != getattr(runtime, 'data_hashes', None): raise ValueError('resume corpus fingerprint changed')
    student.writer.load_state_dict(saved['adapters'], strict=True); student.visual.load_state_dict(saved['visual'], strict=True)
    if resume:
        optimizer.load_state_dict(saved['optimizer'])
        rng = saved['rng']; random.setstate(rng['python']); torch.set_rng_state(rng['torch'].cpu())
        n = rng['numpy']; np.random.set_state((n['kind'], np.array(n['keys'],dtype=np.uint32), n['position'],n['has_gauss'],n['cached_gauss']))
        if 'accelerator_selected' in rng:
            getattr(torch,runtime.device.type).set_rng_state(rng['accelerator_selected'].cpu(),runtime.device)
    return saved
