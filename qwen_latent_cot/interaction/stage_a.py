"""Stage A v2: native Teacher residuals, predictive canvases and Memory semantics."""
from dataclasses import dataclass, replace
import random
import re
import torch
from .loop import image_entrance, canvas_entrance
from ..diagnostics.flow_training import conditional_gen
from ..diagnostics.native import teacher_forcing
from ..diagnostics.numerics import autocast_for


@dataclass(frozen=True)
class PromptSplit:
    full: str
    partial: str
    removed: str
    units: tuple
    dropped: tuple
    reason: str


def prompt_units(text):
    """Sentence/semicolon boundaries outside quotes and parentheses; never commas.

    Keep exact spans. A single sentence without safe boundaries stays whole.
    Decimal points, initials and common abbreviations are not boundaries.
    """
    units, start, stack, quote = [], 0, [], None
    pairs = {')': '(', ']': '[', '}': '{'}
    abbreviations = {'mr', 'mrs', 'ms', 'dr', 'prof', 'st', 'vs', 'etc', 'e.g', 'i.e'}
    for i, char in enumerate(text):
        if char in ('"', '“', '”'):
            quote = None if quote else char
        elif not quote:
            if char in '([{': stack.append(char)
            elif char in pairs and stack and stack[-1] == pairs[char]: stack.pop()
        if quote or stack or char not in '.!?;。！？；': continue
        if i+1 < len(text) and not text[i+1].isspace(): continue
        if char == '.':
            last = re.search(r'([\w.]+)$', text[start:i])
            word = last.group(1) if last else ''
            if word.lower() in abbreviations or (len(word) == 1 and word.isalpha()): continue
        units.append(text[start:i+1]); start = i+1
    if text[start:].strip(): units.append(text[start:])
    elif start < len(text) and units: units[-1] += text[start:]
    return tuple(units)


def split_prompt(text, seed, *, drop_fraction=.4, full_probability=.2, min_words=30):
    if not text.strip() or not 0 < drop_fraction < 1 or not 0 <= full_probability <= 1:
        raise ValueError('Invalid Stage A prompt or masking probabilities')
    units = prompt_units(text)
    rng = random.Random(seed)
    reason = ('short' if len(text.split()) < min_words else 'unsplittable' if len(units) < 2 else
              'full_mix' if rng.random() < full_probability else 'clause_drop')
    if reason != 'clause_drop': return PromptSplit(text, text, '', units, (), reason)
    # Keep the first sentence as a coherent scene anchor. Drop whole later units.
    choices = list(range(1, len(units))); rng.shuffle(choices)
    goal = len(text.split())*drop_fraction
    dropped, count = [], 0
    for index in choices:
        length = len(units[index].split())
        if not dropped or abs(count+length-goal) < abs(count-goal):
            dropped.append(index); count += length
    dropped = tuple(sorted(dropped))
    partial = ''.join(unit for i,unit in enumerate(units) if i not in dropped).strip()
    removed = ''.join(units[i] for i in dropped).strip()
    return PromptSplit(text, partial, removed, units, dropped, reason)


@torch.no_grad()
def training_state(student, row, t, noise_seed, split_seed):
    cfg, runtime = student.config, student.runtime
    full = row['caption']
    split = split_prompt(full, split_seed, drop_fraction=cfg['clause_drop_fraction'],
                         full_probability=cfg['full_prompt_probability'], min_words=cfg['min_prompt_words'])
    # All GEN caches, including baseline and CFG, derive ONLY from partial.
    with student.native_context():
        state = runtime.training_state(dict(row, prompt=split.partial), t, noise_seed)
        teacher_flow, lengths, positions = runtime.flow_inputs(full)
    teacher_state = replace(state, flow=teacher_flow, lengths=tuple(lengths), positions=positions)
    teacher = conditional_gen(runtime, teacher_state, capture=False, native_scope=student.native_context).velocity
    eta = cfg['teacher_mix']
    state.correction_target = (eta*teacher.float()+(1-eta)*state.flow_target()).detach()
    state.prompt_kv = runtime.text(full)[0]  # Reader-only; never replaces state.flow.
    state.metadata.update(full_prompt=full, removed_clauses=split.removed,
                          prompt_split_reason=split.reason, split_seed=split_seed,
                          dropped_units=list(split.dropped))
    return state, split


def semantic_loss(student, bank, split):
    # These are deleted TEXT clauses, not claims about observed visual omissions.
    target = split.removed or split.full
    instruction = ('Recover the withheld image-description sentences from the memory. Return only those sentences.'
                   if split.removed else 'Describe the intended image from the memory.')
    runtime = student.runtime
    with student.adapters.scope('writer', False), student.adapters.scope('consumer', False), autocast_for(runtime.device):
        return teacher_forcing(runtime.bundle, runtime.prompt_ids(instruction), runtime.answer_ids(target),
                               runtime.caption_prefix(bank), bank.slots)[0]


@torch.no_grad()
def memory_controls(student, state, output, donor_bank):
    """Fix recipient x/t/prompt/hidden/LoRA; vary ONLY last-pass Memory."""
    with autocast_for(student.runtime.device):
        previous = output.baseline if len(output.exits) == 1 else output.exits[-2]
        entrance = student.bridge(image_entrance(student.runtime, state),
                                  canvas_entrance(student.runtime, state, previous.velocity))
    kwargs = dict(hidden_input=entrance, capture=False, detach_source=True,
                  adapter_scope=student.adapters.context('consumer'),
                  velocity_baseline=previous.velocity, velocity_scale=student.bridge.vel_scale[-1],
                  memory_gates=student.bridge.memory_gates, memory_gate_scale=student.bridge.gate_scale,
                  native_scope=student.native_context)
    off = conditional_gen(student.runtime, state, output.banks[-1], masked=True, **kwargs)
    shuffled = conditional_gen(student.runtime, state, donor_bank, **kwargs)
    target = state.correction_target.detach()
    return ((off.velocity.float()-target).square().mean(),
            (shuffled.velocity.float()-target).square().mean())


def losses(student, state, output, split, *, shuffled_error=None):
    cfg = student.config
    flow = correction_loss(output, state, cfg)
    sem = sum(w*semantic_loss(student, bank, split) for w,bank in zip(cfg['round_loss_weights'],output.banks))
    correct = flow.round_flow[-1]
    contrast = (torch.relu(correct-shuffled_error.detach()+cfg['memory_margin'])
                if shuffled_error is not None else correct.new_zeros(()))
    total = flow.total+cfg['semantic_weight']*sem+cfg['memory_contrast_weight']*contrast
    return total, flow, sem, contrast


@dataclass
class CorrectionLoss:
    total: torch.Tensor
    round_flow: list
    round_cos: list


def residual_correction_loss(pred_vel, base_vel, target_vel, cos_weight=0.1):
    import torch.nn.functional as F
    residual = (pred_vel.float()-base_vel.detach().float()).reshape(1, -1)
    target = (target_vel.detach().float()-base_vel.detach().float()).reshape(1, -1)
    mse = F.mse_loss(residual, target)
    # Each TrainingState contains one image. Ignore direction for a zero target.
    # Zero residual starts with MSE; enable direction as its norm reaches 1e-2.
    # Detach the ramp so shrinking the residual cannot reduce loss through the weight.
    direction_weight = (residual.detach().norm(dim=-1)/1e-2).clamp(max=1.)
    cos = (direction_weight*(1-F.cosine_similarity(residual, target, dim=-1, eps=1e-2))).mean()
    cos = torch.where(target.square().sum() > 0, cos, cos.new_zeros(()))
    return mse+cos_weight*cos, mse, cos


def correction_loss(output, state, config):
    total, mses, cosines = 0., [], []
    previous = output.baseline.velocity
    for weight, value in zip(config['round_loss_weights'], output.exits):
        applied, mse, cosine = residual_correction_loss(value.velocity, previous, state.correction_target, config['cos_weight'])
        total = total+weight*applied
        mses.append(mse); cosines.append(cosine)
        previous = value.velocity.detach()
    return CorrectionLoss(total, mses, cosines)
