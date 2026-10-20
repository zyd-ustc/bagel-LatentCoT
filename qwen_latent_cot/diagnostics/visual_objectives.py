"""Separate semantic, shared-response, complete flow and text-keep objectives."""
import torch
from .native import teacher_forcing, free_caption
from .memory import consume_bank
from .response import capture_queries, response_loss
from .flow_training import conditional_gen
from .numerics import autocast_for
from .data import score_facts


def semantic_loss(runtime, row, bank):
    instruction = runtime.config['label_instructions']['feedback' if row.get('feedback') else 'observe']
    with autocast_for(runtime.device):
        return teacher_forcing(runtime.bundle, runtime.prompt_ids(instruction),
                               runtime.answer_ids(row['caption']), runtime.caption_prefix(bank), bank.slots)[0]


@torch.no_grad()
def semantic_result(runtime, row, bank):
    instruction = runtime.config['label_instructions']['feedback' if row.get('feedback') else 'observe']
    with autocast_for(runtime.device):
        loss = semantic_loss(runtime, row, bank)
        text, ids = free_caption(runtime.bundle, runtime.prompt_ids(instruction),
                                 runtime.caption_prefix(bank), bank.slots,
                                 next((b for b in runtime.config.get('output_token_budgets',[runtime.config['max_caption_tokens']]) if b>=row.get('output_tokens',0)),runtime.config.get('output_token_budgets',[runtime.config['max_caption_tokens']])[-1]))
    return dict(nll=float(loss), text=text, generated_ids=ids, **score_facts(text, row['facts']))


@torch.no_grad()
def state_probe(runtime, state):
    if runtime.config['response_position_reserve'] != runtime.writer.slots:
        raise ValueError('v1 response teacher is compressed to native 16 slots; reserve must equal slots')
    with capture_queries(runtime, state.flow, runtime.config['response_layers'], runtime.writer.slots,
                         runtime.config['response_query_count']) as tensors:
        conditional_gen(runtime, state, capture=False)
    return dict(layers=tensors, prompt_positions=state.positions, metadata=state.metadata)


def bank_response(runtime, bank, teacher_bank, probe):
    teacher = consume_bank(runtime.decoder, teacher_bank, probe['prompt_positions'])
    condition = {index: (teacher[index].keys, teacher[index].values)
                 for index in runtime.config['response_layers']}
    return response_loss(runtime, bank, condition, probe,
                         filter_weak_teacher=runtime.config.get('response_filter_weak_teacher', True))


def training_objective(student, row, state, stage, *, policy=None, feedback_provider=None):
    """One task-aware objective. Labels and flow legality do not follow x0 presence."""
    from .curriculum import NoisePolicy
    runtime=student.runtime;cfg=runtime.config
    policy=policy or NoisePolicy(cfg)
    names=('sem','teacher','flow');values={k:torch.zeros((),device=runtime.device) for k in names}
    available=policy.masks(row,state.t,teacher_available=student.text_teacher is not None)
    if row['task']=='observe':
        if not available.sem:raise ValueError('observation caption outside calibrated noise range')
        with torch.no_grad():source=conditional_gen(runtime,state).source
        bank=student.write(source,state.t)
        values['sem']=semantic_loss(runtime,row,bank)
        return values,dict(passes=1,feedback_updates=0,gen_calls=1,rounds=[],masks=available.asdict())
    if stage == 'C2-feedback':
        if row['task'] != 'feedback' or not available.sem:
            raise ValueError('C2 feedback requires a valid correction or hold label')
        with torch.no_grad():
            source = conditional_gen(runtime, state).source
        bank = student.write(source, state.t, prompt=state.prompt_kv)
        values['sem'] = semantic_loss(runtime, row, bank)
        return values, dict(passes=1, feedback_updates=0, gen_calls=1, rounds=[],
                            masks=available.asdict(), target_satisfied=row['feedback']['satisfied'])
    raise ValueError('Generation objectives use interaction full-loop SFT')
