"""Separate semantic, shared-response, complete flow and text-keep objectives."""
import torch
from .native import teacher_forcing, free_caption
from .memory import consume_bank
from .response import capture_queries, response_loss
from .flow_training import conditional_gen
from .numerics import autocast_for
from .data import score_facts


def semantic_loss(runtime, row, bank):
    instruction = runtime.config['label_instructions'][row.get('label_kind', 'target_clean')]
    with autocast_for(runtime.device):
        return teacher_forcing(runtime.bundle, runtime.prompt_ids(instruction),
                               runtime.answer_ids(row['caption']), runtime.caption_prefix(bank), bank.slots)[0]


@torch.no_grad()
def semantic_result(runtime, row, bank):
    instruction = runtime.config['label_instructions'][row.get('label_kind', 'target_clean')]
    with autocast_for(runtime.device):
        loss = semantic_loss(runtime, row, bank)
        text, ids = free_caption(runtime.bundle, runtime.prompt_ids(instruction),
                                 runtime.caption_prefix(bank), bank.slots,
                                 runtime.config['max_caption_tokens'])
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
    return response_loss(runtime, bank, condition, probe)


def generation_caption(row, config):
    if row.get('label_kind', 'target_clean') == 'target_clean':
        return row['target_caption']
    # Observation is UND-only by default. Enabling its response teacher requires
    # an explicit reviewed label AND explicit confirmation of its generation use.
    if (config.get('observation_response_teacher') and row.get('confirmed_label')
        and row.get('observation_generation_verified') is True):
        return row['observation_caption']
    return None


def text_keep(student, row, probe):
    runtime = student.runtime
    bank = student.write(runtime.text(row['caption'])[0], None, visual=False)
    teacher_bank = student.teacher_bank(row['caption'])
    caption = semantic_loss(runtime, dict(row, label_kind='target_clean'), bank)
    response, metrics = bank_response(runtime, bank, teacher_bank, probe)
    return caption+response, dict(caption=float(caption.detach()), response=float(response.detach()),
                                 provides_gradient=any(p.requires_grad for p in student.writer.parameters()),
                                 layers=metrics)
