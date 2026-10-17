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


def generation_caption(row, config):
    if row.get('label_kind') == 'feedback':
        return row['feedback']['correction'] or None
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


def training_objective(student, row, state, stage, *, policy=None, feedback_provider=None):
    """One task-aware objective. Labels and flow legality do not follow x0 presence."""
    from .curriculum import NoisePolicy
    from .loop_training import memory_loop
    from .flow_training import flow_mse
    from .explicit_feedback import Feedback, shifted_layout
    from dataclasses import replace
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
    if not any(available.asdict().values()):raise ValueError('sample has no legal supervision at this noise level')
    updates=cfg.get('feedback_updates',2 if stage=='C3-loop' else 1)
    if updates>1 and row['task']=='feedback' and not available.flow and feedback_provider is None:
        raise ValueError('later feedback rounds require online feedback; no source-image reconstruction fallback')
    loop=memory_loop(student,state,updates,checkpoint_activations=cfg.get('activation_checkpointing',True))
    terms={name:[] for name in names};reports=[];gen_calls=loop.gen_calls
    for index,(bank,prediction) in enumerate(zip(loop.banks,loop.exits)):
        target_row=dict(row);fresh=False
        if index and row['task']=='feedback':
            if feedback_provider:
                feedback=feedback_provider(state,loop.banks[index-1]);fresh=True
                target_row.update(feedback=__import__('dataclasses').asdict(feedback),
                                  annotation=dict(origin=feedback.origin,scope='preview'),
                                  caption=feedback.correction if not feedback.satisfied else 'No change is needed.',facts=[])
            else:target_row.pop('feedback',None)
        feedback=Feedback(**target_row['feedback']) if target_row.get('feedback') else None
        mask=policy.masks(target_row,state.t,pass_index=index,fresh=fresh,
                          teacher_available=student.text_teacher is not None)
        if mask.sem and cfg['loss_weights']['sem']:terms['sem'].append(semantic_loss(runtime,target_row,bank))
        cached=row.get('teacher') if index==0 else None
        error=None
        if mask.teacher and cfg['loss_weights']['teacher']:
            with torch.no_grad():
                if cached:
                    saved=torch.load(cached,map_location=runtime.device,weights_only=True)
                    expected=(state.metadata['x_t_hash'],state.metadata['prompt'],state.t,student.text_teacher_hash,
                              'conditional_pre_cfg',feedback.correction,feedback.satisfied)
                    actual=tuple(saved[k] for k in ('x_t_hash','prompt','t','text_teacher_hash','basis','correction','satisfied'))
                    if actual!=expected or saved.get('condition_hashes',{})!=state.metadata.get('condition_hashes',{}):raise ValueError('cached teacher differs from state, feedback or text checkpoint')
                    target=saved['velocity']
                elif feedback.satisfied:
                    layout=replace(state,flow=shifted_layout(state.flow,bank.slots))
                    target=conditional_gen(runtime,layout,capture=False).velocity
                else:
                    teacher=student.teacher_bank(feedback.correction,state)
                    target=conditional_gen(runtime,state,teacher,capture=False).velocity
            error=(prediction.velocity.float()-target.float()).square().mean()
            terms['teacher'].append(error);gen_calls+=int(not cached)
        if mask.flow and cfg['loss_weights']['flow']:terms['flow'].append(flow_mse(prediction,state))
        reports.append(dict(pass_index=index+2,feedback_origin=feedback.origin if feedback else None,
                            correction=feedback.correction if feedback else None,
                            satisfied=feedback.satisfied if feedback else None,masks=mask.asdict(),
                            teacher_velocity_mse=float(error.detach()) if error is not None else None))
    for name,items in terms.items():
        if items:values[name]=torch.stack(items).mean()
    return values,dict(passes=updates+1,feedback_updates=updates,gen_calls=gen_calls,rounds=reports,masks=available.asdict())


def online_feedback_provider(runtime):
    """Build fresh BAGEL text feedback for the preceding student's full preview."""
    from .explicit_feedback import reflect
    from .schedule import native_suffix
    from ..bagel.modeling.native_image_transform import ImageTransform
    cfg = runtime.config
    transform = ImageTransform(cfg.get('vit_max_image_size', 980), cfg.get('vit_min_image_size', 224),
                               runtime.model.vit_patch_size)

    @torch.no_grad()
    def provide(state, bank):
        velocity = runtime.gen_arm(state.x_t, state.t, state.flow, bank, state.lengths, state.positions)[1]
        preview = runtime.image(runtime.continue_once(state.x_t, state.t, velocity, state.flow,
                                                       native_suffix(cfg, state.t)))
        return reflect(runtime, preview, state.metadata['prompt'], transform,
                       cfg.get('max_reflection_tokens', 384))
    return provide
