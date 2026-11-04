"""Text-only entry: probe, diagnose, continue or roll back inside the same ODE."""

import re
import torch
from .diagnosis import (
    decompose, affirmative_constraints, entailment_cap, semantic_correction_condition,
    und_score_image,
)
from . import KIND, TRACE_KIND
from .field import (
    NativeCFG, phase_grid, segment_times, integrate, rollback, protocol,
    total_velocity, training_free_velocity,
)


def score(rt, image, questions):
    return entailment_cap(und_score_image(rt, image, questions), questions)


def choose_arm(failed):
    # Attribute-only diagnoses can preserve more of the original composition.
    attribute = r"\b(colou?r|red|blue|green|yellow|black|white|material|wooden|metallic|spelled|spelling|legible)\b|颜色|材质|拼写"
    structural = r"\b(count|exactly|distinct|fused|behind|front|holding|there|entities)\b|数量|缺失|遮挡|融合"
    return "shallow" if all(re.search(attribute, q, re.I) and not re.search(structural, q, re.I)
                            for q in failed) else "deep"


@torch.no_grad()
def probe(rt, raw, seed, cfg, question_record=None):
    native = NativeCFG(rt, raw, cfg)
    grid = phase_grid(rt, cfg)
    generator = torch.Generator(device=rt.device).manual_seed(seed)
    noise = torch.randn((1,3,rt.height,rt.width), device=rt.device, generator=generator)*rt.noise_scale()
    history = {}
    read = cfg["read_time"]
    z = integrate(noise.clone(), segment_times(grid, 0., read), native, capture=history)
    head_calls = native.calls
    # Two coarse Heun intervals; final Euler avoids evaluating the t=1 singularity.
    # Choose the interior node from the actual native grid (also supports dynamic shift).
    tail_nodes = segment_times([float(t) for t in rt.times()], read, 1.)
    middle = tail_nodes[len(tail_nodes)//2]
    if not read < middle < 1:
        raise ValueError("Native grid has no interior lookahead node after read_time")
    preview = integrate(z.clone(), [read, middle, 1.], native)
    preview_calls = native.calls-head_calls
    if question_record is None:
        questions, decomposition = decompose(rt, raw, cfg)
    else:
        if question_record["raw_prompt"] != raw:
            raise ValueError("Question cache belongs to a different prompt")
        questions, decomposition = question_record["questions"], question_record
    if not questions:
        raise ValueError("Diagnosis requires nonempty atomic questions")
    if question_record is not None and "clauses" in question_record:
        if len(question_record["clauses"]) != len(questions):
            raise ValueError("Question cache must contain one clause per question")
    preview_score = score(rt, preview, questions)
    failed = [q for q, s in zip(questions, preview_score["questions"])
              if s["p_yes"] < cfg["pass_threshold"]]
    if question_record is not None and "clauses" in question_record:
        clauses = [c for q,c in zip(questions,question_record["clauses"]) if q in failed]
        rejected = []
    else:
        clauses, rejected = affirmative_constraints(failed)
    condition = semantic_correction_condition(raw, clauses)
    record = dict(kind=KIND, raw_prompt=raw, seed=seed, protocol=protocol(rt,cfg),
                  questions=questions, decomposition=decomposition, preview_score=preview_score,
                  failed_questions=failed, clauses=clauses, rejected_questions=rejected,
                  condition=condition, early_exit=not failed,
                  nfe=dict(head=head_calls, preview=preview_calls,
                           correction=0, native_tail=0, comparison=0,
                           sketch_prefill=0, und_image_scores=len(questions),
                           und_decomposition=int(question_record is None)))
    return native, grid, noise, history, z, preview, record


@torch.no_grad()
def native_trace(rt, raw, seed, cfg, question_record=None):
    """Capture a candidate trajectory; selection into a training pair is explicit."""
    native, grid, noise, history, z, preview, record = probe(rt,raw,seed,cfg,question_record)
    before = native.calls
    final = integrate(z, segment_times(grid,cfg["read_time"],1.), native, capture=history)
    record["nfe"]["native_tail"] = native.calls-before
    record["final_score"] = score(rt,final,record["questions"])
    record["nfe"]["und_image_scores"] += len(record["questions"])
    return dict(kind=TRACE_KIND, record=record, noise=noise.cpu(), preview=preview.cpu(),
                final=final.cpu(), history={str(t):v.cpu() for t,v in history.items()},
                valid=torch.ones((1,1,rt.height,rt.width)))


@torch.no_grad()
def run(rt, raw, seed, cfg, *, corrector=None, question_record=None, compare_native=False):
    native, grid, noise, history, read_state, preview, record = probe(rt,raw,seed,cfg,question_record)
    nfe = record["nfe"]
    images = {"preview":preview}
    if record["early_exit"]:
        before = native.calls
        result = integrate(read_state, segment_times(grid,cfg["read_time"],1.),native)
        nfe["native_tail"] = native.calls-before
        if compare_native:
            images["native"] = result  # Same continuation, not a second generation.
        record.update(arm=None, tau=None, route="native_passthrough")
    else:
        arm = choose_arm(record["failed_questions"])
        tau, handoff = cfg[arm+"_time"], cfg["handoff_time"]
        state = rollback(rt,history[tau],noise,tau,arm,seed+1000,cfg)
        if corrector is None:
            prefix = rt.prefix(record["condition"])
            velocity = lambda z,t: training_free_velocity(
                native,prefix,z,t,tau,handoff,cfg["edit_scale"])
        else:
            sketch = corrector.prefill(preview,record["condition"])
            nfe["sketch_prefill"] = 1
            before_sketch = corrector.calls
            velocity = lambda z,t: total_velocity(native,corrector,z,t,tau,handoff,sketch)
        before = native.calls
        state = integrate(state,segment_times(grid,tau,handoff),velocity)
        nfe["correction"] = native.calls-before
        if corrector is not None:
            nfe["correction"] += corrector.calls-before_sketch
        before = native.calls
        result = integrate(state,segment_times(grid,handoff,1.),native)
        nfe["native_tail"] = native.calls-before
        record.update(arm=arm,tau=tau,handoff=handoff,perturbation_seed=seed+1000,
                      route="trained_residual" if corrector is not None else "training_free")
        if compare_native:
            before = native.calls
            images["native"] = integrate(read_state.clone(),segment_times(grid,cfg["read_time"],1.),native)
            nfe["comparison"] = native.calls-before
    images["result"] = result
    record["scores"] = {"result":score(rt,result,record["questions"])}
    nfe["und_image_scores"] += len(record["questions"])
    if "native" in images:
        if record["early_exit"]:
            record["scores"]["native"] = record["scores"]["result"]
        else:
            record["scores"]["native"] = score(rt,images["native"],record["questions"])
            nfe["und_image_scores"] += len(record["questions"])
    record["generation_calls"] = sum(nfe[k] for k in ("head","preview","correction","native_tail"))
    return images, record
