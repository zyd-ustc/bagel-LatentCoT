"""Phase 1A T0 OPD contract, gate and single-device training loop."""

import json
import math
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

from .cot_teacher import load_teacher_cache, sha256_file
from .memory_training import append_json
from .opd import opd_velocity_loss
from .opd_runtime import OPDRuntime


SCHEMA = "bagel-selfcot-opd-phase1a-v4"


def validate_opd_config(source, *, require_warmup=True):
    config = dict(source)
    memory_reader=config.get("memory_reader",{})
    reader_controls=dict(load_warmup=True,injection_gate_init=0.0,train_gate=True,
                         train_adapter=False,train_q_lora=False)
    if not isinstance(memory_reader,dict) or set(memory_reader)-set(reader_controls):
        raise ValueError("unknown OPD memory_reader controls")
    for key,value in reader_controls.items():
        if memory_reader.get(key,value)!=value:
            raise ValueError(f"Phase 1A.1a requires memory_reader.{key}={value}")
    required = ("model_path", "prompt_data", "teacher_cot_data", "output_dir")
    if require_warmup:
        required += ("reader_warmup_checkpoint", "reader_warmup_eval_json")
    for key in required:
        if not config.get(key):
            raise ValueError(f"missing {key}")
    paths = ("model_path", "prompt_data", "teacher_cot_data")
    if require_warmup:
        paths += ("reader_warmup_checkpoint", "reader_warmup_eval_json")
    for key in paths:
        if not Path(config[key]).exists():
            raise FileNotFoundError(f"{key}: {config[key]}")
    locked = dict(num_loop_tokens=8, memory_loop_start_layer=12,
                  memory_loop_end_layer=20, memory_init="prompt_hidden_uniform",
                  memory_writer_trainable=False, teacher_type="bagel_text_cot",
                  teacher_mode="t2i", cfg_text_scale=1.0, cfg_img_scale=1.0,
                  states_per_rollout=2, loss_type="velocity_mse",
                  o_adapter_rank=8, o_adapter_alpha=16)
    for key, value in locked.items():
        if config.get(key, value) != value:
            raise ValueError(f"Phase 1A T0 requires {key}={value}")
        config[key] = value
    forbidden = ("source_image", "target_image", "paired_data", "adapter_path",
                 "trainable_routes", "lambda_teacher", "lambda_dep_shuffle",
                 "lambda_dep_zero", "dependency_margin", "mask_prompt_kv_during_write",
                 "reader_o_enabled", "loop_distill", "memory_loop_repeat")
    if any(key in config for key in forbidden):
        raise ValueError("Phase 1A T0 forbids v2 pair/ranking/loop/mask controls")
    if float(config.get("injection_gate_init", 0.0)) != 0.0:
        raise ValueError("Phase 1A.1a requires exact zero injection gate initialization")
    if config.get("train_reader_adapter", False) or not config.get("train_gate", True):
        raise ValueError("Phase 1A.1a trains injection gates only")
    if config.get("train_q_lora",False):
        raise ValueError("Phase 1A.1a freezes native Q projections")
    for key, default in (("height",512),("width",512),("num_steps",50),
                         ("max_steps",5000),("save_steps",250),("eval_steps",250),
                         ("seed",42)):
        config.setdefault(key, default)
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["height"] % 16 or config["width"] % 16 or config["num_steps"] < 4:
        raise ValueError("invalid image resolution or NFE")
    if "max_prompts" in config and (isinstance(config["max_prompts"], bool)
            or not isinstance(config["max_prompts"], int) or config["max_prompts"] < 1):
        raise ValueError("max_prompts must be a positive integer")
    config.setdefault("timestep_shift", 3.0)
    config.setdefault("timestep_bucket_weights", [.5,.4,.1])
    weights = config["timestep_bucket_weights"]
    if len(weights) != 3 or any(not math.isfinite(float(v)) or float(v)<0 for v in weights) or sum(weights)<=0:
        raise ValueError("three positive-total timestep bucket weights required")
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("max_grad_norm", 1.0)
    if any(not math.isfinite(float(config[key])) or float(config[key]) <= 0
           for key in ("learning_rate","max_grad_norm","timestep_shift")):
        raise ValueError("learning rate and gradient clip must be positive")
    config.setdefault("device", "auto")
    config.setdefault("allow_field_only_debug", False)
    if not isinstance(config["allow_field_only_debug"], bool):
        raise ValueError("allow_field_only_debug must be boolean")
    if config["allow_field_only_debug"] and config["max_steps"] > 10:
        raise ValueError("field-only debug is limited to at most 10 steps")
    if require_warmup:
        from .reader_warmup import check_warmup_gate
        check_warmup_gate(config)
    return config


def load_training_records(config, *, tokenizer=None):
    source = [json.loads(line) for line in Path(config["prompt_data"]).read_text().splitlines() if line.strip()]
    source_by_id = {}
    seen_prompts = set()
    for index, row in enumerate(source):
        if "source_image" in row or "target_image" in row:
            raise ValueError("T0 only accepts T2I semantic prompts, not paired image data")
        key = str(row.get("prompt_id", row.get("id", index)))
        if (key in source_by_id or not isinstance(row.get("prompt"), str)
                or not row["prompt"].strip() or row["prompt"] in seen_prompts):
            raise ValueError("prompt source contains duplicate id/prompt or empty prompt")
        source_by_id[key] = row
        seen_prompts.add(row["prompt"])
    records = load_teacher_cache(config["teacher_cot_data"], tokenizer=tokenizer)
    for row in records:
        key = str(row["prompt_id"])
        if key not in source_by_id or source_by_id[key]["prompt"] != row["prompt"]:
            raise ValueError(f"teacher cache does not match prompt source: {key}")
        if source_by_id[key].get("category") != row["category"]:
            raise ValueError(f"teacher category does not match prompt source: {key}")
    if config.get("max_prompts"):
        records = records[:int(config["max_prompts"])]
    if not records:
        raise ValueError("no matching teacher records")
    return records


def check_teacher_baseline(config):
    path = config.get("teacher_baseline_json")
    if not path or not Path(path).is_file():
        raise ValueError("pre-training teacher baseline evidence JSON is required")
    gate = json.loads(Path(path).read_text())
    if (gate.get("schema") != SCHEMA or gate.get("kind") != "teacher_baseline"
            or gate.get("teacher_cache_sha256") != sha256_file(config["teacher_cot_data"])
            or gate.get("model_path") != str(Path(config["model_path"]).resolve())
            or gate.get("num_steps") != config["num_steps"]
            or gate.get("cfg") != 1.0):
        raise ValueError("teacher baseline evidence does not match this run")
    field = gate.get("field_rms_by_seed")
    prompt_ids = gate.get("prompt_ids")
    stable_field = (isinstance(prompt_ids, list) and len(prompt_ids) >= 8
        and len(set(prompt_ids)) == len(prompt_ids)
        and isinstance(field, dict) and len(field) >= 2 and all(
        not isinstance(value, bool) and isinstance(value, (int,float))
        and math.isfinite(value) and value > 1e-4 for value in field.values())
        )
    if config.get("allow_field_only_debug", False):
        if not stable_field:
            raise ValueError("field-only debug requires stable field difference on 8 prompts and 2 seeds")
        return {**gate, "training_gate": "field_only_debug_not_semantic_improvement"}

    evidence = gate.get("semantic_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("formal OPD training requires held-out semantic_evidence")
    heldout_path = Path(str(evidence.get("heldout_prompt_data", "")))
    report_path = Path(str(evidence.get("score_report", "")))
    if (not heldout_path.is_file() or not report_path.is_file()
            or evidence.get("heldout_prompt_sha256") != sha256_file(heldout_path)
            or evidence.get("score_report_sha256") != sha256_file(report_path)
            or evidence.get("scorer") not in ("geneval2", "core", "human")
            or evidence.get("model_path") != str(Path(config["model_path"]).resolve())
            or evidence.get("num_steps") != config["num_steps"]
            or evidence.get("cfg") != 1.0):
        raise ValueError("semantic evidence provenance is missing or mismatched")
    heldout_rows = [json.loads(line) for line in heldout_path.read_text().splitlines()
                    if line.strip()]
    train_rows = [json.loads(line) for line in Path(config["prompt_data"]).read_text().splitlines()
                  if line.strip()]
    heldout_prompts = [row.get("prompt") for row in heldout_rows]
    train_prompts = {row.get("prompt") for row in train_rows}
    if (len(heldout_prompts) < 8
            or any(not isinstance(prompt, str) or not prompt.strip()
                   for prompt in heldout_prompts)
            or len(heldout_prompts) != len(set(heldout_prompts))
            or any(prompt in train_prompts for prompt in heldout_prompts)
            or evidence.get("prompt_count") != len(heldout_prompts)):
        raise ValueError("semantic evidence needs at least 8 distinct held-out prompts")
    native = evidence.get("native_score")
    teacher = evidence.get("teacher_score")
    if (any(isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) for value in (native, teacher))
            or teacher <= native):
        raise ValueError("teacher semantic score must exceed native on held-out prompts")
    return {**gate, "training_gate": "heldout_semantic_gain",
            "semantic_teacher_minus_native": teacher - native}


def checkpoint(runtime, output, step, optimizer, config):
    stem = Path(output) / f"reader_step_{step:07d}"
    tensors = {name: parameter.detach().cpu().contiguous() for name, parameter
               in runtime.model.named_parameters() if parameter.requires_grad}
    if set(tensors) != set(runtime.trainable_names):
        raise RuntimeError("checkpoint contains an unexpected trainable route")
    save_file(tensors, str(stem.with_suffix(".safetensors")))
    stem.with_suffix(".json").write_text(json.dumps(dict(schema=SCHEMA, stage="Phase 1A.1a",
        step=step, body=[12,20], K=8, trainable_names=runtime.trainable_names,
        reader_warmup_checkpoint_sha256=sha256_file(config["reader_warmup_checkpoint"]),
        teacher_cache_sha256=sha256_file(config["teacher_cot_data"]),
        prompt_data_sha256=sha256_file(config["prompt_data"]),
        config=config), indent=2) + "\n")
    torch.save(dict(step=step, optimizer=optimizer.state_dict()),
               stem.with_suffix(".optimizer.pt"))


def train(config, records):
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite OPD output: {output}")
    from .reader_warmup import check_warmup_gate
    warmup_gate = check_warmup_gate(config)
    gate = check_teacher_baseline(config)
    runtime = OPDRuntime.load_model(config)
    if any("memory_reader.injection_gate" not in name for name in runtime.trainable_names):
        raise RuntimeError("OPD must train injection gates only")
    load_training_records(config, tokenizer=runtime.inferencer.tokenizer)
    output.mkdir(parents=True, exist_ok=False)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2)+"\n")
    (output / "run_manifest.json").write_text(json.dumps(dict(schema=SCHEMA,
        reader_warmup_gate=warmup_gate,
        teacher_baseline=gate, teacher_cache_sha256=sha256_file(config["teacher_cot_data"]),
        prompt_data_sha256=sha256_file(config["prompt_data"]),
        model_path=str(Path(config["model_path"]).resolve()),
        torch_version=torch.__version__, seed=config["seed"]), indent=2)+"\n")
    (output / "trainable_routes.json").write_text(json.dumps(runtime.trainable_names,indent=2)+"\n")
    params = [p for p in runtime.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(config["learning_rate"]),
                                  betas=(.9,.95), weight_decay=0.)
    status = dict(status="running", step=0)
    (output / "status.json").write_text(json.dumps(status)+"\n")
    try:
        # Loading a trained adapter must still produce exact native velocity
        # before the first gate update, on the same native state and timestep.
        parity_state = runtime.rollout(records[0],int(config["seed"]))[0]
        with torch.no_grad():
            difference=(runtime.student_velocity(parity_state).float()
                        - runtime.native_velocity(parity_state).float()).abs().max()
        if float(difference) > 1e-6:
            raise RuntimeError("loaded warm-up reader violates OPD step-0 native parity")
        (output / "step0_parity.json").write_text(json.dumps(dict(max_abs=float(difference)))+"\n")
        for step in range(1, config["max_steps"]+1):
            record = records[(step-1) % len(records)]
            seed = int(config["seed"]) + step*100003
            states = runtime.rollout(record, seed)
            optimizer.zero_grad(set_to_none=True)
            output_metrics = []
            for state in states:
                teacher = runtime.teacher_velocity(state)
                student = runtime.student_velocity(state)
                result = opd_velocity_loss(student, teacher)
                if not bool(torch.isfinite(result.loss)):
                    raise FloatingPointError(f"non-finite OPD loss at step {step}")
                (result.loss / len(states)).backward()
                output_metrics.append(dict(loss=float(result.loss.detach()),
                    teacher_rms=float(result.teacher_rms), student_rms=float(result.student_rms),
                    relative_error=float(result.relative_error), cosine=float(result.cosine)))
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
                raise FloatingPointError(f"missing/nonfinite reader gradient at step {step}")
            norm = torch.nn.utils.clip_grad_norm_(params, float(config["max_grad_norm"]),
                                                  error_if_nonfinite=True)
            optimizer.step()
            metrics = dict(step=step, prompt_id=record["prompt_id"],
                state_steps=[s.step_index for s in states],
                loss_opd=sum(o["loss"] for o in output_metrics)/len(output_metrics),
                teacher_velocity_rms=sum(o["teacher_rms"] for o in output_metrics)/len(output_metrics),
                student_velocity_rms=sum(o["student_rms"] for o in output_metrics)/len(output_metrics),
                relative_velocity_error=sum(o["relative_error"] for o in output_metrics)/len(output_metrics),
                velocity_cosine=sum(o["cosine"] for o in output_metrics)/len(output_metrics),
                grad_norm=float(norm), **runtime.model.last_opd_memory_stats,
                **runtime.reader_diagnostics())
            append_json(output / "metrics.jsonl", metrics)
            if step % config["save_steps"] == 0 or step == config["max_steps"]:
                checkpoint(runtime, output, step, optimizer, config)
            if step % config["eval_steps"] == 0:
                # Diagnostics only; no shuffled/zero branch contributes to loss.
                from .opd_evaluation import fixed_state_metrics
                if len(records) >= 2:
                    evaluation = fixed_state_metrics(runtime, records[:2], seed=int(config["seed"]))
                    append_json(output / "train_diagnostics.jsonl",
                                dict(step=step, scope="training_prompts_not_heldout", **evaluation))
            print(json.dumps(dict(step=step, loss_opd=metrics["loss_opd"])), flush=True)
        status = dict(status="complete", step=config["max_steps"],
                      semantic_gate="not_evaluated")
    except Exception as exc:
        status = dict(status="failed", step=step if "step" in locals() else 0,
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output / "status.json").write_text(json.dumps(status, indent=2)+"\n")
