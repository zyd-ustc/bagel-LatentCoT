"""Fixed-gate image diagnosis for a Warm-up checkpoint, not trained OPD."""

from contextlib import contextmanager
import hashlib
import html
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from .cot_teacher import sha256_file
from .opd_runtime import OPDState
from .reader_warmup import (ReaderWarmupRuntime, inspect_warmup_checkpoint,
                            load_warmup_checkpoint, load_warmup_records,
                            validate_warmup_config)
from ..evaluation.geneval2 import load_benchmark
from ..evaluation.image_scoring import prepare_phase1a_image_maps


ARMS = ("native", "untrained_reader", "step5000_reader")
SCHEMA = "bagel-fixed-reader-injection-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def tensor_digest(value):
    value = value.detach().float().cpu().contiguous()
    return hashlib.sha256(str(tuple(value.shape)).encode() + value.numpy().tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def select_hard_records(path, count):
    if isinstance(count, bool) or not isinstance(count, int) or count < 4 or count % 4:
        raise ValueError("max_prompts must be a positive multiple of four")
    load_benchmark(path)  # Validate VQA/skills and unique prompts before selection.
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    selected = []
    for atomicity in (7, 8, 9, 10):
        bucket = [row for row in rows if row["atom_count"] == atomicity]
        if len(bucket) < count // 4:
            raise ValueError(f"not enough atomicity-{atomicity} prompts")
        selected.extend(bucket[:count // 4])
    ids = [row.get("prompt_id") for row in selected]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != count:
        raise ValueError("benchmark needs distinct prompt_id values")
    if any(not row.get("vqa_list") for row in selected):
        raise ValueError("benchmark must preserve semantic VQA questions")
    return selected


def build_plan(checkpoint, benchmark, output, *, gate_scale=1., max_prompts=16,
               num_shards=8, seed=None, model_path=None):
    if isinstance(gate_scale, bool) or not math.isfinite(gate_scale):
        raise ValueError("gate_scale must be finite")
    if isinstance(num_shards, bool) or not isinstance(num_shards, int) or not 1 <= num_shards <= max_prompts:
        raise ValueError("num_shards must be between one and max_prompts")
    checkpoint = Path(checkpoint).resolve()
    meta = inspect_warmup_checkpoint(checkpoint)
    if meta.get("step") != 5000 or meta.get("stage") != "Phase 1A.0":
        raise ValueError("step5000_reader requires the Phase 1A.0 step-5000 checkpoint")
    source = checkpoint.parent / "resolved_config.json"
    config = json.loads(source.read_text())
    config.pop("resume_checkpoint", None)
    config.update(output_dir=str(Path(output).resolve()), device="cuda:0")
    expected_model = str(Path(meta["model_path"]).resolve())
    if model_path is not None and str(Path(model_path).resolve()) != expected_model:
        raise ValueError("model_path must match checkpoint training model")
    if str(Path(config["model_path"]).resolve()) != expected_model:
        raise ValueError("source config model mismatch")
    config = validate_warmup_config(config)
    expected = dict(K=config["num_loop_tokens"], body=[config["memory_loop_start_layer"],
        config["memory_loop_end_layer"]], adapter_rank=config["o_adapter_rank"],
        adapter_alpha=config["o_adapter_alpha"], memory_init="prompt_hidden_uniform",
        query_source="native_gen_q", memory_kv_source="frozen_read_native_kv",
        prompt_target="native_prompt_bank", generation_injection=False)
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("source config reader geometry mismatch")
    train, heldout = load_warmup_records(config)
    for name in ("prompt_data", "heldout_prompt_data"):
        if sha256_file(config[name]) != meta[name + "_sha256"]:
            raise ValueError(f"original {name} SHA-256 mismatch")
    records = select_hard_records(benchmark, max_prompts)
    normalize = lambda prompt: " ".join(prompt.split()).casefold()
    source_prompts = {normalize(row["prompt"]) for row in train + heldout}
    if any(normalize(row["prompt"]) in source_prompts for row in records):
        raise ValueError("semantic benchmark overlaps Warm-up train/heldout prompts")
    selected_seed = config["seed"] if seed is None else seed
    if isinstance(selected_seed, bool) or not isinstance(selected_seed, int) or selected_seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    plan = dict(schema=SCHEMA, checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
        metadata_sha256=sha256_file(checkpoint.with_suffix(".json")),
        source_config_sha256=sha256_file(source), benchmark=str(Path(benchmark).resolve()),
        benchmark_sha256=sha256_file(benchmark), output=str(Path(output).resolve()),
        gate_scale=float(gate_scale), arms=list(ARMS), seed=selected_seed,
        num_shards=num_shards, max_prompts=max_prompts, config=config, records=records,
        prompt_seeds=[selected_seed + index for index in range(max_prompts)],
        untrained_definition="B=0: exact initialized translation output independent of A",
        evidence_scope="fixed-injection diagnostic; not gate-trained OPD or proven semantic gain")
    code_root = Path(__file__).resolve().parents[2]
    plan["source_sha256"] = {str(path.relative_to(code_root)): sha256_file(path) for path in
        (Path(__file__), code_root / "qwen_latent_cot/bagel/reader_warmup.py",
         code_root / "qwen_latent_cot/bagel/opd_runtime.py",
         code_root / "qwen_latent_cot/bagel/memory_reader.py",
         code_root / "qwen_latent_cot/bagel/modeling/bagel/qwen2_navit.py",
         code_root / "qwen_latent_cot/bagel/modeling/bagel/bagel.py",
         code_root / "qwen_latent_cot/bagel/inferencer.py",
         code_root / "scripts/evaluate/bagel_reader_injection.py",
         code_root / "scripts/evaluate/run_bagel_reader_injection.sh")}
    return plan


@contextmanager
def reader_arm(runtime, trained, arm, scale):
    if arm not in ARMS or not math.isfinite(scale):
        raise ValueError("invalid reader arm/scale")
    named = dict(runtime.model.named_parameters())
    gates = [layer.memory_reader.injection_gate for layer in
             runtime.model.language_model.model.layers[runtime.body_start:runtime.body_end]]
    old_gates = [value.detach().clone() for value in gates]
    saved = {name: named[name].detach().clone() for name in trained}
    try:
        with torch.no_grad():
            for name, value in trained.items():
                named[name].copy_(value.to(named[name]))
                if arm == "untrained_reader" and name.endswith(".B.weight"):
                    named[name].zero_()
            for gate in gates:
                gate.fill_(0. if arm == "native" else scale)
        yield
    finally:
        with torch.no_grad():
            for name, value in saved.items():
                named[name].copy_(value)
            for gate, value in zip(gates, old_gates):
                gate.copy_(value)


@torch.no_grad()
def generate_prompt(runtime, record, *, seed, gate_scale, trained, save_dir=None):
    condition = runtime.prepare(record, seed)
    ts, dts = runtime.model.prepare_image_schedule(runtime.config["num_steps"],
        runtime.config["timestep_shift"], runtime.device)
    schedule = [[float(t), float(dt)] for t, dt in zip(ts, dts)]
    first = OPDState(condition, condition.noise.detach().clone(), float(ts[0]), 0)
    with reader_arm(runtime, trained, "step5000_reader", 0.):
        parity = float((runtime.student_velocity(first).float() - runtime.native_velocity(first).float()).abs().max())
    if not math.isfinite(parity) or parity > 1e-6:
        raise RuntimeError(f"gate-zero native parity failed: {parity}")
    arm_rows, samples = {}, {}
    for arm in ARMS:
        sample = condition.noise.detach().clone()
        initial_hash = tensor_digest(sample)
        with reader_arm(runtime, trained, arm, gate_scale):
            for step, (t, dt) in enumerate(zip(ts, dts)):
                state = OPDState(condition, sample, float(t), step)
                velocity = runtime.native_velocity(state) if arm == "native" else runtime.student_velocity(state)
                if not bool(torch.isfinite(velocity).all()):
                    raise FloatingPointError(f"non-finite {arm} velocity at step {step}")
                sample = runtime.model.image_euler_step(sample, velocity, dt).detach()
                if not bool(torch.isfinite(sample).all()):
                    raise FloatingPointError(f"non-finite {arm} latent at step {step}")
        samples[arm] = sample
        if save_dir is not None:
            runtime.inferencer.decode_image(sample, runtime.shape).save(Path(save_dir) / f"{arm}.png")
        arm_rows[arm] = dict(initial_noise_sha256=initial_hash, final_latent_sha256=tensor_digest(sample),
                             schedule_sha256=digest(schedule), euler_steps=len(dts))
    return dict(prompt_id=record["prompt_id"], prompt=record["prompt"], seed=seed,
                gate_scale=gate_scale, gate_zero_native_parity_max_abs=parity, arms=arm_rows), samples


def run_shard(plan, shard_id):
    if not 0 <= shard_id < plan["num_shards"]:
        raise ValueError("invalid shard_id")
    root = Path(plan["output"])
    shard = root / "shards" / f"s{shard_id:03d}"
    shard.mkdir(parents=True, exist_ok=False)
    write_json(shard / "status.json", dict(status="loading", shard_id=shard_id))
    try:
        runtime = ReaderWarmupRuntime.load_model(plan["config"])
        load_warmup_checkpoint(runtime, plan["checkpoint"])
        runtime.model.requires_grad_(False).eval()
        trained = load_file(plan["checkpoint"])
        rows = []
        for index, record in enumerate(plan["records"]):
            if index % plan["num_shards"] != shard_id:
                continue
            folder = root / f"p{index:03d}"
            folder.mkdir(parents=True, exist_ok=False)
            row, _ = generate_prompt(runtime, record, seed=plan["prompt_seeds"][index],
                                    gate_scale=plan["gate_scale"], trained=trained, save_dir=folder)
            row["index"] = index
            row["image_sha256"] = {arm: sha256_file(folder / f"{arm}.png") for arm in ARMS}
            (folder / "prompt.txt").write_text(record["prompt"] + "\n")
            write_json(folder / "generation.json", row)
            rows.append(row)
            print(json.dumps(dict(event="prompt_complete", index=index, prompt_id=record["prompt_id"],
                                  seed=row["seed"], arms=ARMS)), flush=True)
        write_json(shard / "manifest.json", dict(plan_sha256=digest(plan), shard_id=shard_id, rows=rows))
        write_json(shard / "status.json", dict(status="complete", shard_id=shard_id, prompts=len(rows)))
    except Exception as exc:
        write_json(shard / "status.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
        raise


def merge_shards(plan):
    root = Path(plan["output"])
    if (root / "index.html").exists():
        raise FileExistsError("refusing to overwrite completed gallery")
    rows = []
    for shard_id in range(plan["num_shards"]):
        shard = root / "shards" / f"s{shard_id:03d}"
        if json.loads((shard / "status.json").read_text())["status"] != "complete":
            raise ValueError("cannot merge incomplete shard")
        manifest = json.loads((shard / "manifest.json").read_text())
        if manifest["plan_sha256"] != digest(plan) or manifest["shard_id"] != shard_id:
            raise ValueError("shard plan mismatch")
        expected = list(range(shard_id, plan["max_prompts"], plan["num_shards"]))
        if [row["index"] for row in manifest["rows"]] != expected:
            raise ValueError("shard coverage mismatch")
        rows.extend(manifest["rows"])
    rows.sort(key=lambda row: row["index"])
    if [row["index"] for row in rows] != list(range(plan["max_prompts"])):
        raise ValueError("prompt coverage mismatch")
    for index, row in enumerate(rows):
        folder = root / f"p{index:03d}"
        if (row["prompt_id"] != plan["records"][index]["prompt_id"]
                or row["prompt"] != plan["records"][index]["prompt"] or set(row["arms"]) != set(ARMS)
                or row["seed"] != plan["prompt_seeds"][index] or row["gate_scale"] != plan["gate_scale"]
                or row != json.loads((folder / "generation.json").read_text())
                or not math.isfinite(row["gate_zero_native_parity_max_abs"])
                or row["gate_zero_native_parity_max_abs"] > 1e-6):
            raise ValueError("prompt generation metadata mismatch")
        for key in ("initial_noise_sha256", "schedule_sha256", "euler_steps"):
            if len({row["arms"][arm][key] for arm in ARMS}) != 1:
                raise ValueError(f"arms disagree on {key}")
        if row["arms"]["native"]["euler_steps"] != plan["config"]["num_steps"] - 1:
            raise ValueError("Euler step count does not match training schedule")
        if any(row["image_sha256"][arm] != sha256_file(folder / f"{arm}.png") for arm in ARMS):
            raise ValueError("image checksum mismatch")
    benchmark = root / "benchmark.jsonl"
    benchmark.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in plan["records"]))
    maps = prepare_phase1a_image_maps(benchmark, root, ARMS)
    if maps["image_shape"] != [plan["config"]["width"], plan["config"]["height"]]:
        raise ValueError("images do not match training geometry")
    for arm, mapping in maps["image_maps"].items():
        write_json(root / f"{arm}_image_map.json", mapping)
    write_json(root / "score_inputs.json", maps)
    write_json(root / "generation_manifest.json", dict(plan=plan, rows=rows, semantic_scores=None))
    cards = []
    for index, row in enumerate(plan["records"]):
        images = "".join(f'<figure><img loading="lazy" src="p{index:03d}/{arm}.png"><figcaption>{arm}</figcaption></figure>' for arm in ARMS)
        cards.append(f'<section><h2>{index + 1}. {html.escape(row["prompt"])}</h2><div>{images}</div></section>')
    page = ('<!doctype html><meta charset="utf-8"><title>Fixed reader injection</title>'
            '<style>body{font-family:sans-serif;max-width:1500px;margin:24px auto}div{display:flex}'
            'figure{margin:8px;flex:1}img{width:100%}h2{font-size:18px}</style>'
            f'<h1>Fixed gate={plan["gate_scale"]}; step-5000 reader diagnosis</h1>'
            '<p>Not gate-trained OPD. Images do not establish semantic improvement.</p>' + "".join(cards))
    (root / "index.html").write_text(page)
    write_json(root / "status.json", dict(status="complete", prompts=len(rows), images=len(rows) * len(ARMS)))
