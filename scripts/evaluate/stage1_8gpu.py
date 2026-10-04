#!/usr/bin/env python3
"""Eight independent inference/scoring workers with matched-noise arm coverage."""

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml

from qwen_latent_cot.evaluation.loop_results import (
    file_sha256, merge_manifests, summarize_results, validate_required_arms, write_reports,
)


def generation_plan(options):
    jobs = []
    # Pair the deeper legacy controls with shallower trained arms.
    legacy_depths = {0: 4, 4: 3, 1: 2, 5: 1}
    for rank in range(8):
        mode = "gen_only" if rank < 4 else "gen_memory_anchored"
        jobs.append(dict(rank=rank, gpu=options.gpus[rank], mode=mode,
                         depth=rank % 4 + 1, legacy=rank in legacy_depths,
                         legacy_depth=legacy_depths.get(rank)))
    return jobs


def check_assets(options):
    if len(options.gpus) != 8 or len(set(options.gpus)) != 8 or any(
        not item.isdigit() for item in options.gpus
    ):
        raise ValueError("select eight unique numeric GPU indexes")
    if options.batch_size < 1 or options.num_timesteps < 2 or options.max_prompts < 1:
        raise ValueError("invalid batch size, timestep count or prompt limit")
    metas = {}
    for path in (options.tools_dir / "GenEval2/evaluation.py",
                 options.tools_dir / "venv/bin/python", options.judge_model_path / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    for mode, slots in [("gen_only", 0), ("gen_memory_anchored", 8)]:
        checkpoint = options.training_dir / mode / "step_001000"
        meta = json.loads((checkpoint / "loop.json").read_text())
        loop = meta["loop_config"]
        if meta["format"] != "umm-t2i-anchored-loop-v4" or meta["step"] != 1000:
            raise ValueError(f"expected final v4 checkpoint: {checkpoint}")
        if loop["loop_mode"] != mode or loop["memory_slots"] != slots:
            raise ValueError(f"checkpoint mode/workspace differs: {checkpoint}")
        if loop["allocated_max_loop_depth"] < 4 or len(meta["round_training_steps"]) < 3 or not all(meta["round_training_steps"][:3]):
            raise ValueError("require trained R1-3 and allocated R4")
        if Path(meta["model_path"]).resolve() != options.model_path.resolve():
            raise ValueError("training and inference native checkpoints differ")
        if not (checkpoint / "loop.safetensors").is_file():
            raise FileNotFoundError(checkpoint)
        metas[mode] = meta
    a, b = (metas[mode]["loop_config"] for mode in ("gen_only", "gen_memory_anchored"))
    if any(a[k] != b[k] for k in ("loop_start_layer", "loop_end_layer", "reentry_rank")):
        raise ValueError("loop layers/rank differ across trained arms")
    for name in (options.hard_benchmark, "easy16"):
        path = options.tools_dir / "data" / f"{name}.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()][:options.max_prompts]
        if not rows or any(not row.get("prompt") or not row.get("vqa_list", row.get("yn_question_list")) for row in rows):
            raise ValueError(f"semantic benchmark questions missing: {path}")
    return metas


def check_memory(options, minimum_mib):
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, capture_output=True, check=True)
    cards = {line.split(",")[0].strip(): [int(v.strip()) for v in line.split(",")[1:]]
             for line in result.stdout.splitlines() if line.strip()}
    selected = {gpu: cards[gpu] for gpu in options.gpus}
    print("GPU free MiB / utilization %:", json.dumps(selected), flush=True)
    if any(value[0] < minimum_mib for value in selected.values()):
        raise RuntimeError(f"need >= {minimum_mib} MiB free on every selected GPU")


def run_workers(options, jobs, phase):
    processes, handles = [], []
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        for job in jobs:
            rank = job["rank"]
            path = options.output_dir / f"{phase}_rank_{rank}.json"
            path.write_text(json.dumps(job, indent=2) + "\n")
            log = (options.output_dir / f"{phase}_rank_{rank}.log").open("wb")
            handles.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=job["gpu"],
                       HF_HUB_CACHE=str(options.tools_dir / "hf_hub"),
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       TOKENIZERS_PARALLELISM="false")
            env.setdefault("OMP_NUM_THREADS", "8")
            env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
            python = sys.executable if phase == "generate" else str(options.tools_dir / "venv/bin/python")
            process = subprocess.Popen([python, "-u", str(Path(__file__).resolve()),
                                        "--worker", phase, "--job", str(path)],
                                       stdout=log, stderr=subprocess.STDOUT, env=env,
                                       start_new_session=True)
            processes.append((process, rank))
            print(f"Started {phase} rank={rank} physical_gpu={job['gpu']} pid={process.pid}; log={log.name}", flush=True)
        pending = {pool.submit(process.wait): rank for process, rank in processes}
        for future in as_completed(pending):
            rank, code = pending[future], future.result()
            print(f"Finished {phase} rank={rank} exit={code}", flush=True)
            if code:
                raise RuntimeError(f"{phase} failed rank {rank}; inspect its log")
    finally:
        for process, _ in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process, _ in processes:
            process.wait()
        for handle in handles:
            handle.close()
        pool.shutdown(wait=True)


def generate_worker(job):
    from t2i_loop_matrix import run_matrix
    backbone = None
    for dataset in job["datasets"]:
        for mode in [job["mode"]] + (["legacy_memory_only"] if job["legacy"] else []):
            args = SimpleNamespace(
                model_path=job["model_path"], checkpoint=job["checkpoint"],
                prompts=job["benchmarks"][dataset], device="cuda:0", modes=mode,
                depths=f"0,{job['legacy_depth'] if mode == 'legacy_memory_only' else job['depth']}",
                memory_slots=8, memory_control="correct",
                start_layer=job["start_layer"], end_layer=job["end_layer"],
                alpha=0.1, gate=0.02, reentry_scale=0.05, image_size=512,
                num_timesteps=job["num_timesteps"], timestep_shift=3.0,
                cfg_text_scale=4.0, seed=job["seed"], batch_size=job["batch_size"],
                max_prompts=job["max_prompts"], save_readouts=False, no_loop_stats=True,
                skip_base=not (job["rank"] == 0 and mode == "gen_only"),
                output_dir=str(Path(job["output_dir"]) / dataset / mode),
            )
            print("Job", json.dumps(vars(args)), flush=True)
            backbone = run_matrix(args, backbone)


def merge_generation(options, jobs):
    manifests = {}
    for dataset in (options.hard_benchmark, "easy16"):
        specs = []
        for job in jobs:
            path = Path(job["output_dir"]) / dataset / job["mode"] / "manifest.json"
            specs.append(dict(path=str(path), modes=[job["mode"]] + (["base"] if job["rank"] == 0 else [])))
            if job["legacy"]:
                path = Path(job["output_dir"]) / dataset / "legacy_memory_only/manifest.json"
                specs.append(dict(path=str(path), modes=["legacy_memory_only"]))
        merged = merge_manifests(specs)
        validate_required_arms(merged, [0, 1, 2, 3, 4])
        reference = json.loads(Path(specs[0]["path"]).read_text())
        merged["arguments"] = {**reference["arguments"], "depths": "0,1,2,3,4"}
        merged["allocated_loop_config"] = reference["allocated_loop_config"]
        merged["source_manifests"] = [{**spec, "sha256": file_sha256(spec["path"])} for spec in specs]
        bases = {row["index"]: row for row in merged["images"] if row["arm"].startswith("base_R0_")}
        expected = set(range(min(options.max_prompts, len(read_benchmark(options, dataset)))))
        by_arm = {}
        for row in merged["images"]:
            base = bases[row["index"]]
            if any(row[k] != base[k] for k in ("prompt", "seed", "initial_noise_sha256")):
                raise ValueError("unmatched prompt/noise across generated arms")
            if not Path(row["path"]).is_file():
                raise FileNotFoundError(row["path"])
            by_arm.setdefault(row["arm"], set()).add(row["index"])
        if len(by_arm) != 13 or any(indices != expected for indices in by_arm.values()):
            raise ValueError("generation matrix has incomplete arm/prompt coverage")
        out = options.output_dir / dataset
        out.mkdir()
        path = out / "manifest.json"
        path.write_text(json.dumps(merged, indent=2) + "\n")
        manifests[dataset] = path
        print(f"Merged {dataset}: {len(merged['images'])} images / 13 arms", flush=True)
    return manifests


def read_benchmark(options, dataset):
    return [json.loads(x) for x in (options.tools_dir / "data" / f"{dataset}.jsonl").read_text().splitlines() if x.strip()][:options.max_prompts]


def scoring_plan(options):
    jobs = []
    for dataset_index, dataset in enumerate((options.hard_benchmark, "easy16")):
        manifest_path = options.output_dir / dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        validate_required_arms(manifest, [0, 1, 2, 3, 4])
        benchmark = read_benchmark(options, dataset)
        for shard in range(min(4, len(benchmark))):
            rank = dataset_index * 4 + shard
            indexes = list(range(shard, len(benchmark), 4))
            remap = {old: new for new, old in enumerate(indexes)}
            part = copy.deepcopy(manifest)
            part["images"] = [{**row, "index": remap[row["index"]],
                               "benchmark_index_global": row["index"]}
                              for row in manifest["images"] if row["index"] in remap]
            path = options.output_dir / "score_inputs" / f"rank_{rank}"
            path.mkdir(parents=True)
            (path / "manifest.json").write_text(json.dumps(part, indent=2) + "\n")
            (path / "benchmark.jsonl").write_text("\n".join(json.dumps(benchmark[i]) for i in indexes) + "\n")
            cfg = dict(
                output_dir=str(options.output_dir / "results/shards" / f"rank_{rank}"),
                geneval2_root=str(options.tools_dir / "GenEval2"),
                geneval2_python=str(options.tools_dir / "venv/bin/python"),
                judge_model_path=str(options.judge_model_path), judge_device="cuda:0",
                require_quality=True, semantic_pass_threshold=0.5, required_depths=[0,1,2,3,4],
                datasets={dataset:dict(kind="geneval2_hard", manifest=str(path / "manifest.json"),
                                       benchmark=str(path / "benchmark.jsonl"))},
            )
            config = path / "config.yaml"
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            jobs.append(dict(rank=rank, gpu=options.gpus[rank], config=str(config), dataset=dataset))
    return jobs


def merge_scoring(options, jobs):
    rows, provenance = [], dict(source="eight_gpu_prompt_shards", shards=[])
    seen = set()
    for job in jobs:
        path = options.output_dir / "results/shards" / f"rank_{job['rank']}"
        shard_rows = [json.loads(x) for x in (path / "scores.jsonl").read_text().splitlines()]
        for row in shard_rows:
            row["index"] = row.pop("benchmark_index_global")
            identity = (row["dataset"], row["arm"], row["index"], row["seed"])
            if identity in seen:
                raise ValueError("duplicate scoring shard identity")
            seen.add(identity)
            rows.append(row)
        provenance["shards"].append(dict(rank=job["rank"], config=job["config"],
                                          config_sha256=file_sha256(job["config"]),
                                          report=json.loads((path / "summary.json").read_text())["provenance"]))
    for dataset in (options.hard_benchmark, "easy16"):
        manifest = json.loads((options.output_dir / dataset / "manifest.json").read_text())
        expected = {(dataset, r["arm"], r["index"], r["seed"]) for r in manifest["images"]}
        actual = {key for key in seen if key[0] == dataset}
        if expected != actual:
            raise ValueError("scoring shards do not cover generated images")
    provenance["launch"] = json.loads((options.output_dir / "launch_provenance.json").read_text())
    write_reports(options.output_dir / "results", rows, summarize_results(rows, threshold=0.5), provenance)
    print("Results:", options.output_dir / "results/summary.md", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["generate", "score"])
    parser.add_argument("--job", type=Path)
    parser.add_argument("--phase", choices=["check", "generate", "score", "all"], default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--training-dir", type=Path, default=Path("/private/yida_workspace/outputs/umm_stage1_8gpu_20261004_075341"))
    parser.add_argument("--tools-dir", type=Path, default=Path("/private/yida_workspace/umm-anchored-eval-tools-d126833"))
    parser.add_argument("--model-path", type=Path, default=Path("/private/yida_workspace/models/BAGEL-7B-MoT"))
    parser.add_argument("--judge-model-path", type=Path, default=Path("/private/yida_workspace/models/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"))
    parser.add_argument("--hard-benchmark", choices=["hard16", "hard128"], default="hard128")
    parser.add_argument("--max-prompts", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()
    if options.worker:
        job = json.loads(options.job.read_text())
        if options.worker == "generate":
            generate_worker(job)
        else:
            from evaluate_t2i_loops import main as score_main
            sys.argv = ["evaluate_t2i_loops.py", "--config", job["config"]]
            score_main()
        return
    if options.output_dir is None:
        parser.error("--output-dir is required")
    options.output_dir = options.output_dir.resolve()
    options.gpus = options.gpus.split(",")
    metas = check_assets(options)
    if options.phase in {"check", "generate", "all"}:
        check_memory(options, 45 * 1024)
        if options.output_dir.exists():
            raise FileExistsError("use a new output directory")
    else:
        check_memory(options, 25 * 1024)
        if (options.output_dir / "results").exists():
            raise FileExistsError("scoring results already exist; preserve previous run")
        # Resume scoring only with the original generation settings.
        recorded = json.loads((options.output_dir / "launch_provenance.json").read_text())
        for key in ("hard_benchmark", "max_prompts", "seed", "batch_size", "num_timesteps"):
            if getattr(options, key) != recorded["settings"][key]:
                raise ValueError(f"scoring settings differ from generation: {key}")
    jobs = generation_plan(options)
    print("Generation plan:", json.dumps(jobs), flush=True)
    if options.phase == "check":
        print("Check passed; no generation started.", flush=True)
        return
    if options.phase in {"generate", "all"}:
        options.output_dir.mkdir(parents=True)
        settings = {key: value if not isinstance(value, Path) else str(value)
                    for key, value in vars(options).items()}
        provenance = dict(settings=settings,
                          native_model_path=str(options.model_path),
                          checkpoint_metadata=metas,
                          checkpoint_hashes={mode:file_sha256(options.training_dir / mode / "step_001000/loop.safetensors") for mode in metas},
                          benchmark_hashes={name:file_sha256(options.tools_dir / "data" / f"{name}.jsonl") for name in (options.hard_benchmark, "easy16")},
                          runner_sha256=file_sha256(__file__), matrix_sha256=file_sha256(ROOT / "scripts/evaluate/t2i_loop_matrix.py"))
        patch = ROOT / "EVAL_8GPU_PATCH.json"
        if patch.is_file():
            provenance["source_patch"] = json.loads(patch.read_text())
        (options.output_dir / "launch_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        for job in jobs:
            job.update(model_path=str(options.model_path),
                       checkpoint=str(options.training_dir / job["mode"] / "step_001000"),
                       datasets=[options.hard_benchmark, "easy16"],
                       benchmarks={name:str(options.tools_dir / "data" / f"{name}.jsonl") for name in (options.hard_benchmark, "easy16")},
                       output_dir=str(options.output_dir / "workers" / f"rank_{job['rank']}"),
                       start_layer=metas[job["mode"]]["loop_config"]["loop_start_layer"],
                       end_layer=metas[job["mode"]]["loop_config"]["loop_end_layer"],
                       num_timesteps=options.num_timesteps, batch_size=options.batch_size,
                       max_prompts=options.max_prompts, seed=options.seed)
        run_workers(options, jobs, "generate")
        merge_generation(options, jobs)
    if options.phase in {"score", "all"}:
        check_memory(options, 25 * 1024)
        jobs = scoring_plan(options)
        run_workers(options, jobs, "score")
        merge_scoring(options, jobs)


if __name__ == "__main__":
    main()
