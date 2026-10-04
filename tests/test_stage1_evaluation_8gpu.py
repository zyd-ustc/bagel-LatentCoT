"""Paired arm generation and full-benchmark reconstruction across eight workers."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image

from test_anchored_t2i_loop import tiny_model
from test_t2i_loop_entrypoints import install_tiny_backbone, script_module
from qwen_latent_cot.bagel.anchored_loop import configure_stage1
from qwen_latent_cot.bagel.loop_checkpoint import save_loop_checkpoint
from qwen_latent_cot.evaluation.loop_results import score_manifest, summarize_results, write_reports

ROOT = Path(__file__).resolve().parents[1]


def test_cached_native_weights_and_fast_readout_keep_matched_trained_images(tmp_path, monkeypatch):
    install_tiny_backbone(monkeypatch)
    matrix = script_module(ROOT / "scripts/evaluate/t2i_loop_matrix.py")
    checkpoints = {}
    for mode, slots in [("gen_only", 0), ("gen_memory_anchored", 2)]:
        model, _ = tiny_model(slots, mode, alpha=0.2)
        configure_stage1(model)
        with torch.no_grad():
            model.t2i_loop.reentry.up.bias.fill_(0.08)
        checkpoint = tmp_path / mode
        save_loop_checkpoint(model, checkpoint, step=1000, model_path=tmp_path / "base")
        checkpoints[mode] = checkpoint
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("\n".join(json.dumps({"prompt": p}) for p in ["two cubes", "one sphere", "red box"]))

    def args(mode, directory, fast):
        return SimpleNamespace(model_path=str(tmp_path / "base"), prompts=str(prompts),
                               checkpoint=str(checkpoints[mode]), output_dir=str(directory),
                               device="cpu", modes=mode, depths="0,1,2", memory_slots=2,
                               memory_control="correct", start_layer=1, end_layer=3,
                               alpha=0.9, gate=0.8, reentry_scale=0.05, image_size=16,
                               num_timesteps=4, timestep_shift=3.0, cfg_text_scale=4.0,
                               seed=13, batch_size=2, max_prompts=3, save_readouts=False,
                               no_loop_stats=fast, skip_base=False)

    backbone = matrix.run_matrix(args("gen_only", tmp_path / "gen", False))
    native = {name: p.clone() for name, p in backbone.bagel.named_parameters()
              if not name.startswith("t2i_loop.")}
    matrix.run_matrix(args("gen_memory_anchored", tmp_path / "reused", True), backbone)
    matrix.run_matrix(args("gen_memory_anchored", tmp_path / "cold", False))
    reused = json.loads((tmp_path / "reused/manifest.json").read_text())
    cold = json.loads((tmp_path / "cold/manifest.json").read_text())
    assert reused["loop_diagnostics_enabled"] is False
    assert all(not json.loads(p.read_text()) for p in (tmp_path / "reused").glob("*/batch_*_loop_logs.json"))
    for a, b in zip(reused["images"], cold["images"]):
        assert a["initial_noise_sha256"] == b["initial_noise_sha256"]
        assert a["seed"] == b["seed"]
        assert Path(a["path"]).read_bytes() == Path(b["path"]).read_bytes()
    for name, p in backbone.bagel.named_parameters():
        if name in native:
            torch.testing.assert_close(p, native[name], rtol=0, atol=0)
    # Legacy with a K=0 trained checkpoint still uses its own parent scratchpad.
    legacy = args("gen_only", tmp_path / "legacy", True)
    legacy.modes = "legacy_memory_only"
    matrix.run_matrix(legacy, backbone)
    manifest = json.loads((tmp_path / "legacy/manifest.json").read_text())
    assert len(manifest["images"]) == 9
    assert manifest["arms"]["legacy_memory_only_R1_K2_correct"]["readout"] == "parent_suffix_no_alpha"


def synthetic_generation(tmp_path):
    module = script_module(ROOT / "scripts/evaluate/stage1_8gpu.py")
    tools, out = tmp_path / "tools", tmp_path / "out"
    (tools / "data").mkdir(parents=True)
    out.mkdir()
    for dataset in ["hard16", "easy16"]:
        rows = [{"prompt": f"{dataset} prompt {i}", "vqa_list": ["Two objects?"]} for i in range(7)]
        (tools / "data" / f"{dataset}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    options = SimpleNamespace(gpus=list(map(str, range(8))), tools_dir=tools,
                              output_dir=out, hard_benchmark="hard16", max_prompts=7,
                              judge_model_path=tmp_path / "judge")
    jobs = module.generation_plan(options)
    for job in jobs:
        job["output_dir"] = str(out / "workers" / str(job["rank"]))
        for dataset in ["hard16", "easy16"]:
            for mode in [job["mode"]] + (["legacy_memory_only"] if job["legacy"] else []):
                directory = Path(job["output_dir"]) / dataset / mode
                directory.mkdir(parents=True)
                depth = job['legacy_depth'] if mode == 'legacy_memory_only' else job['depth']
                arms = [f"{mode}_R{depth}_K{0 if mode == 'gen_only' else 8}_correct"]
                if job["rank"] == 0 and mode == "gen_only":
                    arms = ["base_R0_K0_correct"] + arms
                images = []
                for arm in arms:
                    for i in range(7):
                        image = directory / f"{arm}_{i}.png"
                        Image.new("RGB", (8, 8), "red").save(image)
                        images.append(dict(arm=arm, index=i, prompt=f"{dataset} prompt {i}",
                                           seed=13 + i // 2 * 2, initial_noise_sha256=f"noise{i}",
                                           path=str(image), depth_status="seen"))
                (directory / "manifest.json").write_text(json.dumps(dict(
                    images=images, arms={arm: {} for arm in arms},
                    allocated_loop_config=dict(loop_start_layer=1, loop_end_layer=3),
                    arguments=dict(model_path="native", seed=13, num_timesteps=4,
                                   timestep_shift=3.0, cfg_text_scale=4.0, start_layer=1,
                                   end_layer=3, batch_size=2, image_size=16, depths=f"0,{job['depth']}"))))
    return module, options, jobs


def test_eight_workers_merge_13_arms_and_scoring_restores_global_pairs(tmp_path):
    module, options, jobs = synthetic_generation(tmp_path)
    assert [(j["mode"], j["depth"]) for j in jobs] == [
        (mode, depth) for mode in ("gen_only", "gen_memory_anchored") for depth in range(1, 5)]
    module.merge_generation(options, jobs)
    (options.output_dir / "launch_provenance.json").write_text('{}')
    score_jobs = module.scoring_plan(options)
    assert len(score_jobs) == 8
    for job in score_jobs:
        cfg = yaml.safe_load(Path(job["config"]).read_text())
        dataset = cfg["datasets"][job["dataset"]]
        manifest = json.loads(Path(dataset["manifest"]).read_text())
        benchmark = [json.loads(x) for x in Path(dataset["benchmark"]).read_text().splitlines()]
        scored = score_manifest(manifest, benchmark, dataset=job["dataset"],
                                semantic_scorer=lambda rows, benchmarks: [[0.8]] * len(rows),
                                quality_judge=lambda _: dict(quality_proxy=0.75, invalid=False))
        write_reports(cfg["output_dir"], scored, summarize_results(scored), {"fixture": True})
    module.merge_scoring(options, score_jobs)
    rows = [json.loads(x) for x in (options.output_dir / "results/scores.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * 7 * 13
    assert len({(r["dataset"], r["arm"], r["index"], r["seed"]) for r in rows}) == len(rows)
    summaries = json.loads((options.output_dir / "results/summary.json").read_text())["summaries"]
    assert len(summaries) == 26
    assert all(r["n"] == 7 for r in summaries)


def test_generation_merge_rejects_changed_noise(tmp_path):
    module, options, jobs = synthetic_generation(tmp_path)
    path = Path(jobs[7]["output_dir"]) / "hard16/gen_memory_anchored/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["images"][0]["initial_noise_sha256"] = "different"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unmatched"):
        module.merge_generation(options, jobs)


def test_failed_worker_stops_its_peers_and_preserves_failure(tmp_path, monkeypatch):
    import signal
    import threading

    module = script_module(ROOT / "scripts/evaluate/stage1_8gpu.py")
    stopped = threading.Event()
    kills = []

    class Process:
        def __init__(self, pid, fail):
            self.pid, self.code, self.fail = pid, None, fail

        def wait(self):
            if self.fail:
                self.code = 1
            else:
                assert stopped.wait(5), "peer was not stopped after worker failure"
                self.code = -15
            return self.code

        def poll(self):
            return self.code

    processes = [Process(100001, True), Process(100002, False)]
    queue = iter(processes)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: next(queue))

    def kill_group(pid, sig):
        kills.append((pid, sig))
        stopped.set()

    monkeypatch.setattr(module.os, "killpg", kill_group)
    options = SimpleNamespace(output_dir=tmp_path, tools_dir=tmp_path)
    with pytest.raises(RuntimeError, match="failed rank 0"):
        module.run_workers(options, [{"rank": 0, "gpu": "0"}, {"rank": 1, "gpu": "1"}], "generate")
    assert kills == [(100002, signal.SIGTERM)]
    assert (tmp_path / "generate_rank_0.log").is_file()
