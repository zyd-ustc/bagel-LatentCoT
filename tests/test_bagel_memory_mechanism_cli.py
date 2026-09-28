from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/evaluate"))
import bagel_memory_mechanism as cli


def arguments(tmp_path):
    data = tmp_path / "prompts.jsonl"
    data.write_text("\n".join(json.dumps({"prompt": f"prompt {i}"}) for i in range(4)))
    return ["--benchmark-data", str(data), "--output-dir", str(tmp_path / "out"),
            "--max-prompts", "4", "--num-steps", "3", "--height", "16", "--width", "16"]


def test_pair_partition_and_protocol_reject_invalid_variants(tmp_path):
    argv = arguments(tmp_path)
    plan = cli.make_plan(cli.parse_args(argv + ["--num-shards", "2"]))
    assert [pair["shard"] for pair in plan["pairs"]] == [0, 1]
    assert plan["body"] == [12, 20]
    assert plan["R_values"] == [2, 4, 6, 8]
    assert plan["mode"] == "normal"
    assert plan["reference_arm"] == "normal_r2"
    assert plan["noise_schema"] == "bagel-memory-mechanism-v1"
    assert plan["denoising_nfe"] == 2
    with pytest.raises(ValueError, match="complete pairs"):
        cli.make_plan(cli.parse_args(argv + ["--num-shards", "4"]))
    with pytest.raises(ValueError, match="even"):
        cli.make_plan(cli.parse_args(argv + ["--max-prompts", "3"]))
    with pytest.raises(SystemExit):
        cli.parse_args(argv + ["--prompt-kv-mask", "1"])


def test_dry_run_does_not_load_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_native_bagel", lambda _: pytest.fail("must not load weights"))
    cli.main(arguments(tmp_path) + ["--dry-run"])
    assert json.loads(capsys.readouterr().out)["K"] == 8


def test_full_artifact_pipeline_and_strict_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "prepare_runtime", lambda _: {"device": "cpu"})
    model = SimpleNamespace(
        latent_downsample=16, latent_channel=1, latent_patch_size=1,
        prepare_image_schedule=lambda *args: (torch.tensor([1., 0.5]), torch.tensor([0.5, 0.5])),
        image_euler_step=lambda x, v, dt: x - v * dt,
    )
    inferencer = SimpleNamespace(
        device=torch.device("cpu"),
        decode_image=lambda latent, shape: Image.new("RGB", (16, 16), color=(int(abs(float(latent.sum())) * 10) % 255, 0, 0)),
    )
    monkeypatch.setattr(cli, "load_native_bagel", lambda _: (SimpleNamespace(bagel=model), inferencer))
    class Engine:
        def __init__(self, inferencer, prompts, noises, shape):
            self.noise = torch.cat(noises)
            self.device = torch.device("cpu")
            self.model = model
            self.lengths = [1, 1]

        def velocity(self, x_t, timestep, mode, diagnostics=False):
            velocity = torch.full_like(x_t, cli.MODES.index(mode) + 1)
            return velocity, {"gen_body_hidden": velocity, "gen_suffix_hidden": velocity,
                              "attention": [{"sample": 0, "stage": "write", "gen_to_prompt": 0.5}]}
    monkeypatch.setattr(cli, "NormalRoundEngine", Engine)
    argv = arguments(tmp_path) + ["--seeds", "42,43", "--num-shards", "2"]
    cli.main(argv)
    with pytest.raises(FileNotFoundError):
        cli.main(argv + ["--merge-only"])
    cli.main(argv + ["--shard-id", "1"])
    cli.main(argv + ["--merge-only"])
    out = tmp_path / "out"
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["complete"]
    assert len(manifest["rows"]) == 8
    assert (out / "index.html").is_file()
    assert len(list(out.rglob("*.png"))) == 32
    assert len(list(out.rglob("step_*.pt"))) == 8
    for row in manifest["rows"]:
        assert len((out / row["trace"]).read_text().splitlines()) == 2
        assert set(row["images"]) == set(cli.MODES)
        assert set(row["pixel_mae_vs_r2"]) == set(cli.MODES)
        assert row["pixel_mae_vs_r2"]["normal_r2"] == 0
        trace = json.loads((out / row["trace"]).read_text().splitlines()[0])
        assert trace["reference_arm"] == "normal_r2"
        assert "v_native_norm" not in trace
    with pytest.raises(FileExistsError):
        cli.main(argv)
    (out / manifest["rows"][0]["images"]["normal_r4"]).unlink()
    with pytest.raises(FileNotFoundError):
        cli.main(argv + ["--merge-only"])


def test_default_is_eight_prompts_from_previous_hard64_prefix(tmp_path):
    args = cli.parse_args(["--output-dir", str(tmp_path / "out")])
    plan = cli.make_plan(args)
    assert len(plan["prompts"]) == 8
    assert args.benchmark_data.name == "geneval2_hard_128.jsonl"
    assert plan["arms"] == ["normal_r2", "normal_r4", "normal_r6", "normal_r8"]
    assert cli.stable_noise_seed(42, plan["prompts"][0], schema=cli.NOISE_SCHEMA) == 7992183842731113559


def test_merge_rejects_old_six_arm_protocol(tmp_path):
    args = cli.parse_args(arguments(tmp_path))
    plan = cli.make_plan(args)
    shard = args.output_dir / "shard_000"
    shard.mkdir(parents=True)
    old_plan = {**plan, "schema": "bagel-memory-mechanism-v1"}
    cli.write_json(shard / "run_manifest.json", {"complete": True, "plan": old_plan, "shard_id": 0, "rows": []})
    with pytest.raises(ValueError, match="stale"):
        cli.merge(args.output_dir, plan)


def test_old_completed_manifest_is_not_overwritten(tmp_path):
    args = cli.parse_args(arguments(tmp_path))
    plan = cli.make_plan(args)
    args.output_dir.mkdir()
    path = args.output_dir / "run_manifest.json"
    cli.write_json(path, {"complete": True, "plan": {"schema": "bagel-memory-mechanism-v1"}})
    previous = path.read_bytes()
    with pytest.raises(ValueError, match="previous experiment schema"):
        cli.merge(args.output_dir, plan)
    assert path.read_bytes() == previous
