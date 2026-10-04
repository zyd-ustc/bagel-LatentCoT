"""True multi-process checks for sharding, gradients and shared checkpoints."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

from qwen_latent_cot.bagel.anchored_loop import LoopConfig
from qwen_latent_cot.data.t2i import BucketBatchSampler, T2IDataset

ROOT = Path(__file__).resolve().parents[1]


def run_workers(kind, argument, world_size=2):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         f"--nproc_per_node={world_size}", str(ROOT / "tests/helpers/stage1_ddp_worker.py"),
         kind, str(argument)],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_bucket_rank_shards_reconstruct_global_draws(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [{"prompt": str(i), "image": "x.png", "bucket": bucket}
            for bucket in ("ordinary", "structural", "easy", "noop")
            for i in range(32)]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    dataset = T2IDataset(path)
    expected = list(BucketBatchSampler(dataset, 8, 30, seed=4))
    sharded = [list(BucketBatchSampler(dataset, 1, 30, seed=4, rank=rank, world_size=8))
               for rank in range(8)]
    assert [[draw for rank in sharded for draw in rank[step]]
            for step in range(30)] == expected
    for step in range(30):
        assert len({dataset.rows[rank[step][0]]["bucket"] for rank in sharded}) == 1
    with pytest.raises(ValueError, match="rank/world_size"):
        BucketBatchSampler(dataset, 1, 1, rank=8, world_size=8)


def test_ddp_loss_matches_global_mean_with_unequal_token_counts(tmp_path):
    result = tmp_path / "result.json"
    run_workers("token_mean", result)
    assert json.loads(result.read_text()) == {"gradient": -0.5, "ranks": 2}


@pytest.mark.parametrize("mode,world_size", [
    ("gen_only", 2), ("gen_memory_anchored", 2), ("gen_memory_anchored", 8),
])
def test_stage1_real_cli_ddp_syncs_models_and_rank0_output(tmp_path, mode, world_size):
    Image.new("RGB", (16, 16), "red").save(tmp_path / "image.png")
    Image.new("RGB", (32, 16), "blue").save(tmp_path / "wide.png")
    rows = [{"prompt": f"two cubes {i}", "image": "image.png" if i % 2 else "wide.png",
             "bucket": "structural"} for i in range(32)]
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows))
    config = LoopConfig(enable_t2i_loop=True, loop_start_layer=1, loop_end_layer=3,
                        runtime_loop_depth=2, loop_mode=mode,
                        memory_slots=0 if mode == "gen_only" else 2,
                        loop_output_alpha_init=0.01 if mode == "gen_only" else 0.0)
    out = tmp_path / "out"
    cfg = {"model_path": str(tmp_path / "base"), "data_path": str(data),
           "output_dir": str(out), "device": "cpu", "image_size": 16,
           "batch_size": 1, "steps": 3, "save_every": 1, "learning_rate": 0.01,
           "depth_curriculum": [{"until": 1 / 3, "depths": [1]},
                                {"until": 1, "depths": [1, 2]}],
           "loop": config.to_dict()}
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run_workers("cli", path, world_size)
    states = [json.loads((out / f"rank_{rank}_state.json").read_text()) for rank in range(world_size)]
    assert len({state["loop_sha256"] for state in states}) == 1
    assert all(value["native_before"] == value["native_after"] for value in states)
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 3
    assert records[0]["depth"] == 1 and records[1]["depth"] == 2
    assert all(record["distributed_training"]["global_batch_size"] == world_size for record in records)
    assert all({item["rank"] for item in record["per_rank"]} == set(range(world_size)) for record in records)
    assert any(record["per_rank"][0]["indices"] != record["per_rank"][1]["indices"] for record in records)
    assert records[-1]["round_training_steps"][1] >= 1
    for step in range(1, 4):
        checkpoint = out / f"step_{step:06d}"
        assert (checkpoint / "loop.safetensors").is_file()
        assert json.loads((checkpoint / "loop.json").read_text())["distributed_training"]["world_size"] == world_size
