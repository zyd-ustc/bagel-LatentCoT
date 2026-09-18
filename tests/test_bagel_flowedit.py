from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from qwen_latent_cot.bagel.inferencer import flowedit_time_branch

EVAL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "evaluate"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from bagel_flowedit_zeroshot import load_pairs, shard_pairs  # noqa: E402


def test_flowedit_window_uses_shifted_t_not_step_index():
    assert flowedit_time_branch(1.0, 0.2, 0.8) == "src"
    assert flowedit_time_branch(0.9, 0.2, 0.8) == "src"
    assert flowedit_time_branch(0.8, 0.2, 0.8) == "delta"
    assert flowedit_time_branch(0.5, 0.2, 0.8) == "delta"
    assert flowedit_time_branch(0.2, 0.2, 0.8) == "delta"
    assert flowedit_time_branch(0.19, 0.2, 0.8) == "tar"
    assert flowedit_time_branch(0.0, 0.2, 0.8) == "tar"


def test_full_window_is_all_delta():
    assert flowedit_time_branch(1.0, 0.0, 1.0) == "delta"
    assert flowedit_time_branch(0.0, 0.0, 1.0) == "delta"


def test_shard_pairs_round_robin():
    pairs = [{"c_src": str(i), "c_tar": str(i)} for i in range(16)]
    shard0 = shard_pairs(pairs, 0, 16)
    shard1 = shard_pairs(pairs, 1, 16)
    assert [i for i, _ in shard0] == [0]
    assert [i for i, _ in shard1] == [1]
    combined = [i for s in range(16) for i, _ in shard_pairs(pairs, s, 16)]
    assert combined == list(range(16))


def test_hard_16_flowedit_pairs_file():
    path = Path(__file__).resolve().parents[1] / "experiments" / "data" / "geneval2_hard_16_flowedit.jsonl"
    args = SimpleNamespace(pairs_file=str(path), c_src="x", c_tar="y")
    rows = load_pairs(args)
    assert len(rows) == 16
    assert rows[0]["c_tar"].startswith("three white bagels")
    assert all(row["c_src"] and row["c_tar"] and row["c_src"] != row["c_tar"] for row in rows)
