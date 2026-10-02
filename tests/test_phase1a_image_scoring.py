"""No GPU/VLM: real tiny PNG fixtures and fake local Soft-TIFA responses."""

import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from qwen_latent_cot.evaluation.geneval2 import (
    compare_summaries, load_benchmark, load_score_lists, summarize_score_lists)
from qwen_latent_cot.evaluation.image_scoring import prepare_phase1a_image_maps, score_image_map
from qwen_latent_cot.evaluation.offline_reader import sha256


def fixtures(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    rows = [dict(prompt=f"{i} colored cubes", atom_count=3 + i,
                 skills=["object", "count"], vqa_list=[["Cubes?", "Yes"], ["How many?", str(i)]])
            for i in range(3)]
    benchmark.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    root = tmp_path / "images"
    for i, row in enumerate(rows):
        folder = root / f"p{i:03d}"
        folder.mkdir(parents=True)
        (folder / "prompt.txt").write_text(row["prompt"] + '\n')
        for arm in ("native", "teacher"):
            Image.new("RGB", (8, 8), (i * 50, 0, 0)).save(folder / f"{arm}.png")
    maps = prepare_phase1a_image_maps(benchmark, root, ["native", "teacher"])
    image_map = tmp_path / "map.json"
    image_map.write_text(json.dumps(maps["image_maps"]["native"]))
    return benchmark, root, image_map, rows


def fake_server(calls, change=None):
    def post(url, *, data, timeout):
        payload = pickle.loads(data)
        calls.append(payload)
        assert url == "http://127.0.0.1:18086" and timeout > 0
        assert payload["only_strict"] is True
        values = [[.5, .8] for _ in payload["meta_datas"]]
        result = dict(atom_scores=values, scores=[math.log(.4) / 2] * len(values),
                      prompt_order=[row["prompt"] for row in payload["meta_datas"]])
        if change:
            change(result)
        return SimpleNamespace(content=pickle.dumps(result), raise_for_status=lambda: None)
    return post


def test_image_map_and_scoring_preserve_coverage_order_hashes_and_final_batch(tmp_path):
    benchmark, root, image_map, rows = fixtures(tmp_path)
    calls = []
    result = score_image_map(benchmark, image_map, server_url="http://127.0.0.1:18086",
                             batch_size=2, post=fake_server(calls))
    assert [len(call["images"]) for call in calls] == [2, 1]
    assert result["prompt_order"] == [row["prompt"] for row in rows]
    assert result["benchmark_sha256"] == sha256(benchmark)
    assert len(result["provenance"]["image_sha256"]) == 3
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps(result, allow_nan=False))
    assert load_score_lists(scores, load_benchmark(benchmark)) == [[.5, .8]] * 3
    summary = summarize_score_lists(load_benchmark(benchmark), result["score_lists"], name="native")
    assert summary["overall"]["soft_tifa_am"] == 65.
    assert summary["overall"]["soft_tifa_gm"] == pytest.approx(math.sqrt(.4) * 100)
    assert summary["skills"]["verb"]["count"] == 0
    assert summary["skills"]["verb"]["soft_tifa_am"] is None
    json.dumps(compare_summaries([summary], baseline_name="native"), allow_nan=False)


@pytest.mark.parametrize("case", ["missing", "corrupt", "prompt", "arm", "reuse", "extra_map", "empty_batch"])
def test_input_failures_are_rejected_before_any_scoring(tmp_path, case):
    benchmark, root, image_map, rows = fixtures(tmp_path)
    calls = []
    if case == "missing":
        (root / "p002/native.png").unlink()
    elif case == "corrupt":
        (root / "p002/native.png").write_bytes(b"not a png")
    elif case == "prompt":
        (root / "p000/prompt.txt").write_text("wrong prompt")
        with pytest.raises(ValueError, match="prompt.txt"):
            prepare_phase1a_image_maps(benchmark, root, ["native"])
        return
    elif case == "arm":
        with pytest.raises(ValueError):
            prepare_phase1a_image_maps(benchmark, root, ["native", "native"])
        return
    elif case in ("reuse", "extra_map"):
        paths = json.loads(image_map.read_text())
        if case == "reuse":
            paths[rows[1]["prompt"]] = paths[rows[0]["prompt"]]
        else:
            paths["unexpected"] = paths[rows[0]["prompt"]]
        image_map.write_text(json.dumps(paths))
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        score_image_map(benchmark, image_map, server_url="http://127.0.0.1:18086",
                        batch_size=0 if case == "empty_batch" else 2, post=fake_server(calls))
    assert calls == []


@pytest.mark.parametrize("case", ["short", "atom", "nan", "bool", "range", "log", "reorder", "server_error"])
def test_malformed_server_results_cannot_become_scores(tmp_path, case):
    benchmark, _, image_map, _ = fixtures(tmp_path)
    def change(result):
        if case == "short":
            result["atom_scores"].pop()
        elif case == "atom":
            result["atom_scores"][0].pop()
        elif case in ("nan", "bool", "range"):
            result["atom_scores"][0][0] = {"nan": float("nan"), "bool": True, "range": 1.1}[case]
        elif case == "log":
            result["scores"][0] = .9
        elif case == "reorder":
            result["prompt_order"].reverse()
        else:
            result["error"] = "scorer unavailable"
    with pytest.raises((ValueError, RuntimeError)):
        score_image_map(benchmark, image_map, server_url="http://127.0.0.1:18086",
                        batch_size=2, post=fake_server([], change))


@pytest.mark.parametrize("case", ["bool", "nan", "negative", "range", "count", "order", "hash", "extra_prompt"])
def test_bad_cached_scores_rejected(tmp_path, case):
    benchmark, _, _, rows = fixtures(tmp_path)
    model = load_benchmark(benchmark)
    payload = dict(score_lists=[[.5, .8]] * 3, benchmark_sha256=model.source_sha256,
                   prompt_order=[row["prompt"] for row in rows])
    if case in ("bool", "nan", "negative", "range"):
        payload["score_lists"] = [[{"bool": True, "nan": float("nan"), "negative": -.1, "range": 1.1}[case], .8]] * 3
    elif case == "count":
        payload["score_lists"].pop()
    elif case == "order":
        payload["prompt_order"].reverse()
    elif case == "hash":
        payload["benchmark_sha256"] = "wrong"
    else:
        payload = {row["prompt"]: [.5, .8] for row in rows} | {"extra prompt": [.5, .8]}
    path = tmp_path / "bad_scores.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_score_lists(path, model)


def test_report_and_image_input_cli_fresh_outputs_and_no_nan(tmp_path):
    benchmark, root, _, _ = fixtures(tmp_path)
    output = tmp_path / "maps"
    command = [sys.executable, "scripts/evaluate/prepare_phase1a_score_inputs.py", "--benchmark-data", str(benchmark),
               "--image-dir", str(root), "--arm", "native", "--output-dir", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    assert (output / "native_image_map.json").is_file()
    assert subprocess.run(command, capture_output=True).returncode != 0
    native, teacher = tmp_path / "native.json", tmp_path / "teacher.json"
    native.write_text(json.dumps([[.5, .5]] * 3))
    teacher.write_text(json.dumps([[.7, .7]] * 3))
    report = tmp_path / "report"
    command = [sys.executable, "scripts/evaluate/geneval2_report.py", "--benchmark-data", str(benchmark),
               "--run", f"native={native}", "--run", f"teacher={teacher}", "--baseline-run", "native", "--output-dir", str(report)]
    subprocess.run(command, check=True, capture_output=True)
    raw = (report / "geneval2_summary.json").read_text()
    summary = json.loads(raw)
    assert "NaN" not in raw
    assert summary["deltas"]["teacher"]["overall"]["soft_tifa_am"] == pytest.approx(20.)
    assert subprocess.run(command, capture_output=True).returncode != 0
    with pytest.raises(ValueError, match="unique"):
        compare_summaries([summary["runs"][0]] * 2)


def test_benchmark_rejects_duplicate_prompt_and_misaligned_questions(tmp_path):
    benchmark, _, _, rows = fixtures(tmp_path)
    rows.append(rows[0])
    benchmark.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match="duplicate"):
        load_benchmark(benchmark)
    rows = rows[:3]
    rows[0]["vqa_list"].pop()
    benchmark.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match="alignment"):
        load_benchmark(benchmark)


def test_arm_image_geometry_must_match(tmp_path):
    benchmark, root, _, _ = fixtures(tmp_path)
    Image.new("RGB", (16, 16)).save(root / "p002/teacher.png")
    with pytest.raises(ValueError, match="geometry"):
        prepare_phase1a_image_maps(benchmark, root, ["native", "teacher"])
