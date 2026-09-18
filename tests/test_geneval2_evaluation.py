import json
import math
from pathlib import Path

import pytest

from qwen_latent_cot.evaluation.geneval2 import (
    build_image_filepath_map,
    compare_summaries,
    geometric_mean,
    load_benchmark,
    load_score_lists,
    render_markdown,
    summarize_score_lists,
    write_atomicity_csv,
    write_skill_csv,
    write_summary_csv,
)


def _write_benchmark(path: Path) -> None:
    rows = [
        {
            "prompt": "a red cube and two dogs",
            "atom_count": 3,
            "skills": ["attribute", "object", "count"],
            "vqa_list": [["q", "a"], ["q", "a"], ["q", "a"]],
        },
        {
            "prompt": "a cat left of a chair",
            "atom_count": 4,
            "skills": ["object", "position"],
            "vqa_list": [["q", "a"], ["q", "a"]],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_geneval2_summary_matches_official_am_gm_and_skill_counts(tmp_path: Path):
    benchmark_path = tmp_path / "geneval2.jsonl"
    _write_benchmark(benchmark_path)
    benchmark = load_benchmark(benchmark_path)
    scores = [[0.8, 0.6, 1.0], [0.5, 0.25]]

    summary = summarize_score_lists(benchmark, scores, name="run")

    expected_am = 100 * (((0.8 + 0.6 + 1.0) / 3) + ((0.5 + 0.25) / 2)) / 2
    expected_gm = 100 * (
        ((0.8 * 0.6 * 1.0) ** (1 / 3)) + math.sqrt(0.5 * 0.25)
    ) / 2
    assert summary["overall"]["soft_tifa_am"] == pytest.approx(expected_am)
    assert summary["overall"]["soft_tifa_gm"] == pytest.approx(expected_gm)
    assert summary["overall"]["atom_weighted_am"] == pytest.approx(100 * 3.15 / 5)
    assert summary["skills"]["object"]["count"] == 2
    assert summary["skills"]["object"]["soft_tifa_am"] == pytest.approx(55.0)
    assert summary["skills"]["position"]["soft_tifa_am"] == pytest.approx(25.0)
    assert summary["atomicity"]["3"]["soft_tifa_gm"] == pytest.approx(
        100 * ((0.8 * 0.6 * 1.0) ** (1 / 3))
    )


def test_load_score_lists_accepts_prompt_mapping(tmp_path: Path):
    benchmark_path = tmp_path / "geneval2.jsonl"
    _write_benchmark(benchmark_path)
    benchmark = load_benchmark(benchmark_path)
    score_path = tmp_path / "scores.json"
    score_path.write_text(
        json.dumps(
            {
                "a cat left of a chair": [0.2, 0.3],
                "a red cube and two dogs": [0.7, 0.8, 0.9],
            }
        ),
        encoding="utf-8",
    )

    assert load_score_lists(score_path, benchmark) == [[0.7, 0.8, 0.9], [0.2, 0.3]]


def test_compare_and_markdown_report(tmp_path: Path):
    benchmark_path = tmp_path / "geneval2.jsonl"
    _write_benchmark(benchmark_path)
    benchmark = load_benchmark(benchmark_path)
    base = summarize_score_lists(benchmark, [[0.5, 0.5, 0.5], [0.5, 0.5]], name="base")
    edit = summarize_score_lists(benchmark, [[0.7, 0.5, 0.9], [0.6, 0.8]], name="edit")

    report = compare_summaries([base, edit], baseline_name="base")
    assert report["deltas"]["edit"]["overall"]["soft_tifa_am"] > 0
    markdown = render_markdown(report)
    assert "GenEval2 Report" in markdown
    assert "| edit |" in markdown

    summary_csv = tmp_path / "summary.csv"
    skill_csv = tmp_path / "skills.csv"
    atomicity_csv = tmp_path / "atomicity.csv"
    write_summary_csv(report, summary_csv)
    write_skill_csv(report, skill_csv)
    write_atomicity_csv(report, atomicity_csv)
    assert "delta_soft_tifa_gm" in summary_csv.read_text(encoding="utf-8")
    assert "position" in skill_csv.read_text(encoding="utf-8")
    assert "atomicity" in atomicity_csv.read_text(encoding="utf-8")


def test_build_image_filepath_map_ordered_layout(tmp_path: Path):
    benchmark_path = tmp_path / "geneval2.jsonl"
    _write_benchmark(benchmark_path)
    benchmark = load_benchmark(benchmark_path)
    image_root = tmp_path / "images"
    (image_root / "00000" / "samples").mkdir(parents=True)
    (image_root / "00001" / "samples").mkdir(parents=True)
    (image_root / "00000" / "samples" / "00000.png").write_bytes(b"png")
    (image_root / "00001" / "samples" / "00000.png").write_bytes(b"png")

    image_map = build_image_filepath_map(benchmark, image_root)

    assert image_map["a red cube and two dogs"].endswith("00000/samples/00000.png")
    assert image_map["a cat left of a chair"].endswith("00001/samples/00000.png")


def test_geometric_mean_handles_zero():
    assert geometric_mean([0.0, 0.5]) == 0.0
