"""Protocol fixtures verify report math and interfaces, not image quality."""

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image
from test_t2i_loop_entrypoints import script_module

from qwen_latent_cot.evaluation.loop_results import (
    LocalVLMJudge,
    OfficialGenEval2,
    merge_manifests,
    score_manifest,
    summarize_human_pairwise,
    summarize_results,
    validate_required_arms,
    write_reports,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = "base_R0_K0_correct"
GEN = "gen_only_R1_K0_correct"
ARMS = [
    BASE,
    GEN,
    "legacy_memory_only_R1_K8_correct",
    "gen_memory_anchored_R1_K8_correct",
]


def fixture_matrix(tmp_path):
    benchmarks = [
        {
            "prompt": f"fixture {i}",
            "vqa_list": [f"question {i}"],
            "yn_question_list": [f"question {i}"],
            "yn_answer_list": ["yes"],
        }
        for i in range(4)
    ]
    images = []
    for arm in ARMS:
        for i, benchmark in enumerate(benchmarks):
            path = tmp_path / f"{arm}_{i}.png"
            Image.new("RGB", (8, 8), (i * 50, 30, 60)).save(path)
            images.append(
                dict(
                    index=i,
                    seed=5,
                    prompt=benchmark["prompt"],
                    arm=arm,
                    path=str(path),
                    depth_status="unseen",
                )
            )
    return {
        "images": images,
        "arguments": {"seed": 5, "model_path": "native"},
    }, benchmarks


def protocol_scores(rows, benchmarks):
    return [
        [float(row["index"] in ({0, 1} if row["arm"] == BASE else {1, 2}))]
        for row in rows
    ]


def fixture_quality(path):
    return {"quality_proxy": 0.5 if BASE in path else 0.75, "invalid": False}


def test_paired_repair_damage_and_quality_denominators(tmp_path):
    manifest, benchmark = fixture_matrix(tmp_path)
    validate_required_arms(manifest, [0, 1])
    scored = score_manifest(
        manifest,
        benchmark,
        dataset="fixture",
        semantic_scorer=protocol_scores,
        quality_judge=fixture_quality,
    )
    summaries = summarize_results(scored)
    gen = next(row for row in summaries if row["arm"] == GEN)
    assert gen["semantic_GM"] == 0.5
    assert gen["semantic_GM_delta_vs_base"]["mean"] == 0
    assert gen["quality_delta_vs_base"]["mean"] == 0.25
    assert gen["quality_delta_vs_base"]["ci95"] == [0.25, 0.25]
    assert gen["depth_status"] == ["unseen"]
    metrics = gen["repair_damage"]
    assert metrics["repair_count"] == metrics["damage_count"] == 1
    assert metrics["base_fail_count"] == metrics["base_pass_count"] == 2
    assert (
        metrics["repair_rate_given_base_fail"]
        == metrics["damage_rate_given_base_pass"]
        == 0.5
    )
    assert (
        metrics["repair_fraction_all_pairs"]
        == metrics["damage_fraction_all_pairs"]
        == 0.25
    )
    assert summarize_results(scored) == summaries


def test_missing_scorers_remain_unscored_and_corrupt_images_fail(tmp_path):
    manifest, benchmark = fixture_matrix(tmp_path)
    Path(manifest["images"][4]["path"]).write_text("corrupted PNG")
    missing = score_manifest(manifest, benchmark, dataset="fixture")
    gen = next(row for row in summarize_results(missing) if row["arm"] == GEN)
    assert (
        gen["semantic_GM"] is gen["quality_proxy_mean"] is gen["invalid_rate"] is None
    )
    assert gen["repair_damage"] is None
    assert gen["semantic_status"] == gen["quality_status"] == "unscored"
    assert gen["decode_invalid_rate"] == 0.25
    scored = score_manifest(
        manifest,
        benchmark,
        dataset="fixture",
        semantic_scorer=protocol_scores,
        quality_judge=fixture_quality,
    )
    corrupt = next(row for row in scored if row["arm"] == GEN and row["index"] == 0)
    assert corrupt["semantic_atoms"] == [0.0]
    assert corrupt["quality_proxy"] == 0 and corrupt["judge_invalid"] is True


def test_constraint_repair_is_separate_from_prompt_repair_and_previous_depth(tmp_path):
    manifest, benchmark = fixture_matrix(tmp_path)
    for record in benchmark:
        record["vqa_list"].append("second constraint")
    r2 = GEN.replace("_R1_", "_R2_")
    manifest["images"].extend(
        [{**row, "arm": r2} for row in manifest["images"] if row["arm"] == GEN]
    )
    base_scores = [[1, 0], [1, 1], [0, 0], [1, 1]]
    r1_scores = [[1, 1], [1, 0], [1, 0], [1, 1]]
    r2_scores = [[1, 1], [1, 1], [1, 1], [1, 0]]

    def scorer(rows, _):
        return [
            (
                base_scores
                if row["arm"] == BASE
                else r2_scores
                if row["arm"] == r2
                else r1_scores
            )[row["index"]]
            for row in rows
        ]

    scored = score_manifest(
        manifest, benchmark, dataset="fixture", semantic_scorer=scorer
    )
    summaries = summarize_results(scored)
    gen = next(row for row in summaries if row["arm"] == GEN)
    assert (
        gen["repair_damage"]["repair_count"]
        == gen["repair_damage"]["damage_count"]
        == 1
    )
    constraints = gen["constraint_repair_damage_vs_base"]
    assert constraints["repair_count"] == 2 and constraints["damage_count"] == 1
    assert constraints["repair_rate_given_reference_fail"] == pytest.approx(2 / 3)
    assert constraints["damage_rate_given_reference_pass"] == pytest.approx(1 / 5)
    assert constraints["paired_constraints"] == 8
    r2_summary = next(row for row in summaries if row["arm"] == r2)
    assert r2_summary["previous_depth_arm"] == GEN
    assert r2_summary["constraint_repair_damage_vs_previous"]["repair_count"] == 2
    assert r2_summary["constraint_repair_damage_vs_previous"]["damage_count"] == 1


def test_bad_pair_coverage_prompt_scores_and_index_are_rejected(tmp_path):
    manifest, benchmark = fixture_matrix(tmp_path)
    scored = score_manifest(manifest, benchmark, dataset="fixture")
    with pytest.raises(ValueError, match="full selected benchmark"):
        score_manifest(
            {**manifest, "images": manifest["images"][:-1]},
            benchmark,
            dataset="fixture",
        )
    with pytest.raises(ValueError, match="coverage"):
        summarize_results(scored[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        score_manifest(
            manifest, benchmark, dataset="fixture", semantic_scorer=lambda *_: []
        )
    with pytest.raises(ValueError, match="invalid atom"):
        score_manifest(
            manifest,
            benchmark,
            dataset="fixture",
            semantic_scorer=lambda rows, _: [[float("nan")]] * len(rows),
        )
    manifest["images"][0]["prompt"] = "wrong prompt"
    with pytest.raises(ValueError, match="prompt"):
        score_manifest(manifest, benchmark, dataset="fixture")
    manifest["images"][0]["index"] = -1
    with pytest.raises(ValueError, match="out of range"):
        score_manifest(manifest, benchmark, dataset="fixture")


def test_required_elastic_arms_cannot_be_silently_omitted(tmp_path):
    manifest, _ = fixture_matrix(tmp_path)
    with pytest.raises(ValueError, match="R=2"):
        validate_required_arms(manifest, [0, 1, 2, 3, 4])
    manifest["images"] = [row for row in manifest["images"] if row["arm"] != BASE]
    with pytest.raises(ValueError, match="Base R0"):
        validate_required_arms(manifest, [0, 1])


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("decoded_shape", [4, 4], "dimensions"),
        ("initial_noise_sha256", "different_noise", "noise"),
    ],
)
def test_paired_shape_and_noise_must_match(tmp_path, field, value, error):
    manifest, benchmark = fixture_matrix(tmp_path)
    scored = score_manifest(manifest, benchmark, dataset="fixture")
    scored[4][field] = value
    with pytest.raises(ValueError, match=error):
        summarize_results(scored)


def test_cross_checkpoint_manifest_selection_requires_matched_generation(tmp_path):
    manifest, _ = fixture_matrix(tmp_path)
    first, second = tmp_path / "gen.json", tmp_path / "memory.json"
    first.write_text(json.dumps(manifest))
    second.write_text(json.dumps(manifest))
    specifications = [
        {"path": str(first), "modes": ["base", "legacy_memory_only", "gen_only"]},
        {"path": str(second), "modes": ["gen_memory_anchored"]},
    ]
    merged = merge_manifests(specifications)
    assert len(merged["images"]) == 16
    manifest["arguments"]["seed"] = 99
    second.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="settings differ"):
        merge_manifests(specifications)
    manifest["arguments"]["seed"] = 5
    manifest["images"][0]["prompt"] = "changed"
    second.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate selected"):
        merge_manifests([str(first), str(second)])
    manifest["images"][0]["prompt"] = "fixture 0"
    manifest["allocated_loop_config"] = {"loop_start_layer": 1, "loop_end_layer": 3}
    second.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="settings differ"):
        merge_manifests(specifications)


def install_official_interface_fixture(tmp_path):
    root = tmp_path / "official_interface_fixture"
    root.mkdir()
    (root / "evaluation.py").write_text("""import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
for name in ["benchmark_data", "image_filepath_data", "method", "output_file"]:
    p.add_argument("--" + name, required=True)
a = p.parse_args()
assert a.method == "soft_tifa_gm"
b = [json.loads(line) for line in Path(a.benchmark_data).read_text().splitlines()]
images = json.loads(Path(a.image_filepath_data).read_text())
assert all(Path(images[row["prompt"]]).is_file() for row in b)
Path(a.output_file).write_text(json.dumps([[.8]*len(row["vqa_list"]) for row in b]))
""")
    return root


def test_official_geneval2_interface_and_repeated_prompt_seed_mapping(tmp_path):
    manifest, benchmarks = fixture_matrix(tmp_path)
    scorer = OfficialGenEval2(
        install_official_interface_fixture(tmp_path), tmp_path / "scores"
    )
    rows = manifest["images"][:4]
    scores = scorer(rows + [{**rows[0], "seed": 6}], benchmarks + [benchmarks[0]])
    assert scores == [[0.8]] * 5
    assert len(scorer.provenance["script_sha256"]) == 64
    exported = tmp_path / "scores" / BASE / "repeat_000" / "benchmark.jsonl"
    assert [
        json.loads(line) for line in exported.read_text().splitlines()
    ] == benchmarks


def test_local_tiif_protocol_and_quality_json_are_strict():
    judge = object.__new__(LocalVLMJudge)
    judge.answer = lambda *_: "Yes."
    assert judge.tiif(
        [{"path": "fixture"}],
        [{"yn_question_list": ["left?"], "yn_answer_list": ["yes"]}],
    ) == [[1.0]]
    judge.answer = lambda *_: "probably"
    with pytest.raises(ValueError, match="no silent fallback"):
        judge.tiif(
            [{"path": "fixture"}],
            [{"yn_question_list": ["left?"], "yn_answer_list": ["yes"]}],
        )
    judge.answer = lambda *_: '```json\n{"quality":4,"invalid":false}\n```'
    assert judge.quality("fixture") == {"quality_proxy": 0.75, "invalid": False}
    judge.answer = lambda *_: '{"quality":true,"invalid":false}'
    with pytest.raises(ValueError, match="quality 1..5"):
        judge.quality("fixture")


def test_human_review_preserved_on_rerun_and_checks_image_hashes(tmp_path):
    manifest, benchmark = fixture_matrix(tmp_path)
    scored = score_manifest(
        manifest, benchmark, dataset="fixture", semantic_scorer=protocol_scores
    )
    summaries = summarize_results(scored)
    write_reports(tmp_path / "report", scored, summaries, {"kind": "protocol_fixture"})
    template = next((tmp_path / "report").glob("human_pairwise_*.csv"))
    with template.open() as handle:
        reviewed = list(csv.DictReader(handle))
    reviewed[0]["preference"] = "loop"
    with template.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reviewed[0]))
        writer.writeheader()
        writer.writerows(reviewed)
    snapshot = template.read_bytes()
    write_reports(tmp_path / "report", scored, summaries, {})
    assert template.read_bytes() == snapshot
    result = summarize_human_pairwise(template, scored)
    assert result[("fixture", GEN)]["loop_wins"] == 1
    scored[4]["image_sha256"] = "changed"
    with pytest.raises(ValueError, match="changed images"):
        summarize_human_pairwise(template, scored)


def test_result_evaluator_cli_scores_three_datasets_with_protocol_fixtures(
    tmp_path, monkeypatch
):
    manifest, benchmark = fixture_matrix(tmp_path)
    generation, questions = tmp_path / "manifest.json", tmp_path / "benchmark.jsonl"
    generation.write_text(json.dumps(manifest))
    questions.write_text("\n".join(json.dumps(row) for row in benchmark))
    official = install_official_interface_fixture(tmp_path)
    settings = dict(
        output_dir=str(tmp_path / "report"),
        geneval2_root=str(official),
        geneval2_python=sys.executable,
        judge_model_path="fixture",
        judge_device="cpu",
        required_depths=[0, 1],
        datasets={
            kind: {
                "kind": kind,
                "manifest": str(generation),
                "benchmark": str(questions),
            }
            for kind in ["geneval2_hard", "tiif_spatial", "quality"]
        },
    )
    config = tmp_path / "eval.yaml"
    config.write_text(yaml.safe_dump(settings))
    module = script_module(ROOT / "scripts/evaluate/evaluate_t2i_loops.py")

    class ProtocolFixtureJudge:
        provenance = {"kind": "protocol_fixture_not_quality_evidence"}

        def __init__(self, *_args, **_kwargs):
            pass

        tiif = staticmethod(protocol_scores)
        quality = staticmethod(fixture_quality)

    monkeypatch.setattr(module, "LocalVLMJudge", ProtocolFixtureJudge)
    monkeypatch.setattr("sys.argv", ["evaluate_t2i_loops.py", "--config", str(config)])
    module.main()
    report = json.loads((tmp_path / "report/summary.json").read_text())
    assert len(report["summaries"]) == 12
    assert (
        report["provenance"]["judge"]["kind"] == "protocol_fixture_not_quality_evidence"
    )
    assert all(row["quality_status"] == "proxy_scored" for row in report["summaries"])
    assert all(
        row["semantic_GM"] is None
        for row in report["summaries"]
        if row["dataset"] == "quality"
    )
    assert (tmp_path / "report/scores.jsonl").is_file()


def test_tiif_normalization_preserves_original_questions_and_answers(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    records = [
        {
            "type": "spatial",
            "short_description": "red left of blue",
            "long_description": "long description",
            "yn_question_list": ["Is red left of blue?"],
            "yn_answer_list": ["yes"],
        },
        {"type": "color"},
    ]
    (source / "native.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    output = tmp_path / "spatial.jsonl"
    module = script_module(ROOT / "scripts/evaluate/prepare_tiif_spatial.py")
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_tiif_spatial.py",
            "--source-dir",
            str(source),
            "--output",
            str(output),
        ],
    )
    module.main()
    result = json.loads(output.read_text())
    assert result["prompt"] == records[0]["short_description"]
    assert result["yn_question_list"] == records[0]["yn_question_list"]
    assert result["yn_answer_list"] == records[0]["yn_answer_list"]
    assert result["source_index"] == 0
