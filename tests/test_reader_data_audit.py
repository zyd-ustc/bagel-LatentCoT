import json
from pathlib import Path
import subprocess
import sys

import pytest

from qwen_latent_cot.evaluation.offline_reader import sha256
from qwen_latent_cot.evaluation.reader_data_audit import (
    audit_reader_data, lexical_neighbors, prepare_semantic_pack, template_key)


def inputs(tmp_path):
    train = tmp_path / "train.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    train_rows = [dict(prompt_id=f"t{i}", prompt=f"Two red cubes behind {i} trees", category="spatial_relation") for i in range(3)]
    heldout_rows = [dict(prompt_id="h0", prompt="Three blue cubes behind 0 trees", category="spatial_relation"),
                    dict(prompt_id="h1", prompt="Five cats below a blue chair", category="count")]
    for path, rows in ((train, train_rows), (heldout, heldout_rows)):
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    export = tmp_path / "export.json"
    export.write_text(json.dumps(dict(train_records=3, heldout_records=2,
        train_sha256=sha256(train), heldout_sha256=sha256(heldout))))
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(''.join(json.dumps(dict(prompt=f"A unique scene {i}", atom_count=3,
        skills=["object"], vqa_list=[["Scene?", "Yes"]])) + '\n' for i in range(10)))
    return train, heldout, export, benchmark


def test_audit_preserves_splits_and_records_exact_and_template_boundaries(tmp_path):
    train, heldout, export, benchmark = inputs(tmp_path)
    digests = [sha256(path) for path in (train, heldout, export, benchmark)]
    report, selected = audit_reader_data(train, heldout, export_report_path=export,
        evaluated_count=2, benchmark_path=benchmark, semantic_count=8)
    assert report["safe_exact_split"]
    assert report["cross_split"]["count_color_template_heldout_hits"] == 1
    assert report["train"]["absent_categories"]
    assert len(selected) == 8 and selected[0]["prompt_id"] == "geneval2-heldout:0000"
    assert selected[0]["vqa_list"] == [["Scene?", "Yes"]]
    assert not report["semantic_pack"]["teacher_reasoning_generated"]
    assert digests == [sha256(path) for path in (train, heldout, export, benchmark)]
    assert template_key("Two red cubes") == template_key("3 blue cubes")


def test_normalized_overlap_and_duplicate_ids_are_visible(tmp_path):
    train, heldout, _, _ = inputs(tmp_path)
    heldout.write_text(''.join(json.dumps(dict(prompt_id="h", prompt=p, category="count")) + '\n'
                              for p in (" TWO  RED CUBES BEHIND 0 TREES ", "Five cats")))
    report, _ = audit_reader_data(train, heldout, evaluated_count=2)
    assert report["cross_split"]["normalized_prompt_overlap"] == 1
    assert report["heldout"]["duplicate_id_extra_rows"] == 1
    assert not report["safe_exact_split"]


def test_lexical_candidates_are_not_automatically_removed():
    train = [dict(prompt_id="t", prompt="A cat behind a red chair")]
    heldout = [dict(prompt_id="h", prompt="A red chair behind a cat")]
    result = lexical_neighbors(train, heldout)
    assert result["pair_count"] == 1 and result["examples"][0]["jaccard"] == 1.
    assert "not proof" in result["interpretation"]


def test_pack_excludes_train_and_reader_heldout_before_fixed_order_selection(tmp_path):
    _, _, _, benchmark = inputs(tmp_path)
    selected, provenance = prepare_semantic_pack(benchmark,
        [dict(prompt=" A UNIQUE SCENE 0 ")], [dict(prompt="A unique scene 1")], count=8)
    assert selected[0]["source_benchmark_index"] == 2
    assert provenance["skipped_source_indices"] == [0, 1]
    with pytest.raises(ValueError, match="only"):
        prepare_semantic_pack(benchmark, [dict(prompt="A unique scene 0")], [], count=10)
    with pytest.raises(ValueError, match="even"):
        prepare_semantic_pack(benchmark, [], [], count=9)


def test_pack_is_balanced_across_atomicities_and_refuses_short_groups(tmp_path):
    _, _, _, benchmark = inputs(tmp_path)
    rows = [json.loads(line) for line in benchmark.read_text().splitlines()]
    for i, row in enumerate(rows):
        row["atom_count"] = 3 if i < 5 else 4
    benchmark.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    selected, report = prepare_semantic_pack(benchmark, [], [], count=8)
    assert report["atomicity_counts"] == {"3": 4, "4": 4}
    assert [row["source_benchmark_index"] for row in selected] == [0, 1, 2, 3, 5, 6, 7, 8]
    with pytest.raises(ValueError, match="group"):
        prepare_semantic_pack(benchmark, [dict(prompt=f"A unique scene {i}") for i in range(2)], [], count=8)


def test_export_digest_tampering_is_rejected(tmp_path):
    train, heldout, export, _ = inputs(tmp_path)
    train.write_text(train.read_text() + '\n')
    with pytest.raises(ValueError, match="hash"):
        audit_reader_data(train, heldout, export_report_path=export, evaluated_count=2)


def test_audit_cli_fresh_outputs_and_actual_manifest_count(tmp_path):
    train, heldout, export, benchmark = inputs(tmp_path)
    output = tmp_path / "report"
    command = [sys.executable, "scripts/data/audit_reader_data.py", "--train-data", str(train),
               "--heldout-data", str(heldout), "--export-report", str(export),
               "--semantic-benchmark", str(benchmark), "--semantic-count", "8",
               "--evaluated-count", "2", "--output-dir", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    manifest = output / "phase1a_semantic_hard8.jsonl"
    assert len(manifest.read_text().splitlines()) == 8
    report = json.loads((output / "data_audit.json").read_text())
    assert report["semantic_pack"]["output_manifest_sha256"] == sha256(manifest)
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_committed_semantic_pack_is_source_bound_and_balanced():
    root = Path(__file__).resolve().parents[1]
    manifest = root / "experiments/data/phase1a_semantic_hard64.jsonl"
    provenance = json.loads(manifest.with_suffix(".provenance.json").read_text())
    source = root / provenance["source"]
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    originals = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(rows) == 64 and sha256(manifest) == provenance["output_manifest_sha256"]
    assert sha256(source) == provenance["source_sha256"]
    assert provenance["atomicity_counts"] == {"7": 16, "8": 16, "9": 16, "10": 16}
    assert len({row["prompt_id"] for row in rows}) == 64
    for row in rows:
        original = originals[row["source_benchmark_index"]]
        for key in ("prompt", "atom_count", "skills", "vqa_list"):
            assert row[key] == original[key]
        assert row["split"] == "heldout" and "reasoning_text" not in row
