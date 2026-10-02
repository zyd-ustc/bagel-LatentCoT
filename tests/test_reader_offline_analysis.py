"""Synthetic CPU fixtures validate analysis, not BAGEL semantic performance."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from qwen_latent_cot.evaluation.offline_reader import (
    analyze_reader, paired_prompt_summary, render_reader_report, sha256, step_bucket)


def evidence(tmp_path):
    main = tmp_path / "main"
    main.mkdir()
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(''.join(json.dumps(dict(prompt_id=f"h{i}", prompt=f"{i} cubes", category="count")) + '\n' for i in range(2)))
    (main / "resolved_config.json").write_text(json.dumps(dict(num_steps=50)))
    (main / "run_manifest.json").write_text(json.dumps(dict(heldout_prompt_data_sha256=sha256(heldout))))
    rows = []
    for step, correct in ((0, 7.), (1, 3.)):
        states = [dict(prompt_id=f"h{i}", donor_prompt_id=f"h{1-i}", timestep=t,
                       step_index=index, errors=dict(correct=correct, shuffled=4., zero=2.))
                  for i in range(2) for index, t in ((1, .9), (30, .4))]
        rows.append(dict(step=step, heldout_error_mse=dict(correct=correct, shuffled=4., zero=2.),
                         native_parity_max_abs=0., per_state=states,
                         per_layer={"12": dict(reader_mse=correct, prompt_target_rms=2., cosine=.7)}))
    (main / "heldout_diagnostics.jsonl").write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (main / "metrics.jsonl").write_text(json.dumps(dict(step=1, loss_reader_mse=3., grad_norm=1.)) + '\n')
    (main / "warmup_gate.json").write_text(json.dumps(dict(heldout_error_mse=rows[-1]["heldout_error_mse"], checks={"readability": True}, ready_for_opd=True)))
    return main, heldout, rows


def test_clustered_mean_equal_prompt_weight_and_seed_reproducibility():
    values = {"a": [-1.] * 10, "b": [3.]}
    result = paired_prompt_summary(values, resamples=100)
    assert result == paired_prompt_summary(values, resamples=100)
    assert result["mean"] == 1. and result["num_prompts"] == 2
    assert result["prompt_win_fraction"] == .5 and result["num_states"] == 11
    assert paired_prompt_summary({"a": [-1., -2.]})["ci95"] is None
    with pytest.raises(ValueError):
        paired_prompt_summary({})


def test_report_keeps_zero_contradiction_separate_from_original_gate(tmp_path):
    main, heldout, _ = evidence(tmp_path)
    hashes = {str(path): sha256(path) for path in main.iterdir()}
    report = analyze_reader(main, heldout, resamples=100)
    assert report["original_ready_for_opd"] is True
    assert report["verdict"] == "reconstruction_improves_but_zero_control_is_better"
    values = report["checkpoints"][-1]["comparisons"]["all"]
    assert values["initial"]["mean"] == -4.
    assert values["shuffled"]["ci95"] == [-1., -1.]
    assert values["zero"]["ci95"] == [1., 1.]
    assert "not image semantic scores" in render_reader_report(report)
    assert hashes == {str(path): sha256(path) for path in main.iterdir()}
    assert step_bucket(19, 50) == "early" and step_bucket(20, 50) == "middle"
    assert step_bucket(38, 50) == "late"


@pytest.mark.parametrize("change", ["nan", "duplicate", "grid", "aggregate", "donor", "step"])
def test_bad_or_noncomparable_evidence_rejected(tmp_path, change):
    main, heldout, original = evidence(tmp_path)
    rows = copy.deepcopy(original)
    if change == "nan":
        rows[1]["per_state"][0]["errors"]["zero"] = float("nan")
    elif change == "duplicate":
        rows[1]["per_state"].append(rows[1]["per_state"][0])
    elif change == "grid":
        rows[1]["per_state"][0]["step_index"] = 2
    elif change == "aggregate":
        rows[1]["heldout_error_mse"]["correct"] = 99.
    elif change == "donor":
        rows[1]["per_state"][0]["donor_prompt_id"] = "changed"
    else:
        rows[1]["step"] = 0
    (main / "heldout_diagnostics.jsonl").write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        analyze_reader(main, heldout, resamples=20)


def test_cli_creates_fresh_report_and_refuses_overwrite(tmp_path):
    main, heldout, _ = evidence(tmp_path)
    output = tmp_path / "report"
    command = [sys.executable, "scripts/evaluate/analyze_reader_warmup.py", "--run-dir", str(main),
               "--heldout-data", str(heldout), "--output-dir", str(output), "--resamples", "20"]
    subprocess.run(command, check=True, capture_output=True)
    assert json.loads((output / "reader_diagnosis.json").read_text())["schema"] == "bagel-reader-offline-v1"
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_data_digest_and_training_gap_rejected(tmp_path):
    main, heldout, _ = evidence(tmp_path)
    heldout.write_text(heldout.read_text() + '\n')
    with pytest.raises(ValueError, match="SHA-256"):
        analyze_reader(main, heldout)
    manifest = dict(heldout_prompt_data_sha256=sha256(heldout))
    (main / "run_manifest.json").write_text(json.dumps(manifest))
    (main / "metrics.jsonl").write_text(json.dumps(dict(step=2, loss_reader_mse=3., grad_norm=1.)) + '\n')
    with pytest.raises(ValueError, match="contiguous"):
        analyze_reader(main, heldout)
