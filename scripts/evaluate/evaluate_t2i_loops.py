#!/usr/bin/env python3
"""Score generated Base/LegacyMem/GEN/GEN+M arms and verify paired outcomes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json

import yaml

from qwen_latent_cot.bagel.accelerator import resolve_device
from qwen_latent_cot.evaluation.loop_results import (
    LocalVLMJudge,
    OfficialGenEval2,
    file_sha256,
    merge_manifests,
    score_manifest,
    summarize_human_pairwise,
    summarize_results,
    validate_required_arms,
    write_reports,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(cfg["output_dir"]).resolve()
    if not cfg.get("judge_model_path") and (
        cfg.get("require_quality", True)
        or any(
            settings["kind"] == "tiif_spatial" for settings in cfg["datasets"].values()
        )
    ):
        raise ValueError("quality/TIIF scoring requires local judge_model_path")
    rows, jobs = [], []
    provenance = {"configuration_sha256": file_sha256(args.config), "datasets": {}}
    for name, settings in cfg["datasets"].items():
        specifications = settings.get("manifests", [settings.get("manifest")])
        manifest = merge_manifests(specifications)
        validate_required_arms(manifest, cfg.get("required_depths", [0, 1, 2, 3, 4]))
        benchmark = [
            json.loads(line)
            for line in Path(settings["benchmark"]).read_text().splitlines()
            if line.strip()
        ]
        kind = settings["kind"]
        scorer = None
        if kind == "geneval2_hard":
            scorer = OfficialGenEval2(
                cfg["geneval2_root"],
                out / "official_scores" / name,
                python=cfg.get("geneval2_python", sys.executable),
            )
        elif kind not in {"tiif_spatial", "quality"}:
            raise ValueError(
                "dataset kind must be geneval2_hard, tiif_spatial, or quality"
            )
        print(f"Scoring {name}: {len(manifest['images'])} generated images", flush=True)
        scored = score_manifest(
            manifest, benchmark, dataset=name, semantic_scorer=scorer
        )
        paths = [
            item if isinstance(item, str) else item["path"] for item in specifications
        ]
        provenance["datasets"][name] = {
            "kind": kind,
            "manifests": [
                {"path": str(path), "sha256": file_sha256(path)} for path in paths
            ],
            "benchmark_sha256": file_sha256(settings["benchmark"]),
            "semantic_scorer": scorer.provenance
            if scorer
            else "tiif_local_yes_no"
            if kind == "tiif_spatial"
            else None,
        }
        jobs.append((name, kind, manifest, benchmark, scored))
    # Official subprocesses finish before a second judge occupies accelerator
    # memory. All TIIF and quality scoring then share one local VLM instance.
    judge = (
        LocalVLMJudge(
            cfg["judge_model_path"],
            device=str(resolve_device(cfg.get("judge_device", "auto"))),
        )
        if cfg.get("judge_model_path")
        else None
    )
    provenance["judge"] = judge.provenance if judge else None
    for name, kind, manifest, benchmark, scored in jobs:
        if kind == "tiif_spatial":
            scored = score_manifest(
                manifest, benchmark, dataset=name, semantic_scorer=judge.tiif
            )
        for row in scored:
            if judge:
                quality = (
                    judge.quality(row["path"])
                    if row["valid_file"]
                    else {"quality_proxy": 0.0, "invalid": True}
                )
                if (
                    not 0 <= quality["quality_proxy"] <= 1
                    or type(quality["invalid"]) is not bool
                ):
                    raise ValueError("invalid quality output")
                row["quality_proxy"], row["judge_invalid"] = (
                    quality["quality_proxy"],
                    quality["invalid"],
                )
        rows.extend(scored)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}_scores.jsonl").write_text(
            "\n".join(json.dumps(row) for row in scored) + "\n"
        )
    summaries = summarize_results(
        rows, threshold=float(cfg.get("semantic_pass_threshold", 0.5))
    )
    if cfg.get("human_pairwise_csv"):
        human = summarize_human_pairwise(cfg["human_pairwise_csv"], rows)
        provenance["human_pairwise_sha256"] = file_sha256(cfg["human_pairwise_csv"])
        for summary in summaries:
            summary["human_pairwise"] = human.get((summary["dataset"], summary["arm"]))
    write_reports(out, rows, summaries, provenance)


if __name__ == "__main__":
    main()
