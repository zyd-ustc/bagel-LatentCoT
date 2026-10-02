"""Audit existing reader splits and prepare an independent semantic evaluation list."""

import argparse
import json
from pathlib import Path

from qwen_latent_cot.evaluation.offline_reader import sha256
from qwen_latent_cot.evaluation.reader_data_audit import audit_reader_data, render_data_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--heldout-data", type=Path, required=True)
    parser.add_argument("--export-report", type=Path)
    parser.add_argument("--semantic-benchmark", type=Path)
    parser.add_argument("--evaluated-count", type=int, default=8)
    parser.add_argument("--semantic-count", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    report, rows = audit_reader_data(args.train_data, args.heldout_data,
        export_report_path=args.export_report, evaluated_count=args.evaluated_count,
        benchmark_path=args.semantic_benchmark, semantic_count=args.semantic_count)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if rows:
        manifest = args.output_dir / "phase1a_semantic_hard64.jsonl"
        # Name includes actual count when explicitly overridden.
        if len(rows) != 64:
            manifest = args.output_dir / f"phase1a_semantic_hard{len(rows)}.jsonl"
        manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        report["semantic_pack"]["output_manifest_sha256"] = sha256(manifest)
        (args.output_dir / "semantic_pack_provenance.json").write_text(json.dumps(report["semantic_pack"], indent=2) + '\n')
    (args.output_dir / "data_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (args.output_dir / "data_audit.md").write_text(render_data_audit(report))
    print(json.dumps(dict(output=str(args.output_dir), safe_exact_split=report["safe_exact_split"],
                          lexical_pairs=report["lexical_neighbors"]["pair_count"])))


if __name__ == "__main__":
    main()
