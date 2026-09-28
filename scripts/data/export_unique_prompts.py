#!/usr/bin/env python3
"""Export first-occurrence, exact-string unique prompts without editing the source."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_unique_prompts(source, output_dir, *, heldout=None):
    source = Path(source).resolve(strict=True)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing export: {output}")
    before = source.stat()
    source_digest = hashlib.sha256()
    unique, selected_ids = {}, set()
    source_counts, retained_counts = Counter(), Counter()
    total, blank_lines = 0, 0
    with source.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            source_digest.update(raw)
            if not raw.strip():
                blank_lines += 1
                continue
            row = json.loads(raw)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"line {line_number}: prompt must be a nonempty string")
            identifier = row.get("sample_id", row.get("id"))
            if identifier is None or not str(identifier).strip():
                raise ValueError(f"line {line_number}: missing sample_id/id")
            total += 1
            origin = str(row.get("source", "unknown"))
            source_counts[origin] += 1
            if prompt in unique:
                unique[prompt]["duplicate_count"] += 1
                continue
            identifier = str(identifier)
            if identifier in selected_ids:
                raise ValueError(f"line {line_number}: repeated id for distinct prompts: {identifier}")
            selected_ids.add(identifier)
            retained_counts[origin] += 1
            unique[prompt] = dict(id=identifier, prompt=prompt, source=origin,
                                  source_line=line_number, duplicate_count=0)
    if not unique:
        raise ValueError("source contains no prompt records")
    source_hash = source_digest.hexdigest()
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("source changed during scan; nothing exported")
    if sha256_file(source) != source_hash:
        raise RuntimeError("source changed during scan; nothing exported")
    overlap = None
    heldout_hash = None
    if heldout is not None:
        heldout = Path(heldout).resolve(strict=True)
        raw = heldout.read_bytes()
        heldout_hash = hashlib.sha256(raw).hexdigest()
        heldout_prompts = {json.loads(line)["prompt"] for line in raw.splitlines() if line.strip()}
        overlap = len(set(unique) & heldout_prompts)
        if overlap:
            raise ValueError(f"{overlap} exact held-out prompt overlaps; no automatic filtering or export")
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "prompts.jsonl"
    with manifest.open("x", encoding="utf-8") as handle:
        for row in unique.values():
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary = dict(schema="bagel-unique-prompts-v1", source_path=str(source),
        source_sha256=source_hash, source_bytes=before.st_size, source_unchanged=True,
        output_path=str(manifest), output_sha256=sha256_file(manifest),
        input_records=total, unique_prompts=len(unique), skipped_duplicate_rows=total-len(unique),
        duplicate_prompt_groups=sum(row["duplicate_count"] > 0 for row in unique.values()),
        blank_lines=blank_lines, deduplication="exact prompt string; first occurrence in source order",
        id_policy="original sample_id (fallback id); no generated ids",
        source_counts=dict(source_counts), retained_source_counts=dict(retained_counts),
        heldout_path=str(heldout) if heldout is not None else None,
        heldout_sha256=heldout_hash, heldout_exact_prompt_overlap=overlap,
        scope="prompt-only native-teacher grounding; not edit pairs or semantic GRPO metadata")
    # Summary is written last and serves as the completion record.
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--heldout", help="fail if any exact prompt overlaps; never silently filter")
    args = parser.parse_args()
    print(json.dumps(export_unique_prompts(args.source, args.output_dir, heldout=args.heldout),
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
