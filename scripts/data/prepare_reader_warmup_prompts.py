"""Read-only CORT manifests -> deduplicated, conservatively tagged T2I prompts.

Heuristic tags are audit metadata, not human labels. No images/CoT are consumed.
Official validation rows remain held out; their prompts are excluded from train.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


SPATIAL = re.compile(r"\b(?:to the left of|to the right of|left of|right of|in front of|"
                     r"on top of|underneath|above|below|behind)\b", re.I)
OBJECTS = ("cubes|spheres|cats|dogs|people|men|women|children|cups|chairs|apples|bananas|"
           "oranges|balls|birds|trees|books|bottles|cars|houses|flowers|statues|robots|"
           "clocks|elephants|turtles|zebras|monkeys|vases|plates|shoes|umbrellas|tables|"
           "bicycles|triangles|squares|circles|horses|cows|ducks|rabbits|bears|boxes")
COUNT = re.compile(r"\b(?:two|three|four|five|six|seven|eight|nine|ten|[2-9]|10)\s+"
                   r"(?:[a-z]+[ -]){0,3}(?:"+OBJECTS+r")\b", re.I)


def prompt_key(prompt):
    return " ".join(prompt.split()).casefold()


def label_prompt(prompt):
    for category, pattern in (("spatial_relation", SPATIAL), ("count", COUNT)):
        match = pattern.search(prompt)
        if match:
            return category, match.group(0)
    return None


def read_manifest(path):
    records, seen, summary = [], set(), Counter()
    with Path(path).open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            summary["source_rows"] += 1
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_number}: missing T2I prompt")
            key = prompt_key(prompt)
            if key in seen:
                summary["duplicates"] += 1
                continue
            seen.add(key)
            label = label_prompt(prompt)
            if label is None:
                summary["no_explicit_count_or_relation"] += 1
                continue
            category, evidence = label
            identifier = str(row.get("sample_id") or row.get("prompt_id") or line_number)
            records.append(dict(prompt_id=identifier,prompt=prompt,category=category,
                category_label_source="explicit_regex_v1_not_human_reviewed",
                category_evidence=evidence,source_manifest=str(Path(path).resolve()),
                source_line=line_number,source=row.get("source")))
    return records, seen, dict(summary)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def prepare(source, heldout_source, output_dir, *, heldout_count=64, seed=42):
    if Path(source).resolve() == Path(heldout_source).resolve():
        raise ValueError("heldout must come from a separate official source split")
    if heldout_count < 2:
        raise ValueError("heldout_count must be >=2")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    train, train_keys, train_stats = read_manifest(source)
    heldout, heldout_keys, heldout_stats = read_manifest(heldout_source)
    # Preserve the official val split: remove overlaps from training, not val.
    overlap = train_keys & heldout_keys
    train = [row for row in train if prompt_key(row["prompt"]) not in heldout_keys]
    ordering = lambda row: hashlib.sha256(f'{seed}:{row["prompt"]}'.encode()).hexdigest()
    train.sort(key=ordering)
    heldout.sort(key=ordering)
    if len(heldout) < heldout_count or len(train) < 8:
        raise ValueError("not enough explicitly tagged independent prompts")
    heldout = heldout[:heldout_count]
    for split, rows in (("train",train),("heldout",heldout)):
        ids = set()
        for row in rows:
            row["prompt_id"] = split+":"+row["prompt_id"]
            row["split"] = split
            if row["prompt_id"] in ids:
                raise ValueError("duplicate source id for distinct prompts")
            ids.add(row["prompt_id"])
    output.mkdir(parents=True,exist_ok=False)
    for name, rows in (("reader_train.jsonl",train),("reader_heldout.jsonl",heldout)):
        with (output/name).open("w") as target:
            for row in rows:
                target.write(json.dumps(row,ensure_ascii=False)+"\n")
    report=dict(schema="bagel-reader-prompt-export-v1",seed=seed,
        source=str(Path(source).resolve()),source_sha256=digest(source),
        heldout_source=str(Path(heldout_source).resolve()),heldout_source_sha256=digest(heldout_source),
        train_records=len(train),heldout_records=len(heldout),
        normalized_source_prompt_overlaps=len(overlap),
        overlap_policy="exclude every official heldout-source prompt from training",
        deduplication="casefold and whitespace normalization; first occurrence",
        selection="explicit_regex_v1 count/spatial only; other prompts skipped",
        category_label_quality="heuristic, not human-reviewed; no semantic success claim",
        train_category_counts=dict(Counter(row["category"] for row in train)),
        heldout_category_counts=dict(Counter(row["category"] for row in heldout)),
        train_source_stats=train_stats,heldout_source_stats=heldout_stats,
        train_sha256=digest(output/"reader_train.jsonl"),
        heldout_sha256=digest(output/"reader_heldout.jsonl"),images_consumed=False)
    (output/"export_report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    parser.add_argument("--heldout-source",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--heldout-count",type=int,default=64)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    print(json.dumps(prepare(args.source,args.heldout_source,args.output_dir,
        heldout_count=args.heldout_count,seed=args.seed),indent=2))
