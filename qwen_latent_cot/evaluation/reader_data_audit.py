"""Read-only prompt audit and deterministic evaluation-pack preparation on CPU."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from statistics import median

from .geneval2 import load_benchmark
from .offline_reader import read_jsonl, sha256


CATEGORIES = ("count", "spatial_relation", "attribute_binding", "multi_object_composition",
              "action_relation", "rare_concept", "reasoning_heavy_t2i")
NUMBERS = set("zero one two three four five six seven eight nine ten".split())
COLORS = set("red blue green yellow white black orange pink purple brown gray grey".split())


def prompt_key(prompt):
    return " ".join(prompt.split()).casefold()


def words(prompt):
    return re.findall(r"\w+(?:'\w+)?", prompt.casefold())


def template_key(prompt):
    # Lexical risk flag only: changed counts/colors can change the task semantics.
    return " ".join("<number>" if word.isdigit() or word in NUMBERS else
                    "<color>" if word in COLORS else word for word in words(prompt))


def load_rows(path):
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError("empty prompt split")
    for row in rows:
        if (not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
                or not isinstance(row.get("prompt_id"), str) or not row["prompt_id"].strip()
                or row.get("category") not in CATEGORIES):
            raise ValueError("reader rows require string prompt/id and an allowed category")
    return rows


def split_summary(rows):
    lengths = sorted(len(words(row["prompt"])) for row in rows)
    templates = Counter(template_key(row["prompt"]) for row in rows)
    ids = Counter(row["prompt_id"] for row in rows)
    exact = Counter(row["prompt"] for row in rows)
    normalized = Counter(prompt_key(row["prompt"]) for row in rows)
    categories = Counter(row["category"] for row in rows)
    return dict(records=len(rows), category_counts=dict(categories),
                category_fraction={key: count / len(rows) for key, count in sorted(categories.items())},
                absent_categories=[category for category in CATEGORIES if category not in categories],
                duplicate_id_extra_rows=sum(count - 1 for count in ids.values()),
                duplicate_exact_prompt_extra_rows=sum(count - 1 for count in exact.values()),
                duplicate_normalized_prompt_extra_rows=sum(count - 1 for count in normalized.values()),
                word_lengths=dict(min=lengths[0], median=median(lengths),
                                  p95=lengths[int((len(lengths) - 1) * .95)], max=lengths[-1]),
                long_prompt_counts={"over_128_words": sum(value > 128 for value in lengths),
                                    "over_256_words": sum(value > 256 for value in lengths)},
                longest_prompt_examples=[dict(prompt_id=row["prompt_id"], category=row["category"],
                    word_count=len(words(row["prompt"])), preview=row["prompt"][:300])
                    for row in sorted(rows, key=lambda row: (-len(words(row["prompt"])), row["prompt_id"]))[:5]],
                category_label_sources=dict(Counter(row.get("category_label_source", "unspecified") for row in rows)),
                largest_lexical_templates=[dict(template=key, records=count) for key, count in templates.most_common(10)])


def lexical_neighbors(train, heldout, *, threshold=.85, limit=50):
    if not 0 < threshold <= 1:
        raise ValueError("Jaccard threshold must be in (0,1]")
    token_sets = [set(words(row["prompt"])) for row in train]
    inverted = defaultdict(set)
    for index, tokens in enumerate(token_sets):
        for token in tokens:
            inverted[token].add(index)
    matches = []
    for recipient in heldout:
        tokens = set(words(recipient["prompt"]))
        candidates = set().union(*(inverted[token] for token in tokens)) if tokens else set()
        for index in sorted(candidates):
            donor_tokens = token_sets[index]
            if not tokens or min(len(tokens), len(donor_tokens)) < threshold * max(len(tokens), len(donor_tokens)):
                continue
            score = len(tokens & donor_tokens) / len(tokens | donor_tokens)
            if score >= threshold:
                matches.append(dict(heldout_prompt_id=recipient["prompt_id"], train_prompt_id=train[index]["prompt_id"],
                                    jaccard=score, heldout_prompt=recipient["prompt"], train_prompt=train[index]["prompt"]))
    matches.sort(key=lambda row: (-row["jaccard"], row["heldout_prompt_id"], row["train_prompt_id"]))
    return dict(threshold=threshold, pair_count=len(matches),
                affected_heldout_prompts=len({row["heldout_prompt_id"] for row in matches}),
                examples=matches[:limit], example_limit=limit,
                interpretation="Unordered word-set similarity only; not proof of semantic leakage; no rows removed.")


def prepare_semantic_pack(benchmark_path, train, heldout, *, count=64):
    if isinstance(count, bool) or not isinstance(count, int) or count < 8 or count % 2:
        raise ValueError("semantic evaluation count must be even and >=8")
    benchmark = load_benchmark(benchmark_path)
    source = list(read_jsonl(benchmark_path))
    if any("vqa_list" not in row for row in source):
        raise ValueError("semantic scoring pack needs existing VQA annotations")
    excluded = {prompt_key(row["prompt"]) for row in train + heldout}
    candidates, skipped, seen = [], [], set()
    for index, row in enumerate(source):
        key = prompt_key(row["prompt"])
        if key in excluded or key in seen:
            skipped.append(index)
            continue
        seen.add(key)
        candidates.append((index, row))
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} nonoverlapping semantic prompts available; requested {count}")
    groups = defaultdict(list)
    for index, row in candidates:
        groups[row["atom_count"]].append((index, row))
    atomicities = sorted({row["atom_count"] for row in source})
    if count % len(atomicities):
        raise ValueError("semantic count must be divisible by source atomicity groups")
    per_group = count // len(atomicities)
    if any(len(groups[key]) < per_group for key in atomicities):
        raise ValueError("insufficient independent prompts in an atomicity group")
    chosen = sorted((pair for key in atomicities for pair in groups[key][:per_group]), key=lambda pair: pair[0])
    selected = []
    for index, row in chosen:
        selected.append({**row, "prompt_id": f"geneval2-heldout:{index:04d}",
                         "category": "multi_object_composition", "split": "heldout",
                         "category_label_source": "existing_GenEval2_compositional_benchmark",
                         "source_benchmark_sha256": benchmark.source_sha256,
                         "source_benchmark_index": index})
    provenance = dict(schema="bagel-phase1a-semantic-pack-v1", source=str(benchmark_path),
        source_sha256=benchmark.source_sha256, source_records=benchmark.prompt_count,
        eligible_records=len(candidates), selected_records=count, skipped_source_indices=skipped,
        selected_source_indices=[row["source_benchmark_index"] for row in selected],
        selection="equal quota per source atomicity; first eligible within group; output source order; no score-based selection",
        prompts_per_atomicity=per_group,
        exclusion="casefold/whitespace normalized exact prompt overlap with reader train and heldout",
        primary_category="multi_object_composition; per-question skills retained unchanged",
        atom_question_skill_counts=dict(Counter(skill for row in selected for skill in row["skills"])),
        atomicity_counts=dict(Counter(str(row["atom_count"]) for row in selected)),
        teacher_reasoning_generated=False, images_generated=False, semantic_scores_available=False)
    return selected, provenance


def audit_reader_data(train_path, heldout_path, *, export_report_path=None,
                      evaluated_count=8, benchmark_path=None, semantic_count=64):
    train, heldout = load_rows(train_path), load_rows(heldout_path)
    if not 2 <= evaluated_count <= len(heldout):
        raise ValueError("evaluated_count must be between two and heldout size")
    train_keys = {prompt_key(row["prompt"]) for row in train}
    heldout_keys = {prompt_key(row["prompt"]) for row in heldout}
    templates = {template_key(row["prompt"]) for row in train}
    report = dict(schema="bagel-reader-data-audit-v1", train=split_summary(train), heldout=split_summary(heldout),
        evaluated_prefix=split_summary(heldout[:evaluated_count]),
        evaluated_prompt_ids=[row["prompt_id"] for row in heldout[:evaluated_count]],
        cross_split=dict(exact_prompt_overlap=len({row["prompt"] for row in train} & {row["prompt"] for row in heldout}),
                         normalized_prompt_overlap=len(train_keys & heldout_keys),
                         prompt_id_overlap=len({row["prompt_id"] for row in train} & {row["prompt_id"] for row in heldout}),
                         count_color_template_heldout_hits=sum(template_key(row["prompt"]) in templates for row in heldout)),
        provenance={str(train_path): sha256(train_path), str(heldout_path): sha256(heldout_path)},
        lexical_neighbors=lexical_neighbors(train, heldout),
        action="audit only; no original split rewritten; no near-duplicate automatically removed",
        limitations=["Labels are existing heuristic/export metadata, not human semantic annotations.",
                     "Word counts are not BAGEL tokenizer lengths.",
                     "Lexical templates and unordered Jaccard are risk flags, not semantic equivalence.",
                     "A prefix of heldout is not a stratified or full-heldout evaluation."])
    if export_report_path:
        export = json.loads(Path(export_report_path).read_text())
        if export.get("train_sha256") != sha256(train_path) or export.get("heldout_sha256") != sha256(heldout_path):
            raise ValueError("export-report split hash mismatch")
        if export.get("train_records") != len(train) or export.get("heldout_records") != len(heldout):
            raise ValueError("export-report split size mismatch")
        report["original_export_report"] = export
        report["provenance"][str(export_report_path)] = sha256(export_report_path)
    report["safe_exact_split"] = (not any(report["cross_split"][key] for key in
        ("exact_prompt_overlap", "normalized_prompt_overlap", "prompt_id_overlap"))
        and all(not report[split][key] for split in ("train", "heldout") for key in
                ("duplicate_id_extra_rows", "duplicate_normalized_prompt_extra_rows")))
    selected = []
    if benchmark_path:
        selected, metadata = prepare_semantic_pack(benchmark_path, train, heldout, count=semantic_count)
        report["semantic_pack"] = metadata
        report["provenance"][str(benchmark_path)] = sha256(benchmark_path)
    return report, selected


def render_data_audit(report):
    lines = ["# Reader data audit", "", "Original data unchanged. CPU-only lexical/provenance audit.", "",
             "| Split | Rows | Count | Spatial | Median words | Exact duplicate extra rows |",
             "|---|---:|---:|---:|---:|---:|"]
    for key in ("train", "heldout", "evaluated_prefix"):
        row = report[key]
        lines.append(f'| {key} | {row["records"]} | {row["category_counts"].get("count", 0)} | {row["category_counts"].get("spatial_relation", 0)} | {row["word_lengths"]["median"]} | {row["duplicate_exact_prompt_extra_rows"]} |')
    lines += ["", f'Exact split integrity: `{report["safe_exact_split"]}`.',
              f'Cross-split checks: `{json.dumps(report["cross_split"], sort_keys=True)}`.',
              f'Lexical Jaccard >=.85: {report["lexical_neighbors"]["pair_count"]} candidate pairs affecting {report["lexical_neighbors"]["affected_heldout_prompts"]} heldout prompts; not proof of semantic leakage.',
              "", "## Coverage", "",
              "Existing train/heldout categories and lexical labels are retained. Missing categories:",
              ", ".join(report["train"]["absent_categories"]), ""]
    if "semantic_pack" in report:
        pack = report["semantic_pack"]
        lines += ["## Prepared image-scoring prompts", "",
                  f'{pack["selected_records"]} prompts from {pack["source_records"]} existing GenEval2 heldout rows.',
                  pack["selection"] + ".", pack["exclusion"] + ".",
                  "Original VQA questions/answers and skills retained. No new teacher CoT or image scores.", ""]
    lines += ["## Limitations", ""] + [f"- {value}" for value in report["limitations"]]
    return "\n".join(lines) + "\n"
