"""Keep the qualified subset of a state-pair cache, ranked by measured error.

The producer already decides which cached states are usable (reconstruction and
convergence gates). This script must not overrule that decision: it only ranks
inside the qualified set and, when more states qualify than the training budget
needs, keeps the best ``--target-count`` by measured error. A record the
producer rejected is never promoted to accepted.

If fewer states qualify than requested, the run keeps what exists and reports the
shortfall. Padding the count by relaxing acceptance would redefine the label.
"""

import argparse
import json
from pathlib import Path

DEFAULT_CAPS = (0.08, 0.10, 0.12, 0.15, 0.20)


def _score(source, target, lpips_weight):
    source_l1 = float(source.get("l1", 1.0))
    target_l1 = float(target.get("l1", 1.0))
    lpips = float(target.get("lpips", 0.0))
    return source_l1 + target_l1 + lpips_weight * lpips


def _gate(item):
    """The producer's decision, recorded once so re-runs stay idempotent."""
    recorded = item.get("gate_accepted")
    return bool(item.get("accepted")) if recorded is None else bool(recorded)


def _passes_caps(item, caps):
    """Re-check explicit caps; ``caps`` may be empty to trust the producer."""
    if not caps:
        return True
    source = item.get("source_reconstruction", {})
    target = item.get("target_reconstruction", {})
    if "source_l1" in caps and float(source.get("l1", 9.0)) > caps["source_l1"]:
        return False
    if "target_l1" in caps and float(target.get("l1", 9.0)) > caps["target_l1"]:
        return False
    if "target_lpips" in caps and float(target.get("lpips", 9.0)) > caps["target_lpips"]:
        return False
    return True


def _select(entries, target_count, caps):
    """Rank *inside* the producer's qualified set; never promote a rejection."""
    qualified = [
        entry for entry in entries
        if _gate(entry["item"]) and _passes_caps(entry["item"], caps)
    ]
    qualified.sort(key=lambda entry: (entry["score"], entry["uid"], entry["key"]))
    return qualified, qualified[: max(0, int(target_count))]


def _cap_report(entries, caps_values):
    report = {}
    for value in caps_values:
        report[str(value)] = sum(
            1 for entry in entries
            if float(entry["item"].get("source_reconstruction", {}).get("l1", 9.0)) <= value
            and float(entry["item"].get("target_reconstruction", {}).get("l1", 9.0)) <= value
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--target-count", type=int, default=4500)
    parser.add_argument("--lpips-weight", type=float, default=0.5)
    parser.add_argument("--max-source-l1", type=float)
    parser.add_argument("--max-target-l1", type=float)
    parser.add_argument("--max-target-lpips", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.cache).resolve()
    manifests = sorted(root.glob("manifest.shard_*.json")) or sorted(root.glob("manifest.json"))
    if not manifests:
        raise ValueError("No shard manifest in " + str(root))

    entries, loaded = [], []
    for path in manifests:
        data = json.loads(path.read_text())
        loaded.append((path, data))
        for record in data.get("records", []):
            for key, item in record.get("states", {}).items():
                entries.append(dict(
                    score=_score(item.get("source_reconstruction", {}),
                                 item.get("target_reconstruction", {}), args.lpips_weight),
                    uid=record["uid"], key=key, item=item, record=record,
                ))

    caps = {}
    if args.max_source_l1 is not None:
        caps["source_l1"] = args.max_source_l1
    if args.max_target_l1 is not None:
        caps["target_l1"] = args.max_target_l1
    if args.max_target_lpips is not None:
        caps["target_lpips"] = args.max_target_lpips

    qualified, selected = _select(entries, args.target_count, caps)
    keep_ids = {(entry["uid"], entry["key"]) for entry in selected}
    cutoff = selected[-1]["score"] if selected else None
    summary = dict(
        shards=len(manifests),
        candidates=len(entries),
        producer_qualified=sum(1 for entry in entries if _gate(entry["item"])),
        qualified_after_caps=len(qualified),
        selected=len(selected),
        requested=args.target_count,
        shortfall=max(0, args.target_count - len(selected)),
        cutoff_score=cutoff,
        records_with_state=len({entry["uid"] for entry in selected}),
        caps=dict(caps),
        qualified_by_l1_cap=_cap_report(entries, DEFAULT_CAPS),
        rejection_reasons={},
    )
    for entry in entries:
        reason = entry["item"].get("reject_reason") or ("selected_out" if entry["item"].get("accepted") else "rejected")
        summary["rejection_reasons"][reason] = summary["rejection_reasons"].get(reason, 0) + 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.dry_run:
        return
    if len(selected) == 0:
        print("No state passed the producer gate; nothing was written.")
        return
    # Only ever narrow the accepted set: rejected states stay rejected.
    for path, data in loaded:
        changed = False
        for record in data.get("records", []):
            for key, item in record.get("states", {}).items():
                gate = _gate(item)
                if "gate_accepted" not in item:
                    item["gate_accepted"] = gate
                    changed = True
                accepted = bool(gate and (record["uid"], key) in keep_ids)
                if accepted != bool(item.get("accepted")):
                    item["accepted"] = accepted
                    changed = True
        if changed:
            partial = path.with_suffix(".partial")
            partial.write_text(json.dumps(data, ensure_ascii=False) + "\n")
            partial.replace(path)
    print(json.dumps(dict(rewritten=len(manifests), cache=str(root), selected=len(selected))))


if __name__ == "__main__":
    main()
