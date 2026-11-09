"""Merge several state-pair caches into one directory of symlinks.

Training takes a single cache directory, so a second production pool has to be
merged. Records are deduplicated by uid, shard manifests are renumbered, and
every state payload and parent image is symlinked at the relative path the
manifest already uses.
"""

import argparse
import json
import os
from pathlib import Path


def _load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caches", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    seen = set()
    shard_index = 0
    merged = []
    requests = []

    for cache_index, cache in enumerate(args.caches):
        root = Path(cache).resolve()
        manifests = sorted(root.glob("manifest.shard_*.json")) or sorted(root.glob("manifest.json"))
        if not manifests:
            raise ValueError("No manifest in " + str(root))
        for path in manifests:
            data = _load(path)
            records = []
            for record in data.get("records", []):
                # One uid can hold a keep record and an explicit-edit record; both
                # are useful, so deduplicate by (uid, task_type) instead.
                key_id = (str(record["uid"]), str(record.get("task_type") or record.get("role")))
                if key_id in seen:
                    continue
                seen.add(key_id)
                for key, item in record.get("states", {}).items():
                    original = item["path"]
                    relative = f"c{cache_index}/{original}"
                    target = out / relative
                    source = root / original
                    if not source.exists():
                        raise ValueError("Missing state payload " + str(source))
                    item["path"] = relative
                    if not args.dry_run:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.is_symlink() or target.exists():
                            target.unlink()
                        os.symlink(source, target)
                records.append(record)
            if not records:
                continue
            data["records"] = records
            merged.append((shard_index, data))
            shard_index += 1

        for image in sorted((root / "targets").glob("*_source_final.png")):
            # Request-only uids also need their parent image; link every file.
            link = out / "targets" / image.name
            if not args.dry_run:
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink() or link.exists():
                    link.unlink()
                os.symlink(image.resolve(), link)
        for request_path in sorted(root.glob("target_requests*.jsonl")):
            for line in request_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    requests.append(json.loads(line))
                except json.JSONDecodeError:
                    # Shards used to append to one shared file; skip torn lines.
                    continue

    unique_requests = {}
    for row in requests:
        unique_requests.setdefault(str(row["uid"]), row)

    print(json.dumps(dict(
        caches=len(args.caches), records=len(seen), shards=len(merged),
        requests=len(unique_requests), output=str(out),
    ), indent=2))
    if args.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    for index, data in merged:
        partial = out / f"manifest.shard_{index:03d}.partial"
        partial.write_text(json.dumps(data, ensure_ascii=False) + "\n")
        partial.replace(out / f"manifest.shard_{index:03d}.json")
    (out / "target_requests.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in unique_requests.values())
    )


if __name__ == "__main__":
    main()
