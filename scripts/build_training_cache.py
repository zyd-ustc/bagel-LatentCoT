"""Build one training cache from several label caches, with a role allow-list.

The trainer reads a single state-pair cache and samples roles from it, so the
R-edit records (built by the coupled route) and the R-keep records (built by the
native pass) have to live in one manifest. This script copies nothing: it rewrites
every payload and image path to an absolute path, then drops records whose role is
not allowed or whose only state was rejected.
"""

import argparse
import json
from pathlib import Path

PATH_FIELDS = ("source_image", "target_image")
STATE_FIELDS = ("source_reconstruction_image", "target_reconstruction_image")


def _manifests(cache):
    root = Path(cache).resolve()
    if root.is_dir():
        found = sorted(root.glob("manifest.json")) or sorted(root.glob("manifest.shard_*.json"))
    else:
        root, found = root.parent, [root]
    if not found:
        raise ValueError("No manifest in " + str(cache))
    return root, found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caches", nargs="+", required=True)
    parser.add_argument("--roles", nargs="+", required=True,
                        help="Default allow-list for caches without an explicit one")
    parser.add_argument("--allow", action="append", default=[],
                        help="Per-cache allow-list, e.g. --allow /path/to/cache=R_keep")
    parser.add_argument("--output", required=True)
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args()

    allowed = set(args.roles)
    per_cache = {}
    for entry in args.allow:
        path, _, roles = entry.partition("=")
        per_cache[str(Path(path).resolve())] = {r for r in roles.split(",") if r}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    header, records, seen = None, [], set()
    for cache in args.caches:
        root, manifests = _manifests(cache)
        for manifest_path in manifests:
            data = json.loads(manifest_path.read_text())
            header = header or {k: v for k, v in data.items() if k != "records"}
            allow_here = per_cache.get(str(root), allowed)
            for record in data.get("records", []):
                if record.get("role") not in allow_here:
                    continue
                uid = str(record["uid"])
                if uid in seen:
                    continue
                states = {
                    key: item for key, item in (record.get("states") or {}).items()
                    if item.get("accepted")
                }
                if not states:
                    continue
                for item in states.values():
                    item["path"] = str((root / item["path"]).resolve())
                    for field in STATE_FIELDS:
                        if item.get(field):
                            item[field] = str((root / item[field]).resolve())
                for field in PATH_FIELDS:
                    if record.get(field):
                        record[field] = str((root / record[field]).resolve())
                record["states"] = states
                seen.add(uid)
                records.append(record)
    if not records:
        raise ValueError("Nothing survived the role filter")
    per = (len(records) + args.shards - 1) // args.shards
    for index in range(args.shards):
        chunk = records[index * per:(index + 1) * per]
        if not chunk:
            continue
        payload = dict(header, records=chunk)
        (output / f"manifest.shard_{index:03d}.json").write_text(
            json.dumps(payload, ensure_ascii=False) + "\n"
        )
    counts = {}
    for record in records:
        counts[record["role"]] = counts.get(record["role"], 0) + 1
    print(json.dumps(dict(records=len(records), roles=counts, shards=args.shards)))


if __name__ == "__main__":
    main()
