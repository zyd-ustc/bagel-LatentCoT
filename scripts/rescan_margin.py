"""Score a finished inverted cache by the paired-margin criterion.

For an off-model corpus the absolute round trip cannot be small, so the usable
question is whether the continued target state belongs to the target rather than
to the source. This script recomputes that margin from the saved reconstruction
images (no accelerator needed) and reports how many records clear a range of
thresholds; with ``--apply`` it also rewrites ``accepted`` and records
``gate_accepted`` so a later ``select_accepted`` run cannot widen the gate.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from qwen_latent_cot.u15_target_correction.images import load_pair


def _read(path):
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)[None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--apply", type=float, default=None,
                        help="Margin fraction to write into accepted/gate_accepted")
    parser.add_argument("--canvas", type=int, default=512)
    args = parser.parse_args()
    root = Path(args.cache).resolve()
    manifests = sorted(root.glob("manifest.shard_*.json"))
    if not manifests:
        raise ValueError("No shard manifest in " + str(root))
    thresholds = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
    tally = {value: 0 for value in thresholds}
    rows = []
    written = 0
    for position, path in enumerate(manifests):
        if position % args.shard_count != args.shard_index:
            continue
        data = json.loads(path.read_text())
        for record in data.get("records", []):
            for key, item in record.get("states", {}).items():
                relative = item.get("target_reconstruction_image")
                if not relative:
                    continue
                recon = _read(root / relative)
                source, target, valid, _ = load_pair(record, args.canvas, args.canvas, "cpu")
                mask = valid[0, 0].numpy().astype(bool)
                against_source = float(np.abs(recon[0].numpy() - source[0].numpy())[:, mask].mean())
                against_target = float(np.abs(recon[0].numpy() - target[0].numpy())[:, mask].mean())
                pair = float(np.abs(target[0].numpy() - source[0].numpy())[:, mask].mean())
                ratio = (against_source - against_target) / max(pair, 1e-6)
                item["margin"] = dict(
                    against_source_l1=against_source,
                    against_target_l1=against_target,
                    pair_l1=pair,
                    ratio=ratio,
                )
                row = dict(uid=record["uid"], key=key, ratio=ratio, pair_l1=pair,
                           against_source_l1=against_source, against_target_l1=against_target)
                rows.append(row)
                for value in thresholds:
                    tally[value] += int(ratio >= value)
        if args.apply is not None:
            changed = False
            for record in data.get("records", []):
                for key, item in record.get("states", {}).items():
                    margin = item.get("margin")
                    if margin is None:
                        continue
                    # Non-finite states and failed target semantics are not label
                    # decisions the margin can overturn.
                    fatal = item.get("reject_reason") in ("nonfinite_state", "target_semantics")
                    gate = bool(margin["ratio"] >= args.apply) and not fatal
                    item["gate_accepted"] = gate
                    if bool(item.get("accepted")) != gate:
                        item["accepted"] = gate
                        changed = True
            if changed:
                partial = path.with_suffix(".partial")
                partial.write_text(json.dumps(data, ensure_ascii=False) + "\n")
                partial.replace(path)
                written += 1
    out = root / f"margin.shard_{args.shard_index:03d}.jsonl"
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(json.dumps(dict(shard=args.shard_index, scored=len(rows),
                          thresholds={str(k): v for k, v in tally.items()},
                          rewritten=written, applied=args.apply)))


if __name__ == "__main__":
    main()
