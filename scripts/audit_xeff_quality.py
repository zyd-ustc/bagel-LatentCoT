"""Audit picture quality across every review album in one command.

The review albums embed, per case, a baseline image plus one or more candidates:
source / candidate / effective in three columns, source / effective in two, or
source / preview / old checkpoint / new checkpoint in four.  Reading them by eye
does not scale and does not produce a number a gate can use, so this script
measures three things per column and compares each against the baseline of the
same case:

``hf``    mean |Laplacian| -- total high-frequency energy.
``grid``  energy share on the 16 px / 32 px lattices, i.e. the token grid.  A
          rough image and a tiled image need different fixes.
``hf_out`` the same as ``hf`` after masking the region that actually changed,
          so "roughened where edited" and "roughened everywhere" stop being the
          same complaint.

Usage::

    python scripts/audit_xeff_quality.py <review-dir>          # one album
    python scripts/audit_xeff_quality.py <parent-dir>          # every album below
    python scripts/audit_xeff_quality.py <review-dir> --json
    python scripts/audit_xeff_quality.py <review-dir> --gate   # nonzero on fail

A review directory holds one ``*.html`` album.  ``cases.json`` is read when
present and used only for the acceptance verdict.
"""

import argparse
import base64
import io
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

IMAGE = re.compile(r"data:image/(?:jpeg|png);base64,([A-Za-z0-9+/=]+)")
CAPTION = re.compile(r"<figcaption>([^<]*)</figcaption>")
SECTION = re.compile(r"<section class='case'>(.*?)</section>", re.S)


def decode(token):
    image = Image.open(io.BytesIO(base64.b64decode(token))).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def hf(picture):
    gray = picture.mean(2)
    lap = (
        4 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    return float(np.abs(lap).mean())


def grid_share(picture):
    """Energy fraction on the token lattice (1/16 and 1/32 cycles per pixel)."""
    gray = picture.mean(2)
    gray = gray - gray.mean()
    power = np.abs(np.fft.rfft2(gray)) ** 2
    rows, cols = gray.shape
    fy = np.fft.fftfreq(rows)[:, None]
    fx = np.fft.rfftfreq(cols)[None, :]
    band = (
        (np.abs(np.abs(fy) - 1 / 16) < 0.010)
        | (np.abs(np.abs(fx) - 1 / 16) < 0.010)
        | (np.abs(np.abs(fy) - 1 / 32) < 0.006)
        | (np.abs(np.abs(fx) - 1 / 32) < 0.006)
    )
    return float(power[band].sum() / power.sum())


def changed_mask(effective, source, quantile=0.99, grow=9):
    """Pixels where the edit landed, dilated to cover soft edges."""
    distance = np.abs(effective - source).mean(2)
    peak = distance >= np.quantile(distance, quantile)
    return ndimage.binary_dilation(peak, np.ones((grow, grow), bool))


def role(caption):
    text = caption.casefold()
    if "label" in text or "teacher" in text:
        return "label"
    if "candidate" in text or "x_edit" in text:
        return "candidate"
    if "effective" in text or "x_eff" in text:
        return "effective"
    if "preview" in text:
        return "preview"
    if "source" in text or "x_src" in text or "no-edit" in text:
        return "baseline"
    return "other"


def read_album(path):
    """Yield ``(caption list, decoded frames, verdict)`` for each case."""
    html = path.read_text(encoding="utf-8")
    cases = path.parent / "cases.json"
    verdicts = []
    if cases.exists():
        verdicts = [bool(row.get("accepted")) for row in json.loads(cases.read_text())]
    for index, body in enumerate(SECTION.findall(html)):
        captions = [c.strip() for c in CAPTION.findall(body)]
        frames = [decode(token) for token in IMAGE.findall(body)]
        if len(captions) != len(frames):
            captions = [""] * len(frames)
        verdict = verdicts[index] if index < len(verdicts) else None
        yield captions, frames, verdict


def measure(baseline, target):
    mask = changed_mask(target, baseline)
    keep = np.where(mask[..., None], baseline, target)
    return dict(
        hf=hf(target) / max(hf(baseline), 1e-9),
        grid=grid_share(target) / max(grid_share(baseline), 1e-9),
        hf_out=hf(keep) / max(hf(baseline), 1e-9),
        changed=float(mask.mean()),
        l1_vs_baseline=float(np.abs(target - baseline).mean()),
        l1_vs_previous=0.0,
    )


def audit(path):
    """Per-album column medians, keyed by column role and caption."""
    grouped = {}
    for captions, frames, verdict in read_album(path):
        roles = [role(caption) for caption in captions]
        if "baseline" not in roles:
            continue
        anchor = roles.index("baseline")
        for index, caption in enumerate(captions):
            if index == anchor:
                continue
            row = measure(frames[anchor], frames[index])
            if index > 0:
                row["l1_vs_previous"] = float(np.abs(frames[index] - frames[index - 1]).mean())
            key = (roles[index], caption or f"column {index}")
            grouped.setdefault(key, []).append(row)
    out = []
    for (kind, caption), rows in grouped.items():
        values = [row["hf"] for row in rows]
        out.append(dict(
            column=caption, kind=kind, cases=len(rows),
            hf=float(statistics.median(values)),
            grid=float(statistics.median([row["grid"] for row in rows])),
            hf_out=float(statistics.median([row["hf_out"] for row in rows])),
            changed=float(statistics.median([row["changed"] for row in rows])),
            l1_vs_baseline=float(statistics.median([row["l1_vs_baseline"] for row in rows])),
            rough=sum(1 for value in values if value > 1.5),
        ))
    return sorted(out, key=lambda row: -row["hf"])


def albums(root):
    """Every review album at or below ``root``."""
    return [path for path in sorted(root.rglob("*.html"))
            if SECTION.search(path.read_text(encoding="utf-8"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--gate", action="store_true",
                        help="nonzero exit when the effective column is rough; this is a texture "
                             "diagnostic, not a semantic acceptance test")
    parser.add_argument("--hf-max", type=float, default=1.5)
    parser.add_argument("--grid-max", type=float, default=2.0)
    args = parser.parse_args()

    report = {}
    for path in albums(args.directory):
        rows = audit(path)
        if rows:
            report[str(path.relative_to(args.directory))] = rows

    if args.json:
        print(json.dumps(report, indent=1))
        return 0

    print(f"{'album':38s} {'column':26s} {'n':>3s} {'hf':>6s} {'hf_out':>6s} {'grid':>6s} {'>1.5':>6s}")
    for name, rows in report.items():
        for row in rows:
            print(f"{name[:38]:38s} {row['column'][:26]:26s} {row['cases']:3d} "
                  f"{row['hf']:6.2f} {row['hf_out']:6.2f} {row['grid']:6.2f} "
                  f"{row['rough']:3d}/{row['cases']:<3d}")

    if args.gate:
        bad = []
        for name, rows in report.items():
            for row in rows:
                if row["kind"] in ("baseline", "preview", "label"):
                    continue
                if row["hf"] > args.hf_max or row["grid"] > args.grid_max:
                    bad.append(f"{name}:{row['column']} hf={row['hf']:.2f} grid={row['grid']:.2f}")
        if bad:
            print("gate failed: " + "; ".join(bad), file=sys.stderr)
            return 1
        print("gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
