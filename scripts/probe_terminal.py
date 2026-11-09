"""Measure a terminal-supervised editor on a fixed probe set.

Training loss is averaged over whatever samples the sampler drew, and every
sample has a different edit region, so a running mean does not say whether the
model improves. This script answers one question on a frozen list of records:
after loading a checkpoint, what do the structural term, the preservation term
and the high-frequency ratio look like on the same pictures every time.

Usage::

    python scripts/probe_terminal.py --model ... --source ... --config ... \
        --cache <coupled cache> --checkpoint <run>/step_000120 --limit 8 \
        --output <run>/probe_120.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.checkpoint import editor_identity, load_stage_weights
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.data import StatePairManifest
from qwen_latent_cot.u15_target_correction.field import (
    NativeT2I, edit_state, euler, phase_grid, segment_times, time_key,
)
from qwen_latent_cot.u15_target_correction.loading import load_native
from qwen_latent_cot.u15_target_correction.model import StateEditor
from qwen_latent_cot.u15_target_correction.terminal import terminal_loss
from qwen_latent_cot.u15_target_correction import NATIVE_PARENT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    device = set_device(args.device)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, distributed=False)
    manifest = StatePairManifest(args.cache, rt, cfg, NATIVE_PARENT)
    editor = StateEditor(rt, cfg).eval()
    if args.checkpoint:
        load_stage_weights(args.checkpoint, editor, editor_identity(rt, cfg))

    grid = phase_grid(rt, cfg)
    time = float(cfg["read_time"])
    key = time_key(time)
    wanted = [
        index for index in manifest.train_indices
        if manifest.rows[index]["states"].get(key, {}).get("accepted")
    ][: int(args.limit)]
    if len(wanted) < 1:
        raise SystemExit("no accepted records at the read time")

    options = {k: v for k, v in cfg["terminal"].items() if k != "enabled"}
    rows = []
    for index in wanted:
        sample = manifest.sample(index, time)
        with torch.no_grad():
            prefixes = editor.prefixes(
                sample["preview"], sample["edit_request"],
                sample["tail_condition"], sample["preserve"],
            )
            base = editor.image_clean(sample["state_a"], time, prefixes["image"])
            edited, _ = edit_state(
                editor, sample["state_a"], time, prefixes, int(cfg["edit_steps"]), base
            )
            velocity = NativeT2I(rt, sample["tail_condition"], cfg)
            picture = euler(edited.float(), segment_times(grid, time, 1.0), velocity)
            loss, detail = terminal_loss(
                picture, sample["x_teacher"], sample["x_source"], sample["valid"], **options
            )
            ceiling, ceiling_detail = terminal_loss(
                sample["x_teacher"], sample["x_teacher"], sample["x_source"],
                sample["valid"], **options,
            )
        rows.append(dict(
            uid=sample["uid"],
            loss=float(loss),
            structure=detail["structure"],
            preserve=detail["preserve"],
            frequency=detail["frequency"],
            hf_ratio=detail["hf_ratio"],
            ceiling_structure=ceiling_detail["structure"],
            ceiling_hf=ceiling_detail["hf_ratio"],
            label_loss=float(ceiling),
        ))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    def mean(key_name):
        return sum(row[key_name] for row in rows) / len(rows)

    summary = dict(
        cases=len(rows),
        checkpoint=str(args.checkpoint) if args.checkpoint else "initialisation",
        loss=mean("loss"),
        structure=mean("structure"),
        preserve=mean("preserve"),
        frequency=mean("frequency"),
        hf_ratio=mean("hf_ratio"),
        ceiling_structure=mean("ceiling_structure"),
        ceiling_hf=mean("ceiling_hf"),
    )
    Path(args.output).write_text(json.dumps(dict(summary=summary, rows=rows), indent=1) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
