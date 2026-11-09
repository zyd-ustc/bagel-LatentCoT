"""Render a three-column review album for a terminal-supervised checkpoint.

Columns are the parent's native completion, the label image the loss was
trained against, and what the editor's state actually produces. The album keeps
the same embedded-JPEG layout the audit script reads, so
``scripts/audit_xeff_quality.py --gate`` can score it without a human eye.
"""

import argparse
import base64
import io
import json
from pathlib import Path

import torch
from PIL import Image

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.checkpoint import editor_identity, load_stage_weights
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.data import StatePairManifest
from qwen_latent_cot.u15_target_correction.field import (
    NativeT2I, edit_state, euler, phase_grid, segment_times, time_key,
)
from qwen_latent_cot.u15_target_correction.images import save_image
from qwen_latent_cot.u15_target_correction.loading import load_native
from qwen_latent_cot.u15_target_correction.model import StateEditor
from qwen_latent_cot.u15_target_correction.terminal import high_frequency, terminal_loss
from qwen_latent_cot.u15_target_correction import NATIVE_PARENT

COLUMNS = ("x_src: parent native completion",
           "label x_edit: teacher image",
           "x_eff: editor output")


def embed(tensor, quality=92):
    array = ((tensor.detach().float().cpu()[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5)
    image = Image.fromarray(array.round().numpy().astype("uint8"))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
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
    options = {k: v for k, v in cfg["terminal"].items() if k != "enabled"}

    out = Path(args.output)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    cases = []
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
            _, detail = terminal_loss(
                picture, sample["x_teacher"], sample["x_source"], sample["valid"], **options
            )
        name = sample["uid"].replace("/", "_")
        for label, tensor in (("src", sample["x_source"]), ("teacher", sample["x_teacher"]), ("hat", picture)):
            save_image(out / "frames" / f"{name}_{label}.png", tensor)
        cases.append(dict(
            uid=sample["uid"],
            edit=sample["edit_request"],
            hf_hat=float(high_frequency(picture)),
            hf_src=float(high_frequency(sample["x_source"])),
            hf_ratio=detail["hf_ratio"],
            structure=detail["structure"],
            preserve=detail["preserve"],
            images=[embed(sample["x_source"]), embed(sample["x_teacher"]), embed(picture)],
        ))
        print(cases[-1]["uid"], round(detail["hf_ratio"], 3), flush=True)

    rows = []
    for case in cases:
        figures = "".join(
            f"<figure><img src='data:image/jpeg;base64,{token}'><figcaption>{caption}</figcaption></figure>"
            for token, caption in zip(case["images"], COLUMNS)
        )
        rows.append(
            f"<section class='case'><div class='head'><b>{case['uid']}</b>"
            f"<span class='m'>hf {case['hf_ratio']:.2f} · struct {case['structure']:.3f}"
            f" · keep {case['preserve']:.3f}</span></div>"
            f"<div class='e'>edit: {case['edit']}</div><div class='s'>{figures}</div></section>"
        )
    mean_ratio = sum(case["hf_ratio"] for case in cases) / max(1, len(cases))
    html = (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<title>terminal review</title><style>body{margin:0;background:#0d1117;color:#e6edf3;"
        "font:13px/1.5 -apple-system,sans-serif}header{padding:14px 18px;border-bottom:1px solid #30363d}"
        ".m{color:#8b949e;margin-left:8px}.e{color:#d29922;margin:4px 0 8px}"
        ".s{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}figure{margin:0}"
        "img{width:100%;aspect-ratio:1;object-fit:cover;border:1px solid #30363d;border-radius:6px;display:block}"
        "figcaption{text-align:center;color:#8b949e;font-size:11px;margin-top:3px}"
        ".case{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:10px;margin-bottom:12px}"
        "main{padding:14px}</style></head><body><header><h1>终端监督三列对照</h1>"
        f"<div class='m'>共 {len(cases)} 例 · 平均高频比 {mean_ratio:.2f}（纹理指标，不是验收）· "
        f"checkpoint {args.checkpoint or 'initialisation'}</div></header><main>"
        + "".join(rows) + "</main></body></html>"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "terminal_album.html").write_text(html, encoding="utf-8")
    (out / "cases.json").write_text(json.dumps(
        [dict(uid=c["uid"], accepted=None, reason="unreviewed", edit=c["edit"],
              quality=dict(hf_effective=c["hf_hat"], hf_parent=c["hf_src"],
                           pending_review=True))
         for c in cases], indent=1) + "\n", encoding="utf-8")
    print(json.dumps(dict(cases=len(cases), mean_hf_ratio=mean_ratio,
                          album=str(out / "terminal_album.html"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
