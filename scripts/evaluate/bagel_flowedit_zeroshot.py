#!/usr/bin/env python3
"""Zero-shot FlowEdit on frozen BAGEL.

Only official pieces:
  InterleaveInferencer.{init_gen_context, update_context_text,
    update_context_image, gen_image, gen_image_flowedit, encode_image,
    decode_image}
  Bagel._forward_flow / prepare_image_schedule (via gen_image_flowedit)
  inference.ipynb T2I / Editing / Understanding hypers

Do not call interleave_inference for the FlowEdit path (two prefixes).
Default prefix is 4.1 (text-only caches). 4.2 is --prefix-mode edit.
"""

from __future__ import annotations

import argparse
import html
import json
from copy import deepcopy
from pathlib import Path

from PIL import Image

from bagel_common import add_native_model_args, load_native_bagel
from qwen_latent_cot.bagel.modeling._bagel_utils import pil_img2rgb


T2I_HYPER = dict(
    cfg_text_scale=4.0,
    cfg_img_scale=1.0,
    cfg_interval=[0.4, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="global",
)
EDIT_HYPER = dict(
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
)
UND_HYPER = dict(
    max_think_token_n=1000,
    do_sample=False,
    understanding_output=True,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_native_model_args(parser)
    parser.add_argument("--source-image", default="")
    parser.add_argument("--c-src", default="four pigs under six yellow candles")
    parser.add_argument(
        "--c-tar",
        default="three white bagels behind four pigs under six yellow candles",
    )
    parser.add_argument(
        "--pairs-file",
        default="",
        help="JSONL with {c_src,c_tar} per line. Overrides --c-src/--c-tar.",
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--prefix-mode",
        default="text",
        choices=("text", "edit"),
        help="text = doc 4.1 paper setting; edit = doc 4.2 official edit caches",
    )
    parser.add_argument("--n-min", type=float, default=0.2)
    parser.add_argument("--n-max", type=float, default=0.8)
    parser.add_argument("--n-avg", type=int, default=1)
    parser.add_argument(
        "--caption-src",
        action="store_true",
        help="Overwrite c_src with official Understanding caption of the source.",
    )
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Rebuild root index.html from pXXX dirs. Does not load the model.",
    )
    return parser.parse_args()


def load_pairs(args) -> list[dict]:
    path = str(args.pairs_file).strip()
    if path:
        rows = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append({"c_src": str(row["c_src"]), "c_tar": str(row["c_tar"])})
        if not rows:
            raise ValueError(f"empty pairs file: {path}")
        return rows
    return [{"c_src": str(args.c_src), "c_tar": str(args.c_tar)}]


def shard_pairs(pairs: list[dict], shard_id: int, num_shards: int) -> list[tuple[int, dict]]:
    if int(num_shards) < 1:
        raise ValueError("num-shards must be >= 1")
    if not 0 <= int(shard_id) < int(num_shards):
        raise ValueError("shard-id must be in [0, num-shards)")
    return [
        (index, pair)
        for index, pair in enumerate(pairs)
        if index % int(num_shards) == int(shard_id)
    ]


def resize_source(inferencer, image: Image.Image):
    image = inferencer.vae_transform.resize_transform(pil_img2rgb(image))
    image_shape = image.size[::-1]
    return image, image_shape


def t2i_contexts(inferencer, text: str):
    ctx = inferencer.update_context_text(text, inferencer.init_gen_context())
    return ctx, inferencer.init_gen_context(), deepcopy(ctx)


def edit_contexts(inferencer, image: Image.Image, text: str):
    gen = inferencer.update_context_image(
        image, inferencer.init_gen_context(), vae=True, vit=True
    )
    cfg_text = deepcopy(gen)
    cfg_img = inferencer.update_context_text(text, inferencer.init_gen_context())
    gen = inferencer.update_context_text(text, gen)
    return gen, cfg_text, cfg_img


def official_t2i(inferencer, text: str, image_shape, **hyper):
    ctx, cfg_text, cfg_img = t2i_contexts(inferencer, text)
    return inferencer.gen_image(
        image_shape,
        ctx,
        cfg_text_precontext=cfg_text,
        cfg_img_precontext=cfg_img,
        **hyper,
    )


def official_edit(inferencer, image: Image.Image, text: str, image_shape, **hyper):
    ctx, cfg_text, cfg_img = edit_contexts(inferencer, image, text)
    return inferencer.gen_image(
        image_shape,
        ctx,
        cfg_text_precontext=cfg_text,
        cfg_img_precontext=cfg_img,
        **hyper,
    )


def write_pair_gallery(
    pair_dir: Path, panels: list[tuple[str, str]], meta: dict
) -> None:
    cells = "".join(
        f"<td><img src='{html.escape(src)}'><div class='cap'>{html.escape(cap)}</div></td>"
        for src, cap in panels
    )
    page = f"""<!doctype html><meta charset='utf-8'>
<title>BAGEL FlowEdit {html.escape(pair_dir.name)}</title>
<style>body{{font:13px system-ui;background:#111;color:#eee;margin:20px}}
td{{padding:6px;border:1px solid #333;vertical-align:top}}
img{{width:256px;height:256px;object-fit:contain;background:#000}}
.cap{{font-size:12px;color:#aaa;max-width:256px}}</style>
<h1>{html.escape(pair_dir.name)}</h1>
<pre>{html.escape(json.dumps(meta, indent=2, ensure_ascii=False))}</pre>
<table><tr>{cells}</tr></table>
"""
    (pair_dir / "index.html").write_text(page, encoding="utf-8")


def merge_gallery(output_dir: Path) -> None:
    blocks = []
    for pair_dir in sorted(output_dir.glob("p[0-9][0-9][0-9]")):
        manifest_path = pair_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        rel = pair_dir.name
        cells = []
        for name, cap in (
            ("source.png", "source T2I(c_src)"),
            ("source_recon.png", "decode(encode)"),
            ("control_t2i_or_edit.png", "ctrl1 gen_image(c_tar)"),
            ("control_from_xsrc0.png", "ctrl2 v_tar from x_src0"),
            ("flowedit.png", "ctrl3 FlowEdit"),
        ):
            if (pair_dir / name).is_file():
                cells.append(
                    f"<td><img src='{html.escape(rel + '/' + name)}'>"
                    f"<div class='cap'>{html.escape(cap)}</div></td>"
                )
        blocks.append(
            "<h2>"
            + html.escape(rel)
            + "</h2><p class='prompt'>src: "
            + html.escape(str(meta.get("c_src", "")))
            + "<br>tar: "
            + html.escape(str(meta.get("c_tar", "")))
            + "</p><table><tr>"
            + "".join(cells)
            + "</tr></table>"
        )
    page = f"""<!doctype html><meta charset='utf-8'>
<title>BAGEL FlowEdit zero-shot</title>
<style>body{{font:13px system-ui;background:#111;color:#eee;margin:20px}}
td{{padding:6px;border:1px solid #333;vertical-align:top}}
img{{width:200px;height:200px;object-fit:contain;background:#000}}
.cap{{font-size:12px;color:#aaa;max-width:200px}}
.prompt{{color:#ffd479;font-size:14px}}</style>
<h1>BAGEL FlowEdit zero-shot</h1>
{"".join(blocks) or "<p>no pair dirs yet</p>"}
"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"[merge] {output_dir / 'index.html'}", flush=True)


def run_pair(args, inferencer, pair_dir: Path, c_src: str, c_tar: str) -> None:
    pair_dir.mkdir(parents=True, exist_ok=True)
    prefix_mode = str(args.prefix_mode)
    hyper = dict(EDIT_HYPER if prefix_mode == "edit" else T2I_HYPER)

    if str(args.source_image).strip():
        source = Image.open(args.source_image)
        print(f"[source] file {args.source_image}", flush=True)
    else:
        print(f"[source] official T2I of c_src: {c_src!r}", flush=True)
        dummy = Image.new("RGB", (512, 512))
        _, image_shape = resize_source(inferencer, dummy)
        source = official_t2i(inferencer, c_src, image_shape, **T2I_HYPER)

    source, image_shape = resize_source(inferencer, source)
    source.save(pair_dir / "source.png")
    x_src0 = inferencer.encode_image(source, image_shape)
    recon = inferencer.decode_image(x_src0, image_shape)
    recon.save(pair_dir / "source_recon.png")
    print(f"[source] shape={image_shape} latent={tuple(x_src0.shape)}", flush=True)

    if args.caption_src:
        caption = inferencer(
            image=source,
            text="Describe this image in one sentence, including object counts.",
            **UND_HYPER,
        )["text"]
        c_src = (caption or "").strip() or c_src
        (pair_dir / "c_src_caption.txt").write_text(c_src + "\n", encoding="utf-8")
        print(f"[und] c_src <- {c_src[:180]!r}", flush=True)

    if prefix_mode == "text":
        src_triple = t2i_contexts(inferencer, c_src)
        tar_triple = t2i_contexts(inferencer, c_tar)
    else:
        src_triple = edit_contexts(inferencer, source, c_src)
        tar_triple = edit_contexts(inferencer, source, c_tar)

    panels = [
        ("source.png", "source"),
        ("source_recon.png", "decode(encode(source))"),
    ]

    if not args.skip_controls:
        print("[ctrl1] official gen_image from noise under c_tar", flush=True)
        if prefix_mode == "text":
            ctrl1 = official_t2i(inferencer, c_tar, image_shape, **T2I_HYPER)
        else:
            ctrl1 = official_edit(inferencer, source, c_tar, image_shape, **EDIT_HYPER)
        ctrl1.save(pair_dir / "control_t2i_or_edit.png")
        panels.append(("control_t2i_or_edit.png", "ctrl1 official gen_image(c_tar)"))

        print("[ctrl2] official gen_image from x_src0 under c_tar (no delta)", flush=True)
        tar_ctx, tar_ct, tar_ci = tar_triple
        ctrl2 = inferencer.gen_image(
            image_shape,
            tar_ctx,
            cfg_text_precontext=tar_ct,
            cfg_img_precontext=tar_ci,
            init_noise=x_src0,
            **hyper,
        )
        ctrl2.save(pair_dir / "control_from_xsrc0.png")
        panels.append(("control_from_xsrc0.png", "ctrl2 v_tar from x_src0"))

    print(
        f"[flowedit] prefix={prefix_mode} n_min={args.n_min} n_max={args.n_max} n_avg={args.n_avg}",
        flush=True,
    )
    src_ctx, src_ct, src_ci = src_triple
    tar_ctx, tar_ct, tar_ci = tar_triple
    edited = inferencer.gen_image_flowedit(
        image_shape,
        x_src0,
        src_ctx,
        src_ct,
        src_ci,
        tar_ctx,
        tar_ct,
        tar_ci,
        n_min=float(args.n_min),
        n_max=float(args.n_max),
        n_avg=int(args.n_avg),
        **hyper,
    )
    edited.save(pair_dir / "flowedit.png")
    panels.append(("flowedit.png", "ctrl3 FlowEdit v_tar - v_src"))

    meta = {
        "schema": "bagel_flowedit_zeroshot_v1",
        "prefix_mode": prefix_mode,
        "c_src": c_src,
        "c_tar": c_tar,
        "n_min": float(args.n_min),
        "n_max": float(args.n_max),
        "n_avg": int(args.n_avg),
        "image_shape": list(image_shape),
        "hyper": hyper,
        "note": "ctrl3 vs ctrl2 is the integrator; ctrl1 is a new sample of c_tar",
    }
    (pair_dir / "run_manifest.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_pair_gallery(pair_dir, panels, meta)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge_gallery(output_dir)
        return

    pairs = load_pairs(args)
    assigned = shard_pairs(pairs, int(args.shard_id), int(args.num_shards))
    print(
        f"[run] pairs={len(pairs)} shard={args.shard_id}/{args.num_shards} "
        f"assigned={len(assigned)}",
        flush=True,
    )
    if not assigned:
        print("[run] nothing assigned to this shard", flush=True)
        return

    print("[model] loading frozen BAGEL", flush=True)
    _, inferencer = load_native_bagel(args)
    for index, pair in assigned:
        tag = f"p{index:03d}"
        print(f"[{tag}] src={pair['c_src']!r}", flush=True)
        print(f"[{tag}] tar={pair['c_tar']!r}", flush=True)
        run_pair(args, inferencer, output_dir / tag, pair["c_src"], pair["c_tar"])
        print(f"[{tag}] done", flush=True)
    merge_gallery(output_dir)
    print(f"[done] shard={args.shard_id} gallery={output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
