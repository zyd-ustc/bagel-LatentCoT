#!/usr/bin/env python3
"""Phase-0 zero-shot T2I: A0–A5 hidden memory loop, identical ε.

Call path is the official notebook T2I:
    inferencer(text=prompt, init_noise=ε, **NOTEBOOK_T2I_HYPER)

Only two intentional deltas vs inference.ipynb:
    1. cfg_interval = [0.0, 1.0]  (notebook T2I uses [0.4, 1.0])
    2. BagelConfig loop fields are swapped per arm; inferencer kwargs stay
       vanilla (no think / FlowEdit / SDE / TaylorSeer / loop_state_scale).

Shared across arms of one prompt: prompt, seed, init_noise, steps, CFG, image size.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from bagel_common import (
    add_native_model_args,
    load_native_bagel,
    make_noise,
    pixel_mae,
    stable_noise_seed,
)


# Official inference.ipynb T2I hyper, with cfg_interval opened for the whole
# reverse trajectory as specified for this round.
NOTEBOOK_T2I_HYPER = dict(
    cfg_text_scale=4.0,
    cfg_img_scale=1.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="global",
)

ARMS: List[Dict[str, Any]] = [
    {
        "id": "A0",
        "slug": "a0_vanilla",
        "title": "A0 Vanilla",
        "K": 0,
        "R": 1,
        "recycle_mode": "same_depth",
        "persist": True,
        "start_layer": 20,
        "end_layer": 28,
    },
    {
        "id": "A1",
        "slug": "a1_token_only",
        "title": "A1 Token-only",
        "K": 8,
        "R": 1,
        "recycle_mode": "same_depth",
        "persist": True,
        "start_layer": 20,
        "end_layer": 28,
    },
    {
        "id": "A2",
        "slug": "a2_main",
        "title": "A2 Main same_depth R=2 persist",
        "K": 8,
        "R": 2,
        "recycle_mode": "same_depth",
        "persist": True,
        "start_layer": 20,
        "end_layer": 28,
    },
    {
        "id": "A3",
        "slug": "a3_depth_r3",
        "title": "A3 Depth scaling R=3",
        "K": 8,
        "R": 3,
        "recycle_mode": "same_depth",
        "persist": True,
        "start_layer": 20,
        "end_layer": 28,
    },
    {
        "id": "A4",
        "slug": "a4_full_depth",
        "title": "A4 Full-depth control",
        "K": 8,
        "R": 2,
        "recycle_mode": "full_depth",
        "persist": True,
        "start_layer": 20,
        "end_layer": 28,
    },
    {
        "id": "A5",
        "slug": "a5_no_persist",
        "title": "A5 No temporal memory",
        "K": 8,
        "R": 2,
        "recycle_mode": "same_depth",
        "persist": False,
        "start_layer": 20,
        "end_layer": 28,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_native_model_args(parser)
    parser.add_argument(
        "--prompt-file",
        default="experiments/data/geneval2_hard_16.txt",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument(
        "--arms",
        default="",
        help="Comma-separated arm ids, e.g. A0,A2. Empty = all.",
    )
    parser.add_argument(
        "--num-loop-tokens",
        type=int,
        default=8,
        help="Allocate loop_memory at load time (max K across arms).",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Rebuild index.html from pXXX dirs. Does not load the model.",
    )
    return parser.parse_args()


def load_prompts(path: str, max_prompts: int = 0) -> List[str]:
    prompts = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not prompts:
        raise ValueError(f"no prompts in {path}")
    if int(max_prompts) > 0:
        prompts = prompts[: int(max_prompts)]
    return prompts


def select_arms(raw: str) -> List[Dict[str, Any]]:
    if not str(raw).strip():
        return list(ARMS)
    wanted = [part.strip().upper() for part in str(raw).split(",") if part.strip()]
    by_id = {arm["id"]: arm for arm in ARMS}
    missing = [name for name in wanted if name not in by_id]
    if missing:
        raise ValueError(f"unknown arms: {missing}")
    return [deepcopy(by_id[name]) for name in wanted]


def shard_indices(n: int, shard_id: int, num_shards: int) -> List[int]:
    if int(num_shards) < 1:
        raise ValueError("num-shards must be >= 1")
    if not 0 <= int(shard_id) < int(num_shards):
        raise ValueError("shard-id must be in [0, num-shards)")
    return [
        index
        for index in range(int(n))
        if index % int(num_shards) == int(shard_id)
    ]


def apply_loop_config(model, arm: Dict[str, Any]) -> None:
    """Swap Phase-0 memory-loop fields. Does not touch the inferencer API."""

    k = int(arm["K"])
    r = int(arm["R"])
    mode = str(arm["recycle_mode"])
    persist = bool(arm["persist"])
    start = int(arm["start_layer"])
    end = int(arm["end_layer"])
    if k < 0:
        raise ValueError("K must be >= 0")
    if r < 1:
        raise ValueError("R must be >= 1")
    if mode not in ("same_depth", "full_depth"):
        raise ValueError("recycle_mode must be same_depth or full_depth")
    if k > 0 and getattr(model, "loop_memory", None) is None:
        raise ValueError("K>0 requires loop_memory allocated at load time")
    if k > 0 and int(model.loop_memory.shape[0]) < k:
        raise ValueError(
            f"loop_memory has K={int(model.loop_memory.shape[0])}, arm wants {k}"
        )

    model.config.num_loop_tokens = k
    model.config.loop_depth = r
    model.config.loop_recycle_mode = mode
    model.config.loop_memory_persist = persist
    model.config.memory_loop_start_layer = start
    model.config.memory_loop_end_layer = end
    model.num_loop_tokens = k
    model.loop_depth = r
    model.loop_recycle_mode = mode
    model.loop_memory_persist = persist
    model.memory_loop_start_layer = start
    model.memory_loop_end_layer = end


def official_t2i(inferencer, prompt: str, init_noise: torch.Tensor, image_shape):
    """Notebook-identical T2I entry. Loop behaviour comes only from BagelConfig."""

    return inferencer(
        text=prompt,
        init_noise=init_noise,
        image_shapes=tuple(image_shape),
        enable_taylorseer=False,
        **NOTEBOOK_T2I_HYPER,
    )["image"]


def tensor_digest(tensor: torch.Tensor) -> str:
    payload = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return None
    return value


def summarize_diagnostics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n_steps": 0, "n_inner": 0}
    def _pairwise_value(row: Dict[str, Any]):
        raw = row.get("mean_abs_pairwise_cosine", row.get("pairwise_cosine"))
        if isinstance(raw, (int, float)) and not math.isnan(float(raw)):
            return float(raw)
        return None

    pairwise = [value for value in (_pairwise_value(row) for row in rows) if value is not None]
    ranks = [
        float(row["effective_rank"])
        for row in rows
        if isinstance(row.get("effective_rank"), (int, float))
        and not math.isnan(float(row["effective_rank"]))
    ]
    cos_prev = [
        float(row["memory_cosine_to_prev"])
        for row in rows
        if isinstance(row.get("memory_cosine_to_prev"), (int, float))
        and not math.isnan(float(row["memory_cosine_to_prev"]))
    ]
    return {
        "n_inner": len(rows),
        "n_steps": len({int(row["step"]) for row in rows if "step" in row}),
        "mean_abs_pairwise_cosine": (
            sum(pairwise) / len(pairwise) if pairwise else None
        ),
        "mean_pairwise_cosine": (
            sum(pairwise) / len(pairwise) if pairwise else None
        ),
        "mean_effective_rank": sum(ranks) / len(ranks) if ranks else None,
        "mean_memory_cosine_to_prev": (
            sum(cos_prev) / len(cos_prev) if cos_prev else None
        ),
        "last": _jsonable(rows[-1]),
    }


def init_frozen_memory(model, inferencer, seed: int) -> Optional[str]:
    if getattr(model, "loop_memory", None) is None:
        return None
    torch.manual_seed(int(seed))
    model.init_loop_memory_from_boundary_embeddings(
        [
            int(inferencer.new_token_ids["start_of_image"]),
            int(inferencer.new_token_ids["end_of_image"]),
        ]
    )
    model.loop_memory.requires_grad_(False)
    return tensor_digest(model.loop_memory)


def write_prompt_gallery(
    prompt_dir: Path, prompt: str, arms: Sequence[Dict[str, Any]], meta: dict
) -> None:
    cells = []
    for arm in arms:
        name = f"{arm['slug']}.png"
        if not (prompt_dir / name).is_file():
            continue
        mae = meta.get("pixel_mae_vs_A0", {}).get(arm["id"])
        mae_txt = "" if mae is None else f" mae_A0={mae:.2f}"
        cells.append(
            "<td><img src='"
            + html.escape(name)
            + "'><div class='cap'>"
            + html.escape(arm["title"] + mae_txt)
            + "</div></td>"
        )
    page = f"""<!doctype html><meta charset='utf-8'>
<title>{html.escape(prompt_dir.name)}</title>
<style>body{{font:13px system-ui;background:#111;color:#eee;margin:20px}}
td{{padding:6px;border:1px solid #333;vertical-align:top}}
img{{width:256px;height:256px;object-fit:contain;background:#000}}
.cap{{font-size:12px;color:#aaa;max-width:256px}}</style>
<h1>{html.escape(prompt_dir.name)}</h1>
<p class='prompt'>{html.escape(prompt)}</p>
<table><tr>{''.join(cells)}</tr></table>
<pre>{html.escape(json.dumps(meta, indent=2, ensure_ascii=False))}</pre>
"""
    (prompt_dir / "index.html").write_text(page, encoding="utf-8")


def merge_gallery(output_dir: Path, arms: Sequence[Dict[str, Any]]) -> None:
    header = "".join(
        f"<th>{html.escape(arm['id'])}<br><span>{html.escape(arm['title'])}</span></th>"
        for arm in arms
    )
    rows = []
    for prompt_dir in sorted(output_dir.glob("p[0-9][0-9][0-9]")):
        manifest_path = prompt_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        prompt = str(meta.get("prompt", ""))
        rel = prompt_dir.name
        cells = [
            f"<td class='id'>{html.escape(rel)}</td>"
            f"<td class='prompt'>{html.escape(prompt)}</td>"
        ]
        mae_map = meta.get("pixel_mae_vs_A0", {})
        for arm in arms:
            name = f"{arm['slug']}.png"
            path = prompt_dir / name
            if not path.is_file():
                cells.append("<td></td>")
                continue
            mae = mae_map.get(arm["id"])
            mae_txt = "" if mae is None else f"<div class='mae'>mae_A0={mae:.2f}</div>"
            cells.append(
                f"<td><img src='{html.escape(rel + '/' + name)}'>"
                f"<div class='cap'>{html.escape(arm['id'])}</div>{mae_txt}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    page = f"""<!doctype html><meta charset='utf-8'>
<title>BAGEL Phase-0 loop zero-shot A0–A5</title>
<style>
body{{font:13px system-ui;background:#111;color:#eee;margin:20px}}
table{{border-collapse:collapse}}
th,td{{padding:6px;border:1px solid #333;vertical-align:top}}
th span{{font-weight:normal;color:#aaa;font-size:11px}}
img{{width:192px;height:192px;object-fit:contain;background:#000}}
.prompt{{color:#ffd479;max-width:280px}}
.id{{color:#9cf;white-space:nowrap}}
.cap,.mae{{font-size:11px;color:#aaa}}
</style>
<h1>Phase-0 memory-loop zero-shot</h1>
<p>Official <code>inferencer(text=P, init_noise=ε, **T2I_hyper)</code>,
<code>cfg_interval=[0,1]</code>, no FlowEdit / SDE / TaylorSeer.</p>
<table>
<tr><th>id</th><th>prompt</th>{header}</tr>
{''.join(rows) or '<tr><td>no prompt dirs yet</td></tr>'}
</table>
"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"[merge] {output_dir / 'index.html'}", flush=True)


def run_prompt(
    args,
    inferencer,
    prompt_dir: Path,
    prompt: str,
    prompt_index: int,
    arms: Sequence[Dict[str, Any]],
) -> None:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    model = inferencer.model
    image_shape = (int(args.image_size), int(args.image_size))
    noise_seed = stable_noise_seed(int(args.seed), prompt)
    init_noise = make_noise(model, image_shape, noise_seed)
    noise_hash = tensor_digest(init_noise)
    (prompt_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    images: Dict[str, Any] = {}
    arm_rows: List[Dict[str, Any]] = []
    for arm in arms:
        apply_loop_config(model, arm)
        print(
            f"[{prompt_dir.name} {arm['id']}] K={arm['K']} R={arm['R']} "
            f"{arm['recycle_mode']} persist={arm['persist']} "
            f"layers=[{arm['start_layer']},{arm['end_layer']})",
            flush=True,
        )
        image = official_t2i(
            inferencer,
            prompt,
            init_noise.clone(),
            image_shape,
        )
        image_path = prompt_dir / f"{arm['slug']}.png"
        image.save(image_path)
        images[arm["id"]] = image
        diag = summarize_diagnostics(getattr(model, "last_loop_diagnostics", []) or [])
        (prompt_dir / f"{arm['slug']}_diag.json").write_text(
            json.dumps(diag, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        arm_rows.append(
            {
                "id": arm["id"],
                "slug": arm["slug"],
                "K": arm["K"],
                "R": arm["R"],
                "recycle_mode": arm["recycle_mode"],
                "persist": arm["persist"],
                "start_layer": arm["start_layer"],
                "end_layer": arm["end_layer"],
                "image": image_path.name,
                "diagnostics": diag,
            }
        )
        print(
            f"[{prompt_dir.name} {arm['id']}] wrote {image_path.name} "
            f"diag_steps={diag.get('n_steps')}",
            flush=True,
        )

    mae = {}
    if "A0" in images:
        for arm in arms:
            if arm["id"] == "A0":
                continue
            if arm["id"] in images:
                mae[arm["id"]] = pixel_mae(images[arm["id"]], images["A0"])

    meta = {
        "schema": "bagel_loop_zeroshot_v1",
        "prompt_index": int(prompt_index),
        "prompt": prompt,
        "seed": int(args.seed),
        "noise_seed": int(noise_seed),
        "noise_sha256": noise_hash,
        "image_shape": list(image_shape),
        "hyper": NOTEBOOK_T2I_HYPER,
        "notebook_cfg_interval_default": [0.4, 1.0],
        "forbidden": {
            "flowedit": False,
            "sde": False,
            "taylorseer": False,
            "think": False,
            "loop_state_scale": None,
        },
        "pixel_mae_vs_A0": mae,
        "arms": arm_rows,
    }
    (prompt_dir / "run_manifest.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_prompt_gallery(prompt_dir, prompt, arms, meta)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arms = select_arms(args.arms)
    if args.merge_only:
        merge_gallery(output_dir, arms)
        return

    prompts = load_prompts(args.prompt_file, int(args.max_prompts))
    assigned = shard_indices(len(prompts), int(args.shard_id), int(args.num_shards))
    print(
        f"[run] prompts={len(prompts)} shard={args.shard_id}/{args.num_shards} "
        f"assigned={assigned} arms={[arm['id'] for arm in arms]}",
        flush=True,
    )
    if not assigned:
        print("[run] nothing assigned to this shard", flush=True)
        return

    print("[model] loading frozen BAGEL with loop_memory K="
          f"{int(args.num_loop_tokens)}", flush=True)
    backbone, inferencer = load_native_bagel(args)
    m0_hash = init_frozen_memory(backbone.bagel, inferencer, int(args.seed))
    print(f"[model] m0_sha256={m0_hash}", flush=True)

    for index in assigned:
        tag = f"p{index:03d}"
        prompt = prompts[index]
        print(f"[{tag}] {prompt!r}", flush=True)
        run_prompt(
            args,
            inferencer,
            output_dir / tag,
            prompt,
            index,
            arms,
        )
        print(f"[{tag}] done", flush=True)
    merge_gallery(output_dir, arms)
    print(f"[done] shard={args.shard_id} gallery={output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
