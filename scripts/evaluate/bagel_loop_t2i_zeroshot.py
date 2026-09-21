#!/usr/bin/env python3
"""Phase 0.5 frozen-BAGEL T2I loop mechanism benchmark.

Every arm shares prompt, initial noise, geometry, CFG, NFE, and timestep
schedule. Only the loop memory configuration changes.
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

from bagel_common import load_native_bagel, make_noise, pixel_mae, stable_noise_seed


T2I_HYPER = dict(
    cfg_text_scale=4.0,
    cfg_img_scale=1.0,
    cfg_interval=[0.4, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="global",
)

ARMS: List[Dict[str, Any]] = [
    {
        "id": "Z0",
        "slug": "z0_vanilla",
        "title": "Z0 vanilla BAGEL",
        "K": 0,
        "R": 1,
        "recycle_mode": "same_depth",
        "persist": False,
        "start_layer": 16,
        "end_layer": 24,
        "round0_memory_write_enabled": False,
    },
    {
        "id": "Z2",
        "slug": "z2_mid",
        "title": "Z2 mid body",
        "K": 8,
        "R": 2,
        "recycle_mode": "same_depth",
        "persist": False,
        "start_layer": 16,
        "end_layer": 24,
        "round0_memory_write_enabled": False,
    },
    {
        "id": "Z3",
        "slug": "z3_early",
        "title": "Z3 early body",
        "K": 8,
        "R": 2,
        "recycle_mode": "same_depth",
        "persist": False,
        "start_layer": 12,
        "end_layer": 20,
        "round0_memory_write_enabled": False,
    },
    {
        "id": "Z4",
        "slug": "z4_late",
        "title": "Z4 late body",
        "K": 8,
        "R": 2,
        "recycle_mode": "same_depth",
        "persist": False,
        "start_layer": 20,
        "end_layer": 28,
        "round0_memory_write_enabled": False,
    },
    {
        "id": "Z6",
        "slug": "z6_early_persist",
        "title": "Z6 early body + persist",
        "K": 8,
        "R": 2,
        "recycle_mode": "same_depth",
        "persist": True,
        "start_layer": 12,
        "end_layer": 20,
        "round0_memory_write_enabled": False,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vit-max-image-size", type=int, default=980)
    parser.add_argument("--vit-min-image-size", type=int, default=224)
    parser.add_argument("--vit-image-stride", type=int, default=14)
    parser.add_argument(
        "--prompt-file",
        default="experiments/data/geneval2_all_800.txt",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument(
        "--arms",
        default="",
        help="Comma-separated arm ids, e.g. Z0,Z2. Empty = all.",
    )
    parser.add_argument(
        "--k-values",
        default="",
        help="Run only Z0 plus strict mid-body K ablation, e.g. 1,4,8.",
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
        help="Rebuild gallery and image maps without loading the model.",
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
        return [deepcopy(arm) for arm in ARMS]
    wanted = [part.strip().upper() for part in str(raw).split(",") if part.strip()]
    by_id = {arm["id"]: arm for arm in ARMS}
    missing = [name for name in wanted if name not in by_id]
    if missing:
        raise ValueError(f"unknown arms: {missing}")
    return [deepcopy(by_id[name]) for name in wanted]


def parse_k_values(raw: str) -> List[int]:
    values = []
    for part in str(raw).split(","):
        if not part.strip():
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid K value: {part!r}") from exc
        if value < 1:
            raise ValueError("K ablation values must be >= 1")
        if value not in values:
            values.append(value)
    return values


def make_k_ablation_arms(values: Sequence[int]) -> List[Dict[str, Any]]:
    vanilla = deepcopy(next(arm for arm in ARMS if arm["id"] == "Z0"))
    base = next(arm for arm in ARMS if arm["id"] == "Z2")
    arms = [vanilla]
    for value in values:
        arm = deepcopy(base)
        arm.update(
            id=f"K{int(value)}",
            slug=f"k{int(value)}_read_write",
            title=f"K={int(value)} strict read→write",
            K=int(value),
        )
        arms.append(arm)
    return arms


def resolve_arms(raw_arms: str, raw_k_values: str) -> List[Dict[str, Any]]:
    k_values = parse_k_values(raw_k_values)
    if k_values:
        if str(raw_arms).strip():
            raise ValueError("--k-values and --arms are separate protocols; choose one")
        return make_k_ablation_arms(k_values)
    return select_arms(raw_arms)


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
    k = int(arm["K"])
    rounds = int(arm["R"])
    mode = str(arm["recycle_mode"])
    round0_write = bool(arm["round0_memory_write_enabled"])
    if k < 0 or rounds < 1:
        raise ValueError("K must be >= 0 and R must be >= 1")
    if mode not in ("same_depth", "full_depth"):
        raise ValueError("recycle_mode must be same_depth or full_depth")
    if k > 0 and getattr(model, "loop_memory", None) is None:
        raise ValueError("K>0 requires loop_memory allocated at load time")
    if k > 0 and int(model.loop_memory.shape[0]) < k:
        raise ValueError(
            f"loop_memory has K={int(model.loop_memory.shape[0])}, arm wants {k}"
        )

    values = {
        "num_loop_tokens": k,
        "loop_depth": rounds,
        "loop_recycle_mode": mode,
        "loop_memory_persist": bool(arm["persist"]),
        "memory_loop_start_layer": int(arm["start_layer"]),
        "memory_loop_end_layer": int(arm["end_layer"]),
        "round0_memory_write_enabled": round0_write,
        "round0_gen_reads_memory": round0_write,
        "num_read_rounds": 0 if round0_write else 1,
        "num_write_rounds": rounds if round0_write else rounds - 1,
    }
    for name, value in values.items():
        setattr(model.config, name, value)
        setattr(model, name, value)


def official_t2i(
    inferencer,
    prompt: str,
    init_noise: torch.Tensor,
    image_shape,
):
    return inferencer(
        image=None,
        text=prompt,
        image_shapes=tuple(image_shape),
        init_noise=init_noise,
        enable_taylorseer=False,
        return_loop_diagnostics=True,
        **T2I_HYPER,
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
        return {
            "n_steps": 0,
            "n_inner": 0,
            "mean_delta_m": None,
            "mean_delta_g": None,
            "mean_delta_v": None,
        }

    def _valid_scalar(row: Dict[str, Any], key: str):
        raw = row.get(key)
        if isinstance(raw, (int, float)) and not math.isnan(float(raw)):
            return float(raw)
        return None

    def _mean_list(key: str):
        values = []
        for row in rows:
            raw = row.get(key)
            if isinstance(raw, (list, tuple)):
                values.extend(
                    float(item)
                    for item in raw
                    if isinstance(item, (int, float)) and not math.isnan(float(item))
                )
        return sum(values) / len(values) if values else None

    pairwise = [
        value
        for value in (
            _valid_scalar(
                row,
                "mean_abs_pairwise_cosine"
                if "mean_abs_pairwise_cosine" in row
                else "pairwise_cosine",
            )
            for row in rows
        )
        if value is not None
    ]
    ranks = [
        value
        for value in (_valid_scalar(row, "effective_rank") for row in rows)
        if value is not None
    ]
    cos_prev = [
        value
        for value in (_valid_scalar(row, "memory_cosine_to_prev") for row in rows)
        if value is not None
    ]
    return {
        "n_inner": len(rows),
        "n_steps": len({int(row["step"]) for row in rows if "step" in row}),
        "mean_abs_pairwise_cosine": sum(pairwise) / len(pairwise)
        if pairwise
        else None,
        "mean_effective_rank": sum(ranks) / len(ranks) if ranks else None,
        "mean_memory_cosine_to_prev": sum(cos_prev) / len(cos_prev)
        if cos_prev
        else None,
        "mean_delta_m": _mean_list("delta_m"),
        "mean_delta_g": _mean_list("delta_g"),
        "mean_delta_v": _mean_list("delta_v"),
        "last": _jsonable(rows[-1]),
    }


def init_frozen_memory(model) -> Optional[str]:
    if getattr(model, "loop_memory", None) is None:
        return None
    model.loop_memory.requires_grad_(False)
    return tensor_digest(model.loop_memory)


def aggregate_mechanism_rows(
    prompt_rows: Sequence[Dict[str, Any]],
    arms: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    def mean(values):
        valid = [float(value) for value in values if isinstance(value, (int, float))]
        return sum(valid) / len(valid) if valid else None

    summary = []
    for arm in arms:
        arm_rows = []
        mae_values = []
        for prompt_row in prompt_rows:
            match = next(
                (row for row in prompt_row.get("arms", []) if row["id"] == arm["id"]),
                None,
            )
            if match is not None:
                arm_rows.append(match)
            if arm["id"] == "Z0":
                mae_values.append(0.0)
            else:
                mae_values.append(
                    prompt_row.get("pixel_mae_vs_Z0", {}).get(arm["id"])
                )
        summary.append(
            {
                **{key: value for key, value in arm.items()},
                "prompt_count": len(arm_rows),
                "mean_pixel_mae_vs_Z0": mean(mae_values),
                "mean_delta_m": mean(
                    row.get("diagnostics", {}).get("mean_delta_m") for row in arm_rows
                ),
                "mean_delta_g": mean(
                    row.get("diagnostics", {}).get("mean_delta_g") for row in arm_rows
                ),
                "mean_delta_v": mean(
                    row.get("diagnostics", {}).get("mean_delta_v") for row in arm_rows
                ),
                "mean_effective_rank": mean(
                    row.get("diagnostics", {}).get("mean_effective_rank")
                    for row in arm_rows
                ),
            }
        )
    return {
        "schema": "bagel_loop_t2i_mechanism_v1",
        "prompt_count": len(prompt_rows),
        "arms": summary,
    }


def write_prompt_gallery(
    prompt_dir: Path,
    prompt: str,
    arms: Sequence[Dict[str, Any]],
    meta: dict,
) -> None:
    cells = []
    for arm in arms:
        name = f"{arm['slug']}.png"
        if not (prompt_dir / name).is_file():
            continue
        mae = meta.get("pixel_mae_vs_Z0", {}).get(arm["id"])
        mae_text = "" if mae is None else f" mae_Z0={mae:.2f}"
        cells.append(
            f"<td><img src='{html.escape(name)}'><div class='cap'>"
            f"{html.escape(arm['title'] + mae_text)}</div></td>"
        )
    page = f"""<!doctype html><meta charset='utf-8'>
<title>{html.escape(prompt_dir.name)}</title>
<style>body{{font:13px system-ui;background:#111;color:#eee;margin:20px}}
td{{padding:6px;border:1px solid #333;vertical-align:top}}
img{{width:256px;height:256px;object-fit:contain;background:#000}}
.cap{{font-size:12px;color:#aaa;max-width:256px}}</style>
<h1>{html.escape(prompt_dir.name)}</h1>
<p>{html.escape(prompt)}</p><table><tr>{''.join(cells)}</tr></table>
<pre>{html.escape(json.dumps(meta, indent=2, ensure_ascii=False))}</pre>
"""
    (prompt_dir / "index.html").write_text(page, encoding="utf-8")


def merge_gallery(output_dir: Path, arms: Sequence[Dict[str, Any]]) -> None:
    header = "".join(
        f"<th>{html.escape(arm['id'])}<br><span>{html.escape(arm['title'])}</span></th>"
        for arm in arms
    )
    rows = []
    prompt_rows = []
    image_maps = {arm["id"]: {} for arm in arms}
    for prompt_dir in sorted(output_dir.glob("p[0-9][0-9][0-9]")):
        manifest_path = prompt_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        prompt = str(meta["prompt"])
        prompt_rows.append(meta)
        rel = prompt_dir.name
        cells = [
            f"<td class='id'>{html.escape(rel)}</td>",
            f"<td class='prompt'>{html.escape(prompt)}</td>",
        ]
        mae_map = meta.get("pixel_mae_vs_Z0", {})
        for arm in arms:
            name = f"{arm['slug']}.png"
            path = prompt_dir / name
            if not path.is_file():
                cells.append("<td></td>")
                continue
            image_maps[arm["id"]][prompt] = str(path.resolve())
            mae = mae_map.get(arm["id"])
            mae_text = "" if mae is None else f"<div>mae_Z0={mae:.2f}</div>"
            cells.append(
                f"<td><img src='{html.escape(rel + '/' + name)}'>"
                f"<div>{html.escape(arm['id'])}</div>{mae_text}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    map_dir = output_dir / "geneval2"
    map_dir.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        (map_dir / f"{arm['slug']}_image_paths.json").write_text(
            json.dumps(image_maps[arm["id"]], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if prompt_rows:
        mechanism = aggregate_mechanism_rows(prompt_rows, arms)
        (output_dir / "mechanism_summary.json").write_text(
            json.dumps(mechanism, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        root_manifest = {
            "schema": "bagel_loop_t2i_phase05_v1",
            "prompts": [row["prompt"] for row in prompt_rows],
            "prompt_count": len(prompt_rows),
            "image_shape": prompt_rows[0]["image_shape"],
            "hyper": T2I_HYPER,
            "arms": [
                {key: value for key, value in arm.items()} for arm in arms
            ],
            "mechanism_summary": "mechanism_summary.json",
            "geneval2_image_maps": {
                arm["id"]: f"geneval2/{arm['slug']}_image_paths.json"
                for arm in arms
            },
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(root_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    page = f"""<!doctype html><meta charset='utf-8'>
<title>BAGEL loop T2I Phase 0.5</title>
<style>body{{font:13px system-ui;background:#111;color:#eee;margin:20px}}
table{{border-collapse:collapse}}th,td{{padding:6px;border:1px solid #333;vertical-align:top}}
th span{{font-weight:normal;color:#aaa;font-size:11px}}
img{{width:192px;height:192px;object-fit:contain;background:#000}}
.prompt{{color:#ffd479;max-width:280px}}.id{{color:#9cf;white-space:nowrap}}</style>
<h1>Frozen BAGEL T2I Read→Write loop</h1>
<p>Same prompt, noise, geometry, CFG, NFE, and schedule across arms.</p>
<table><tr><th>id</th><th>prompt</th>{header}</tr>
{''.join(rows) or '<tr><td>no prompt dirs yet</td></tr>'}</table>
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
    image_shape = (int(args.height), int(args.width))
    noise_seed = stable_noise_seed(
        int(args.seed), prompt, schema="bagel-loop-t2i-v1"
    )
    init_noise = make_noise(model, image_shape, noise_seed)
    noise_hash = tensor_digest(init_noise)
    (prompt_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    images: Dict[str, Any] = {}
    arm_rows = []
    for arm in arms:
        apply_loop_config(model, arm)
        print(
            f"[{prompt_dir.name} {arm['id']}] K={arm['K']} R={arm['R']} "
            f"{arm['recycle_mode']} persist={arm['persist']} "
            f"layers=[{arm['start_layer']},{arm['end_layer']}) "
            f"round0_write={arm['round0_memory_write_enabled']}",
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
        diagnostics = summarize_diagnostics(
            getattr(model, "last_loop_diagnostics", []) or []
        )
        (prompt_dir / f"{arm['slug']}_diag.json").write_text(
            json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        arm_rows.append(
            {
                **{key: value for key, value in arm.items()},
                "num_read_rounds": 0
                if arm["round0_memory_write_enabled"]
                else 1,
                "num_write_rounds": arm["R"]
                if arm["round0_memory_write_enabled"]
                else arm["R"] - 1,
                "image": image_path.name,
                "diagnostics": diagnostics,
            }
        )

    mae = {}
    if "Z0" in images:
        for arm in arms:
            if arm["id"] != "Z0" and arm["id"] in images:
                mae[arm["id"]] = pixel_mae(images[arm["id"]], images["Z0"])
    meta = {
        "schema": "bagel_loop_t2i_phase05_v1",
        "prompt_index": int(prompt_index),
        "prompt": prompt,
        "seed": int(args.seed),
        "noise_seed": int(noise_seed),
        "noise_sha256": noise_hash,
        "image_shape": list(image_shape),
        "hyper": T2I_HYPER,
        "pixel_mae_vs_Z0": mae,
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
    arms = resolve_arms(args.arms, args.k_values)
    required_k = max(int(arm["K"]) for arm in arms)
    if required_k > int(args.num_loop_tokens):
        raise ValueError(
            f"selected arms require K={required_k}, "
            f"but --num-loop-tokens={int(args.num_loop_tokens)}"
        )
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
        return

    print(
        f"[model] loading frozen BAGEL with loop_memory K={args.num_loop_tokens}",
        flush=True,
    )
    backbone, inferencer = load_native_bagel(args)
    model = backbone.bagel
    downsample = int(model.latent_downsample)
    if int(args.height) % downsample or int(args.width) % downsample:
        raise ValueError(
            f"height and width must be divisible by latent_downsample={downsample}"
        )
    print(f"[model] m0_sha256={init_frozen_memory(model)}", flush=True)

    for index in assigned:
        tag = f"p{index:03d}"
        prompt = prompts[index]
        print(f"[{tag}] {prompt!r}", flush=True)
        run_prompt(args, inferencer, output_dir / tag, prompt, index, arms)
    if int(args.num_shards) == 1:
        merge_gallery(output_dir, arms)
    print(f"[done] shard={args.shard_id}", flush=True)


if __name__ == "__main__":
    main()
