#!/usr/bin/env python3
"""Phase 0.5 paired semantic-edit evaluation harness and isolated K ablation.

Call path is the official notebook editor:
    inferencer(image=source, text=edit, init_noise=ε, **NOTEBOOK_EDIT_HYPER)

Inferencer kwargs stay notebook-like (no think / FlowEdit / SDE / TaylorSeer).
Shared across arms: source, instruction, seed, init noise, CFG, NFE.
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
from qwen_latent_cot.bagel.modeling._bagel_utils import pil_img2rgb


NOTEBOOK_EDIT_HYPER = dict(
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
)

ARMS: List[Dict[str, Any]] = [
    {
        "id": "Z0",
        "slug": "z0_vanilla",
        "title": "Z0 Vanilla",
        "K": 0,
        "R": 1,
        "recycle_mode": "same_depth",
        "persist": False,
        "start_layer": 16,
        "end_layer": 24,
        "remove_old_prompt": True,
        "round0_memory_write_enabled": False,
    },
    {
        "id": "Z1",
        "slug": "z1_old_loop",
        "title": "Z1 old loop",
        "K": 8,
        "R": 2,
        "recycle_mode": "same_depth",
        "persist": True,
        "start_layer": 20,
        "end_layer": 28,
        "remove_old_prompt": False,
        "round0_memory_write_enabled": True,
    },
    {
        "id": "Z2",
        "slug": "z2_read_write",
        "title": "Z2 strict read→write",
        "K": 8,
        "R": 2,
        "recycle_mode": "same_depth",
        "persist": False,
        "start_layer": 16,
        "end_layer": 24,
        "remove_old_prompt": True,
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
        "remove_old_prompt": True,
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
        "remove_old_prompt": True,
        "round0_memory_write_enabled": False,
    },
    {
        "id": "Z5",
        "slug": "z5_persist",
        "title": "Z5 persistent memory",
        "K": 8,
        "R": 2,
        "recycle_mode": "same_depth",
        "persist": True,
        "start_layer": 16,
        "end_layer": 24,
        "remove_old_prompt": True,
        "round0_memory_write_enabled": False,
    },
    {
        "id": "C0",
        "slug": "c0_full_depth",
        "title": "C0 full-depth control",
        "K": 8,
        "R": 2,
        "recycle_mode": "full_depth",
        "persist": False,
        "start_layer": 16,
        "end_layer": 24,
        "remove_old_prompt": True,
        "round0_memory_write_enabled": False,
    },
    {
        "id": "C1",
        "slug": "c1_read_only",
        "title": "C1 strict read-only",
        "K": 8,
        "R": 1,
        "recycle_mode": "same_depth",
        "persist": False,
        "start_layer": 16,
        "end_layer": 24,
        "remove_old_prompt": True,
        "round0_memory_write_enabled": False,
    },
    {
        "id": "C2",
        "slug": "c2_keep_old_prompt",
        "title": "C2 strict read→write + old prompt",
        "K": 8,
        "R": 2,
        "recycle_mode": "same_depth",
        "persist": False,
        "start_layer": 16,
        "end_layer": 24,
        "remove_old_prompt": False,
        "round0_memory_write_enabled": False,
    },
]

PAIRED_EDIT_FIELDS = (
    "id",
    "source_image",
    "source_prompt",
    "instruction",
    "target",
    "preserve",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_native_model_args(parser)
    parser.add_argument(
        "--prompt-file",
        default="experiments/data/geneval2_hard_16.txt",
    )
    parser.add_argument(
        "--edit-file",
        default="",
        help="Paired-edit JSONL. When set, source image/prompt are read per case.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Deprecated compatibility option; geometry follows the resized source.",
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument(
        "--arms",
        default="",
        help="Comma-separated main arm ids, e.g. Z0,Z2. Empty = all.",
    )
    parser.add_argument(
        "--k-values",
        default="",
        help="Run only strict mid-body K ablation, e.g. 1,4,8; cannot combine with --arms.",
    )
    parser.add_argument(
        "--num-loop-tokens",
        type=int,
        default=8,
        help="Allocate loop_memory at load time (max K across arms).",
    )
    parser.add_argument("--source-image", default="")
    parser.add_argument("--source-prompt", default="")
    parser.add_argument(
        "--old-prompt",
        default="",
        help="Original T2I prompt for Z1 keep-old-prompt arms. Defaults to source-prompt.",
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


def load_edit_cases(path: str, max_cases: int = 0) -> List[Dict[str, Any]]:
    """Load paired semantic edits and resolve source images relative to JSONL."""

    jsonl_path = Path(path).expanduser().resolve()
    cases: List[Dict[str, Any]] = []
    seen = set()
    for line_number, raw in enumerate(
        jsonl_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            case = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{jsonl_path}:{line_number}: invalid JSON: {exc}") from exc
        missing = [field for field in PAIRED_EDIT_FIELDS if field not in case]
        if missing:
            raise ValueError(f"{jsonl_path}:{line_number}: missing fields {missing}")
        case_id = str(case["id"]).strip()
        if not case_id or case_id in seen:
            raise ValueError(f"{jsonl_path}:{line_number}: empty or duplicate id {case_id!r}")
        seen.add(case_id)
        for field in ("source_image", "source_prompt", "instruction", "target"):
            if not str(case[field]).strip():
                raise ValueError(f"{jsonl_path}:{line_number}: {field} must be non-empty")
        preserve = case["preserve"]
        if not isinstance(preserve, list) or not all(
            isinstance(item, str) and item.strip() for item in preserve
        ):
            raise ValueError(f"{jsonl_path}:{line_number}: preserve must be a string list")
        source_path = Path(str(case["source_image"])).expanduser()
        if not source_path.is_absolute():
            source_path = jsonl_path.parent / source_path
        case = dict(case)
        case["id"] = case_id
        case["source_image"] = str(source_path.resolve())
        cases.append(case)
    if not cases:
        raise ValueError(f"no edit cases in {jsonl_path}")
    if int(max_cases) > 0:
        cases = cases[: int(max_cases)]
    return cases


def parse_k_values(raw: str) -> List[int]:
    values: List[int] = []
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
    base = next(arm for arm in ARMS if arm["id"] == "Z2")
    vanilla = deepcopy(next(arm for arm in ARMS if arm["id"] == "Z0"))
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


def prepare_source_image(inferencer, source_image):
    """Match BAGEL's native aspect-ratio-preserving edit geometry."""

    prepared = inferencer.vae_transform.resize_transform(pil_img2rgb(source_image))
    return prepared, prepared.size[::-1]


def resolve_arms(raw_arms: str, raw_k_values: str) -> List[Dict[str, Any]]:
    k_values = parse_k_values(raw_k_values)
    if k_values:
        if str(raw_arms).strip():
            raise ValueError("--k-values and --arms are separate protocols; choose one")
        return make_k_ablation_arms(k_values)
    return select_arms(raw_arms)


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
    """Swap Read–Write loop fields. Does not touch the inferencer API."""

    k = int(arm["K"])
    r = int(arm["R"])
    mode = str(arm["recycle_mode"])
    persist = bool(arm["persist"])
    start = int(arm["start_layer"])
    end = int(arm["end_layer"])
    round0_write = bool(arm["round0_memory_write_enabled"])
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
    model.config.round0_memory_write_enabled = round0_write
    # Keep the deprecated alias synchronized for checkpoints/config readers that
    # have not migrated yet. New call sites use the canonical name above.
    model.config.round0_gen_reads_memory = round0_write
    model.config.num_read_rounds = 0 if round0_write else 1
    model.config.num_write_rounds = r if round0_write else r - 1
    model.num_loop_tokens = k
    model.loop_depth = r
    model.loop_recycle_mode = mode
    model.loop_memory_persist = persist
    model.memory_loop_start_layer = start
    model.memory_loop_end_layer = end
    model.round0_memory_write_enabled = round0_write
    model.round0_gen_reads_memory = round0_write
    model.num_read_rounds = 0 if round0_write else 1
    model.num_write_rounds = r if round0_write else r - 1


def official_edit(
    inferencer,
    source_image,
    edit_text: str,
    init_noise: torch.Tensor,
    image_shape,
    *,
    old_prompt: str = "",
    remove_old_prompt: bool = True,
):
    kwargs = dict(
        init_noise=init_noise,
        image_shapes=tuple(image_shape),
        enable_taylorseer=False,
        remove_old_prompt=bool(remove_old_prompt),
        return_loop_diagnostics=True,
        **NOTEBOOK_EDIT_HYPER,
    )
    if old_prompt and not remove_old_prompt:
        outputs = inferencer.interleave_inference(
            [str(old_prompt), source_image, str(edit_text)],
            **kwargs,
        )
        for item in outputs:
            if hasattr(item, "save"):
                return item
        raise RuntimeError("edit inference produced no image")
    return inferencer(
        image=source_image,
        text=edit_text,
        **kwargs,
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
        "mean_delta_m": _mean_list("delta_m"),
        "mean_delta_g": _mean_list("delta_g"),
        "mean_delta_v": _mean_list("delta_v"),
        "last": _jsonable(rows[-1]),
    }


def init_frozen_memory(model) -> Optional[str]:
    if getattr(model, "loop_memory", None) is None:
        return None
    # BagelBackbone.load() owns the deterministic boundary initialization so
    # zero-shot, SFT, rollout, and replay all start from the same m0.
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
        mae = meta.get("pixel_mae_vs_Z0", {}).get(arm["id"])
        mae_txt = "" if mae is None else f" mae_Z0={mae:.2f}"
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
        prompt = str(meta.get("instruction", meta.get("prompt", "")))
        rel = prompt_dir.name
        cells = [
            f"<td class='id'>{html.escape(rel)}</td>"
            f"<td class='prompt'>{html.escape(prompt)}</td>"
        ]
        mae_map = meta.get("pixel_mae_vs_Z0", {})
        for arm in arms:
            name = f"{arm['slug']}.png"
            path = prompt_dir / name
            if not path.is_file():
                cells.append("<td></td>")
                continue
            mae = mae_map.get(arm["id"])
            mae_txt = "" if mae is None else f"<div class='mae'>mae_Z0={mae:.2f}</div>"
            cells.append(
                f"<td><img src='{html.escape(rel + '/' + name)}'>"
                f"<div class='cap'>{html.escape(arm['id'])}</div>{mae_txt}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    page = f"""<!doctype html><meta charset='utf-8'>
<title>BAGEL Read–Write loop Phase-0.5 controls</title>
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
<h1>Read–Write loop zero-shot</h1>
<p>Official <code>inferencer(image=I, text=e, init_noise=ε, **EDIT_hyper)</code>,
no FlowEdit / SDE / TaylorSeer.</p>
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
    case: Dict[str, Any],
    prompt_index: int,
    arms: Sequence[Dict[str, Any]],
    source_image,
) -> None:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    model = inferencer.model
    source_image, image_shape = prepare_source_image(inferencer, source_image)
    prompt = str(case["instruction"])
    old_prompt = str(case["source_prompt"])
    noise_seed = stable_noise_seed(int(args.seed), f"{case['id']}\n{prompt}")
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
            f"layers=[{arm['start_layer']},{arm['end_layer']}) "
            f"old_prompt={not arm['remove_old_prompt']} "
            f"round0_write={arm['round0_memory_write_enabled']}",
            flush=True,
        )
        image = official_edit(
            inferencer,
            source_image,
            prompt,
            init_noise.clone(),
            image_shape,
            old_prompt=old_prompt,
            remove_old_prompt=bool(arm["remove_old_prompt"]),
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
                "remove_old_prompt": arm["remove_old_prompt"],
                "round0_memory_write_enabled": arm[
                    "round0_memory_write_enabled"
                ],
                "num_read_rounds": 0
                if arm["round0_memory_write_enabled"]
                else 1,
                "num_write_rounds": arm["R"]
                if arm["round0_memory_write_enabled"]
                else arm["R"] - 1,
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
    if "Z0" in images:
        for arm in arms:
            if arm["id"] == "Z0":
                continue
            if arm["id"] in images:
                mae[arm["id"]] = pixel_mae(images[arm["id"]], images["Z0"])

    meta = {
        "schema": "bagel_loop_paired_edit_v3",
        "prompt_index": int(prompt_index),
        "id": str(case["id"]),
        "prompt": prompt,
        "instruction": prompt,
        "source_image": str(case["source_image"]),
        "source_prompt": old_prompt,
        "target": str(case.get("target", "")),
        "preserve": list(case.get("preserve", [])),
        "old_prompt": old_prompt,
        "seed": int(args.seed),
        "noise_seed": int(noise_seed),
        "noise_sha256": noise_hash,
        "image_shape": list(image_shape),
        "hyper": NOTEBOOK_EDIT_HYPER,
        "forbidden": {
            "flowedit": False,
            "sde": False,
            "taylorseer": False,
            "think": False,
        },
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
    if arms and max(int(arm["K"]) for arm in arms) > int(args.num_loop_tokens):
        raise ValueError(
            f"selected arms require K={max(int(arm['K']) for arm in arms)}, "
            f"but --num-loop-tokens={int(args.num_loop_tokens)}"
        )
    if args.merge_only:
        merge_gallery(output_dir, arms)
        return

    edit_file = str(args.edit_file or "").strip()
    if edit_file:
        cases = load_edit_cases(edit_file, int(args.max_prompts))
    else:
        source_path = str(args.source_image or "").strip()
        if not source_path:
            raise ValueError("--source-image is required when --edit-file is not set")
        source_prompt = str(args.old_prompt or args.source_prompt or "").strip()
        if any(not bool(arm["remove_old_prompt"]) for arm in arms) and not source_prompt:
            raise ValueError(
                "arms that keep the old prompt require --old-prompt or --source-prompt"
            )
        prompts = load_prompts(args.prompt_file, int(args.max_prompts))
        cases = [
            {
                "id": f"legacy-{index:03d}",
                "source_image": str(Path(source_path).expanduser().resolve()),
                "source_prompt": source_prompt,
                "instruction": prompt,
                "target": prompt,
                "preserve": [],
            }
            for index, prompt in enumerate(prompts)
        ]
    assigned = shard_indices(len(cases), int(args.shard_id), int(args.num_shards))
    print(
        f"[run] cases={len(cases)} shard={args.shard_id}/{args.num_shards} "
        f"assigned={assigned} arms={[arm['id'] for arm in arms]}",
        flush=True,
    )
    if not assigned:
        print("[run] nothing assigned to this shard", flush=True)
        return

    missing_sources = [
        f"{cases[index]['id']}: {cases[index]['source_image']}"
        for index in assigned
        if not Path(str(cases[index]["source_image"])).is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "missing paired source images before model load:\n"
            + "\n".join(missing_sources)
        )

    print("[model] loading frozen BAGEL with loop_memory K="
          f"{int(args.num_loop_tokens)}", flush=True)
    backbone, inferencer = load_native_bagel(args)
    m0_hash = init_frozen_memory(backbone.bagel)
    print(f"[model] m0_sha256={m0_hash}", flush=True)

    from PIL import Image

    for index in assigned:
        tag = f"p{index:03d}"
        case = cases[index]
        source_path = Path(str(case["source_image"]))
        if not source_path.is_file():
            raise FileNotFoundError(
                f"case {case['id']}: source image does not exist: {source_path}"
            )
        source_image = Image.open(source_path).convert("RGB")
        print(f"[{tag}] {case['id']}: {case['instruction']!r}", flush=True)
        run_prompt(
            args,
            inferencer,
            output_dir / tag,
            case,
            index,
            arms,
            source_image,
        )
        print(f"[{tag}] done", flush=True)
    merge_gallery(output_dir, arms)
    print(f"[done] shard={args.shard_id} gallery={output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
