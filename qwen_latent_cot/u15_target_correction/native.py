"""Explicit native T2I and IT2I baselines, separate from the state editor."""

import json
from pathlib import Path

import torch
from PIL import Image

from .data import read_rows
from .images import save_image


@torch.no_grad()
def run_native(rt, cfg, mode, prompts, output, limit, device):
    if mode not in ("t2i", "it2i"):
        raise ValueError("Native mode must be t2i or it2i")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    base = Path(prompts).resolve().parent
    rows = read_rows(prompts)[:limit]
    for index, row in enumerate(rows):
        raw = row.get("raw_prompt") or row.get("prompt")
        if not raw:
            raise ValueError(f"Row {index} is missing a prompt")
        seed = int(row.get("seed", cfg["seed"] + index))
        common = dict(
            cfg_scale=float(cfg["cfg_scale"]),
            cfg_norm=cfg["cfg_norm"],
            image_size=(rt.width, rt.height),
            num_steps=int(cfg["num_steps"]),
            timestep_shift=float(cfg["timestep_shift"]),
            enable_timestep_shift=True,
            cfg_interval=tuple(float(t) for t in cfg["cfg_interval"]),
            t_eps=float(cfg["t_eps"]),
            think_mode=False,
            seed=seed,
        )
        if mode == "t2i":
            image = rt.model.t2i_generate(rt.tokenizer, raw, **common)
        else:
            source = row.get("source_image")
            if not source:
                raise ValueError(f"Row {index} needs source_image for native IT2I")
            source = Path(source)
            if not source.is_absolute():
                source = base / source
            with Image.open(source) as opened:
                image = rt.model.it2i_generate(
                    rt.tokenizer,
                    raw,
                    [opened.convert("RGB")],
                    img_cfg_scale=float(cfg["image_cfg_scale"]),
                    **common,
                )
        if image.ndim == 3:
            image = image.unsqueeze(0)
        save_image(out / f"{index:05d}.png", image)
        (out / f"{index:05d}.json").write_text(
            json.dumps(
                dict(
                    uid=row.get("uid", f"{index:05d}"),
                    mode=mode,
                    raw_prompt=raw,
                    seed=seed,
                    native_model_id=str(rt.identity.get("model")),
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    (out / "complete.json").write_text(json.dumps(dict(samples=len(rows), mode=mode)) + "\n")
