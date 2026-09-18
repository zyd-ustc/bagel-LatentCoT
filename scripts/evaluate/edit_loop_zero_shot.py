#!/usr/bin/env python3
"""Zero-shot Markov edit loop on frozen BAGEL — official inference API only.

State (Markov, no history needed)::

    s_r = (I_r, a_r)
    a_r     = UND(I_r, target P)                      -> edit instruction
    I_{r+1} = GEN(I_r, a_r)                           -> P is NEVER visible to GEN

Both calls go through the official ``InterleaveInferencer.__call__`` with the
hyperparameters from ``inference.ipynb``:

  understanding : ``understanding_output=True, max_think_token_n=1000, do_sample=False``
  editing       : ``cfg_text_scale=4.0, cfg_img_scale=2.0, cfg_interval=[0.0,1.0],
                   timestep_shift=3.0, num_timesteps=50, cfg_renorm_min=0.0,
                   cfg_renorm_type="text_channel"``

Round 0 is a native T2I with the text prompt; every later round sees only the
previous image plus the instruction. T2I pins ``init_noise`` for reproducibility.
Editing noise is a separate switch:

  same   reuse the T2I ε (reconstruction-lock diagnostic)
  fresh  omit init_noise so GEN samples like the official notebook

Controls:
  none      UND reflection -> parsed INSTRUCTION
  no_text   image only (no instruction)
  fixed     constant appearance instruction
  gold      constant strong semantic instruction (skips UND)
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Optional

from bagel_common import (
    add_native_model_args,
    load_native_bagel,
    make_noise,
    pixel_mae,
    relative_l2,
    stable_noise_seed,
)
from qwen_latent_cot.bagel import accelerator


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

REFLECT_QUERY = """Target: {prompt}

Look at the current image and compare it with the target.
Report the concrete mismatches (object count, position, colour/shape attributes).

Answer in exactly this format:
MISMATCHES:
- <mismatch 1>
INSTRUCTION: <one imperative sentence that fixes the most important mismatch>"""

GOLD_TEXT = (
    "Replace every animal in the image with a bright red sports car. "
    "The edited image must contain zero animals and at least three red cars, "
    "photographed from the same camera angle."
)

INSTRUCTION_MARKER = "INSTRUCTION:"
CONTROLS_WITHOUT_REFLECTION = frozenset({"no_text", "fixed", "gold"})


def parse_instruction(text: str) -> str:
    """Extract the imperative edit instruction; fall back to the whole answer."""

    text = (text or "").strip()
    if INSTRUCTION_MARKER in text:
        tail = text.split(INSTRUCTION_MARKER, 1)[1].strip()
        first_line = tail.split("\n", 1)[0].strip()
        if first_line:
            return first_line
    return text


def select_edit_text(
    control: str,
    reflection: Optional[str],
    *,
    fixed_text: str,
    gold_text: str,
) -> Optional[str]:
    """Pick the GEN-side instruction. ``none`` is the only control that uses UND."""

    if control == "no_text":
        return None
    if control == "fixed":
        return str(fixed_text)
    if control == "gold":
        return str(gold_text)
    if control == "none":
        return parse_instruction(reflection or "")
    raise ValueError(f"unknown control: {control}")


def build_edit_kwargs(
    *,
    image_shape: tuple[int, int],
    edit_text: Optional[str],
    t2i_noise: Any,
    edit_noise: str,
) -> dict[str, Any]:
    """Official editing ``__call__`` kwargs. ``fresh`` omits T2I ``init_noise``."""

    if edit_noise not in {"same", "fresh"}:
        raise ValueError(f"unknown edit_noise: {edit_noise}")
    kwargs: dict[str, Any] = dict(
        image_shapes=image_shape,
        return_latent=True,
        **EDIT_HYPER,
    )
    if edit_text is not None:
        kwargs["text"] = edit_text
    if edit_noise == "same":
        kwargs["init_noise"] = t2i_noise
    return kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_native_model_args(parser)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-prompts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--control", default="none", choices=["none", "no_text", "fixed", "gold"]
    )
    parser.add_argument(
        "--edit-noise",
        default="fresh",
        choices=["same", "fresh"],
        help="same: reuse T2I ε (reconstruction lock). fresh: official notebook sampling.",
    )
    parser.add_argument("--fixed-text", default="Make the image a watercolor painting.")
    parser.add_argument("--gold-text", default=GOLD_TEXT)
    parser.add_argument("--reflect-query", default=REFLECT_QUERY)
    parser.add_argument("--reflection-max-tokens", type=int, default=1000)
    parser.add_argument("--reflection-sample", action="store_true")
    return parser.parse_args()


def load_prompts(args) -> list[str]:
    prompts: list[str] = []
    if args.prompt_file:
        for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                prompts.append(
                    str(json.loads(line)["prompt"]).strip()
                    if line.startswith("{")
                    else line
                )
    elif str(args.prompt).strip():
        prompts.append(str(args.prompt).strip())
    if not prompts:
        raise ValueError("provide --prompt or --prompt-file")
    if int(args.max_prompts) > 0:
        prompts = prompts[: int(args.max_prompts)]
    return prompts


def _caption(round_idx: int, edit_text: Optional[str]) -> str:
    if round_idx == 0:
        return "I_0 (T2I)"
    instr = edit_text if edit_text else "(no text)"
    if len(instr) > 90:
        instr = instr[:87] + "..."
    return f"r{round_idx}: {instr}"


def write_gallery(
    output_dir: Path, rows: list[dict], *, rounds: int, control: str, edit_noise: str
) -> None:
    headers = ["prompt", "I_0 (T2I)"] + [f"round {i}" for i in range(1, rounds + 1)]
    header_html = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    cells = []
    for row in rows:
        tds = "".join(
            f"<td><img src='{html.escape(src)}'><div class='cap'>{html.escape(cap)}</div></td>"
            for src, cap in row["panels"]
        )
        cells.append(
            f"<tr><td class='prompt'>{html.escape(row['prompt'])}</td>{tds}</tr>"
        )
    page = f"""<!doctype html><meta charset='utf-8'>
<title>Zero-shot Markov edit loop</title>
<style>body{{font:13px system-ui;background:#111;color:#eee;margin:20px}}
table{{border-collapse:collapse}} td{{padding:6px;border:1px solid #333;vertical-align:top}}
img{{width:170px;height:170px;object-fit:contain;background:#000}}
.cap{{font-size:11px;color:#aaa;text-align:center;max-width:170px}}
.prompt{{max-width:220px;color:#ffd479}}</style>
<h1>Zero-shot Markov edit loop (frozen BAGEL)</h1>
<p>control={html.escape(control)} edit_noise={html.escape(edit_noise)}</p>
<table><tr>{header_html}</tr>
{"".join(cells)}</table>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_shape = (int(args.image_height), int(args.image_width))
    need_reflection = str(args.control) not in CONTROLS_WITHOUT_REFLECTION

    print(
        f"[run] {len(prompts)} prompts x {int(args.rounds)} rounds, "
        f"control={args.control} edit_noise={args.edit_noise}",
        flush=True,
    )
    print("[model] loading frozen native BAGEL checkpoint", flush=True)
    backbone, inferencer = load_native_bagel(args)
    model = backbone.bagel
    assert model is not None

    rows, records = [], []
    for index, prompt in enumerate(prompts):
        pdir = output_dir / f"p{index:03d}"
        pdir.mkdir(parents=True, exist_ok=True)
        seed = stable_noise_seed(int(args.seed), prompt, schema="bagel-edit-loop-v1")
        t2i_noise = make_noise(model, image_shape, seed).to(
            accelerator.resolve_device(args.device)
        )

        out = inferencer(
            text=prompt,
            image_shapes=image_shape,
            init_noise=t2i_noise,
            return_latent=True,
            **T2I_HYPER,
        )
        image = out["image"]
        latent = inferencer.last_latent
        image0, latent0 = image, latent
        image.save(pdir / "round0.png")
        panels = [(f"p{index:03d}/round0.png", _caption(0, None))]
        rounds = [{"round": 0, "image": "round0.png", "reflection": None, "edit_text": None}]
        print(f"[p{index:03d}] round0 ok", flush=True)

        for r in range(1, int(args.rounds) + 1):
            reflection = None
            if need_reflection:
                query = str(args.reflect_query).format(prompt=prompt)
                reflection = inferencer(
                    image=image,
                    text=query,
                    understanding_output=True,
                    max_think_token_n=int(args.reflection_max_tokens),
                    do_sample=bool(args.reflection_sample),
                )["text"]
                (pdir / f"reflection_r{r}.txt").write_text(
                    reflection + "\n", encoding="utf-8"
                )

            edit_text = select_edit_text(
                str(args.control),
                reflection,
                fixed_text=str(args.fixed_text),
                gold_text=str(args.gold_text),
            )
            (pdir / f"instruction_r{r}.txt").write_text(
                (edit_text or "") + "\n", encoding="utf-8"
            )

            call_kwargs = build_edit_kwargs(
                image_shape=image_shape,
                edit_text=edit_text,
                t2i_noise=t2i_noise,
                edit_noise=str(args.edit_noise),
            )
            out = inferencer(image=image, **call_kwargs)
            new_image = out["image"]
            new_latent = inferencer.last_latent
            new_image.save(pdir / f"round{r}.png")
            panels.append(
                (f"p{index:03d}/round{r}.png", _caption(r, edit_text))
            )

            rounds.append(
                {
                    "round": r,
                    "image": f"round{r}.png",
                    "edit_text": edit_text,
                    "reflection": reflection,
                    "edit_noise": str(args.edit_noise),
                    "pinned_t2i_noise": str(args.edit_noise) == "same",
                    "relative_l2_vs_round0": relative_l2(new_latent, latent0),
                    "pixel_mae_vs_round0": pixel_mae(new_image, image0),
                    "relative_l2_vs_prev": relative_l2(new_latent, latent),
                    "pixel_mae_vs_prev": pixel_mae(new_image, image),
                }
            )
            print(
                f"[p{index:03d}] round{r} mae_vs_round0="
                f"{rounds[-1]['pixel_mae_vs_round0']:.2f} "
                f"mae_vs_prev={rounds[-1]['pixel_mae_vs_prev']:.2f}\n"
                f"    reflection: {reflection}\n"
                f"    instruction: {edit_text}",
                flush=True,
            )
            image, latent = new_image, new_latent

        rows.append({"prompt": prompt, "panels": panels})
        records.append({"prompt": prompt, "noise_seed": int(seed), "rounds": rounds})

    manifest = {
        "schema": "bagel_edit_loop_zero_shot_v2",
        "control": str(args.control),
        "edit_noise": str(args.edit_noise),
        "gold_text": str(args.gold_text),
        "fixed_text": str(args.fixed_text),
        "rounds": int(args.rounds),
        "prompts": prompts,
        "image_shape": list(image_shape),
        "t2i_hyper": T2I_HYPER,
        "edit_hyper": EDIT_HYPER,
        "reflect_query": str(args.reflect_query),
        "records": records,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_gallery(
        output_dir,
        rows,
        rounds=int(args.rounds),
        control=str(args.control),
        edit_noise=str(args.edit_noise),
    )
    print(f"[done] gallery={output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
