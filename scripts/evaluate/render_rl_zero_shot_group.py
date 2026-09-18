#!/usr/bin/env python3
"""Render paired BAGEL/RL candidates with semantic, quality, and RL scores."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BG = "#F6F3EE"
INK = "#242726"
MUTED = "#707674"
BASE = "#8A9199"
LOOP = "#587568"
BEST = "#3F745E"
WORST = "#A45F5F"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _local_image(path_string: str, root: Path) -> Path:
    path = Path(path_string)
    return path if path.is_file() else root / path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--semantic-scores", required=True)
    parser.add_argument("--quality-scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-output", required=True)
    parser.add_argument("--quality-noise-level", default="0.05")
    parser.add_argument("--quality-tolerance", type=float, default=0.0)
    parser.add_argument("--quality-penalty-weight", type=float, default=1.0)
    args = parser.parse_args()

    report_path = Path(args.report)
    root = report_path.parent
    report = json.loads(report_path.read_text(encoding="utf-8"))
    semantic = json.loads(Path(args.semantic_scores).read_text(encoding="utf-8"))
    quality = json.loads(Path(args.quality_scores).read_text(encoding="utf-8"))
    semantic_by_key = {
        (int(row["seed"]), row["depth"]): row for row in semantic["candidates"]
    }
    quality_rows = quality["prompts"][0]["scores"][str(args.quality_noise_level)]
    quality_by_key = {
        (int(row["seed"]), row["depth"]): row for row in quality_rows
    }

    rows = []
    for run in report["runs"]:
        seed = int(run["seed"])
        base_sem = semantic_by_key[(seed, "depth1")]["semantic_log_gm"]
        loop_sem = semantic_by_key[(seed, "depth2")]["semantic_log_gm"]
        base_quality = quality_by_key[(seed, "depth1")]["score_mean"]
        loop_quality = quality_by_key[(seed, "depth2")]["score_mean"]
        semantic_delta = loop_sem - base_sem
        quality_delta = loop_quality - base_quality
        quality_penalty = float(args.quality_penalty_weight) * max(
            0.0, -float(args.quality_tolerance) - quality_delta
        )
        rows.append(
            {
                "seed": seed,
                "base_image": str(_local_image(run["images"]["depth1"], root)),
                "loop_image": str(_local_image(run["images"]["depth2"], root)),
                "base_semantic_log_gm": base_sem,
                "loop_semantic_log_gm": loop_sem,
                "base_semantic_gm_percent": semantic_by_key[(seed, "depth1")][
                    "semantic_gm_percent"
                ],
                "loop_semantic_gm_percent": semantic_by_key[(seed, "depth2")][
                    "semantic_gm_percent"
                ],
                "base_quality": base_quality,
                "loop_quality": loop_quality,
                "semantic_delta": semantic_delta,
                "quality_delta": quality_delta,
                "quality_penalty": quality_penalty,
                "objective": semantic_delta - quality_penalty,
            }
        )
    ranked = sorted(rows, key=lambda row: row["objective"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    by_seed = {row["seed"]: row for row in ranked}
    rows = [by_seed[int(run["seed"])] for run in report["runs"]]

    selection = {
        "schema": "bagel_rl_zero_shot_selection_v1",
        "prompt": report["prompt"],
        "adapter": report["adapter"],
        "quality_noise_level": float(args.quality_noise_level),
        "quality_tolerance": float(args.quality_tolerance),
        "quality_penalty_weight": float(args.quality_penalty_weight),
        "selection_rule": "semantic_delta - quality_penalty",
        "best_seed": ranked[0]["seed"],
        "worst_seed": ranked[-1]["seed"],
        "candidates": ranked,
    }
    Path(args.selection_output).write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    columns = len(rows)
    tile = 512
    gutter = 22
    margin = 36
    header_h = 156
    caption_h = 132
    footer_h = 82
    width = margin * 2 + columns * tile + (columns - 1) * gutter
    height = header_h + 2 * (tile + caption_h) + gutter + footer_h
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    title_font = _font(31, bold=True)
    prompt_font = _font(24)
    label_font = _font(22, bold=True)
    score_font = _font(19)
    note_font = _font(17)

    draw.text((margin, 24), "BAGEL RL zero-shot candidate group", font=title_font, fill=INK)
    wrapped = textwrap.wrap(f'Prompt: {report["prompt"]}', width=110)
    for index, line in enumerate(wrapped[:2]):
        draw.text((margin, 68 + index * 30), line, font=prompt_font, fill=INK)

    row_specs = (("depth1", "BASE / depth 1"), ("depth2", "RL LOOP / depth 2"))
    for row_index, (depth, row_label) in enumerate(row_specs):
        y = header_h + row_index * (tile + caption_h + gutter)
        for column, record in enumerate(rows):
            x = margin + column * (tile + gutter)
            image_key = "base_image" if depth == "depth1" else "loop_image"
            candidate = Image.open(record[image_key]).convert("RGB").resize((tile, tile))
            canvas.paste(candidate, (x, y))
            border = BASE
            badge = ""
            if depth == "depth2" and record["rank"] == 1:
                border, badge = BEST, "BEST"
            elif depth == "depth2" and record["rank"] == columns:
                border, badge = WORST, "WORST"
            draw.rounded_rectangle(
                (x - 4, y - 4, x + tile + 3, y + tile + 3),
                radius=8,
                outline=border,
                width=6 if badge else 3,
            )
            row_badge_w = 154 if depth == "depth2" else 96
            row_badge_label = "RL LOOP" if depth == "depth2" else "BASE"
            draw.rounded_rectangle(
                (x + tile - row_badge_w - 12, y + 12, x + tile - 12, y + 50),
                radius=8,
                fill=LOOP if depth == "depth2" else BASE,
            )
            draw.text(
                (x + tile - row_badge_w, y + 19),
                row_badge_label,
                font=label_font,
                fill="white",
            )
            if badge:
                box_w = 94
                draw.rounded_rectangle(
                    (x + 12, y + 12, x + 12 + box_w, y + 50),
                    radius=8,
                    fill=border,
                )
                draw.text((x + 23, y + 19), badge, font=label_font, fill="white")
            caption_y = y + tile + 14
            sem_key = "base_semantic_gm_percent" if depth == "depth1" else "loop_semantic_gm_percent"
            quality_key = "base_quality" if depth == "depth1" else "loop_quality"
            draw.text(
                (x, caption_y),
                f'Seed {record["seed"]}  |  rank {record["rank"] if depth == "depth2" else "-"}',
                font=label_font,
                fill=INK,
            )
            draw.text(
                (x, caption_y + 32),
                f'Semantic GM {record[sem_key]:.2f}%   Quality {record[quality_key]:.3f}',
                font=score_font,
                fill=INK,
            )
            if depth == "depth2":
                draw.text(
                    (x, caption_y + 62),
                    f'dSem {record["semantic_delta"]:+.3f}   dQual {record["quality_delta"]:+.3f}',
                    font=score_font,
                    fill=MUTED,
                )
                draw.text(
                    (x, caption_y + 91),
                    f'RL objective {record["objective"]:+.3f}',
                    font=label_font,
                    fill=border,
                )

    footer_y = height - footer_h + 8
    draw.line((margin, footer_y, width - margin, footer_y), fill="#D5D0C8", width=2)
    draw.text(
        (margin, footer_y + 16),
        "Higher is better. RL objective = semantic delta - one-sided quality penalty; scores compare each loop image with its same-seed base.",
        font=note_font,
        fill=MUTED,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    print(json.dumps(selection, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
