"""Linear-pair diagnostic evaluation at a fixed native time."""

import html
import json
import random
import re
import time as clock
from collections import defaultdict
from pathlib import Path

import torch

from . import distributed as dist
from .field import NativeT2I, edit_state, euler, phase_grid, preview_image, segment_times
from .images import save_image
from .loss import build_perceptual, masked_mse, perceptual_distance

TEXT = re.compile(r"\b(text|word|letter|spell|translate|font|caption\s+text)\b", re.I)
ADD = re.compile(r"\b(add|insert|place)\b", re.I)
REMOVE = re.compile(r"\b(remove|delete|erase)\b", re.I)
COUNT = re.compile(r"\b(count|number|exactly|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.I)
SPATIAL = re.compile(
    r"\b(move|position|behind|front|left|right|under|above|top|bottom|background|"
    r"foreground|beside|next to|between)\b",
    re.I,
)
ATTRIBUTE = re.compile(
    r"\b(colou?r|material|style|hair|makeup|age|clothing|wear|red|blue|green|"
    r"yellow|black|white|brown|pink|purple|orange|grey|gray|wooden|metal)\b",
    re.I,
)


def category(edit):
    if TEXT.search(edit):
        return "text"
    if ADD.search(edit):
        return "add"
    if REMOVE.search(edit):
        return "remove"
    if COUNT.search(edit):
        return "count"
    if SPATIAL.search(edit):
        return "spatial"
    if ATTRIBUTE.search(edit):
        return "attribute"
    return "other"


def select_cases(manifest, limit, seed):
    """Deterministic category-balanced selection from the validation split."""
    groups = defaultdict(list)
    for index, row in enumerate(manifest.rows):
        if row["split"] == "val":
            groups[category(row["edit_instruction"])].append(index)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    names = sorted(groups)
    selected = []
    while len(selected) < min(limit, sum(len(v) for v in groups.values())):
        progressed = False
        for name in names:
            if groups[name]:
                selected.append((groups[name].pop(), name))
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


def _html(records, meta):
    def esc(value):
        return html.escape("" if value is None else str(value))

    rows = []
    for record in records:
        metrics = (
            f"位移 MSE {record['displacement_mse']:.5f}"
            f" · LPIPS {record.get('displacement_lpips', float('nan')):.4f}"
            f" · 学生尾段 MSE {record['student_tail_mse']:.5f}"
            f" · 目标态尾段 MSE {record['target_tail_mse']:.5f}"
        )
        figures = []
        for key, label in (
            ("source", "源图"),
            ("target", "编辑目标"),
            ("preview", "预览 P"),
            ("displacement", "位移诊断 x̂_edit"),
            ("student", "学生状态接续"),
            ("target_tail", "目标态接续"),
        ):
            figures.append(
                f"<figure><img loading='lazy' src='{esc(record['case'])}_{key}.png'>"
                f"<figcaption>{label}</figcaption></figure>"
            )
        rows.append(
            f"<section class='case'><div class='head'><b>{esc(record['uid'])}</b>"
            f"<span>{esc(record['task'])} / {esc(record['category'])}</span>"
            f"<span>{esc(record['edit_instruction'])}</span></div>"
            f"<div class='metrics'>{metrics}</div><div class='strip'>{''.join(figures)}</div></section>"
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Linear edit evaluation</title>
<style>
body{{margin:0;background:#0d1117;color:#e6edf3;font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{padding:16px 20px;border-bottom:1px solid #30363d;position:sticky;top:0;background:#0d1117ee;backdrop-filter:blur(8px)}}
h1{{font-size:18px;margin:0 0 6px}} .muted{{color:#8b949e}} main{{padding:16px}}
.case{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px;margin-bottom:14px}}
.head{{display:flex;gap:12px;flex-wrap:wrap;align-items:baseline}} .head b{{color:#58a6ff}}
.head span{{color:#8b949e}} .metrics{{color:#d29922;margin:6px 0 10px;font-size:12px}}
.strip{{display:grid;grid-template-columns:repeat(6,minmax(150px,1fr));gap:8px;overflow-x:auto}}
figure{{margin:0}} img{{width:100%;aspect-ratio:1;object-fit:cover;border:1px solid #30363d;border-radius:6px;background:#000;cursor:zoom-in}}
figcaption{{text-align:center;color:#8b949e;font-size:11px;margin-top:3px}}
</style></head><body><header><h1>Linear-pair 状态编辑诊断</h1>
<div class="muted">checkpoint {esc(meta['checkpoint'])} · {meta['cases']} 例 · t={meta['time']} · {meta['steps']} 步 · seed={meta['seed']} · val split · 不调用诊断器</div></header>
<main>{''.join(rows)}</main><script>document.querySelectorAll('img').forEach(i=>i.onclick=()=>window.open(i.src,'_blank'));</script></body></html>"""


@torch.no_grad()
def evaluate_linear(rt, cfg, editor, manifest, output, *, limit, seed, time, steps, no_lpips=False):
    output = Path(output).resolve()
    if dist.rank() == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    cases = select_cases(manifest, limit, seed)
    grid = phase_grid(rt, cfg)
    perceptual = None if no_lpips else build_perceptual(rt.device)
    records = []
    for position, (manifest_index, category_name) in enumerate(cases):
        if position % dist.world_size() != dist.rank():
            continue
        began = clock.perf_counter()
        sample = manifest.sample(manifest_index, time)
        source, target = sample["source"], sample["target"]
        valid = sample["valid"]
        generator = torch.Generator(device=rt.device).manual_seed(int(seed) + manifest_index)
        noise = torch.randn(source.shape, device=rt.device, generator=generator) * rt.noise_scale()
        anchor = time * source + (1 - time) * noise
        destination = time * target + (1 - time) * noise

        parent = NativeT2I(rt, sample["parent_condition"], cfg)
        before_preview = parent.calls
        preview = preview_image(anchor, time, grid, parent)
        preview_calls = parent.calls - before_preview
        prefixes = editor.prefixes(
            preview, sample["edit_request"], sample["tail_condition"], sample["preserve"]
        )
        before_baseline = editor.baseline_calls
        base = editor.image_clean(anchor, time, prefixes["image"])
        baseline_calls = editor.baseline_calls - before_baseline
        before_state = editor.state_calls
        edited, _ = edit_state(editor, anchor, time, prefixes, steps, base_prediction=base)
        state_calls = editor.state_calls - before_state

        displacement = source + (edited - anchor) / time
        tail = NativeT2I(rt, sample["tail_condition"], cfg)
        tail_times = segment_times(grid, time, 1.0)
        student_tail = euler(edited, tail_times, tail)
        target_tail = euler(destination, tail_times, tail)

        displacement_mse = float(masked_mse(displacement, target, valid))
        displacement_lpips = (
            perceptual_distance(perceptual, displacement, target, valid)
            if perceptual is not None else float("nan")
        )
        student_tail_mse = float(masked_mse(student_tail, target, valid))
        target_tail_mse = float(masked_mse(target_tail, target, valid))
        student_vs_target_tail = float(masked_mse(student_tail, target_tail, valid))
        case = f"case_{position:05d}"
        images = dict(
            source=source,
            target=target,
            preview=preview,
            displacement=displacement,
            student=student_tail,
            target_tail=target_tail,
        )
        for key, image in images.items():
            save_image(output / f"{case}_{key}.png", image)
        record = dict(
            case=case,
            uid=sample["uid"],
            manifest_index=manifest_index,
            task=sample.get("task"),
            category=category_name,
            edit_instruction=sample["edit_request"],
            parent_condition=sample["parent_condition"],
            tail_condition=sample["tail_condition"],
            selection="val",
            noise_seed=int(seed) + manifest_index,
            time=float(time),
            steps=int(steps),
            seconds=clock.perf_counter() - began,
            displacement_mse=displacement_mse,
            displacement_lpips=displacement_lpips,
            student_tail_mse=student_tail_mse,
            target_tail_mse=target_tail_mse,
            student_vs_target_tail_mse=student_vs_target_tail,
            nfe=dict(
                preview=preview_calls,
                prefill=2,
                baseline=baseline_calls,
                state_edit=state_calls,
                tail=tail.calls,
            ),
        )
        (output / f"{case}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        records.append(record)
    dist.barrier()
    if dist.rank() == 0:
        records = [
            json.loads(path.read_text())
            for path in sorted(output.glob("case_*.json"))
        ]
        records.sort(key=lambda row: row["case"])
        summary = dict(
            checkpoint=str(getattr(editor, "checkpoint", "")),
            cases=len(records),
            time=float(time),
            steps=int(steps),
            seed=int(seed),
            mean_displacement_mse=sum(row["displacement_mse"] for row in records) / max(1, len(records)),
            mean_displacement_lpips=sum(row["displacement_lpips"] for row in records) / max(1, len(records)),
            mean_student_tail_mse=sum(row["student_tail_mse"] for row in records) / max(1, len(records)),
            mean_target_tail_mse=sum(row["target_tail_mse"] for row in records) / max(1, len(records)),
            categories={name: sum(row["category"] == name for row in records) for name in sorted({row["category"] for row in records})},
        )
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        (output / "review.html").write_text(_html(records, summary))
        (output / "complete.json").write_text(json.dumps(dict(cases=len(records))) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return records
