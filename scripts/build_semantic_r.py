"""Plan, render, and accept semantic R records on cached native parent states."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from qwen_latent_cot.u15_target_correction import CACHE_KIND, CONDITION_PROTOCOL
from qwen_latent_cot.u15_target_correction.edit_request import SCHEMA, parse_edit
from qwen_latent_cot.u15_target_correction.judge import _object


def read_rows(path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())["records"]


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def ask(client, model, instruction, images):
    import base64

    content = [dict(type="text", text=instruction)]
    for label, path in images:
        content.append(dict(type="text", text=label))
        encoded = base64.b64encode(Path(path).read_bytes()).decode()
        content.append(dict(type="image_url", image_url=dict(url=f"data:image/png;base64,{encoded}")))
    response = client.chat.completions.create(
        model=model, temperature=0,
        messages=[dict(role="user", content=content)],
        response_format={"type": "json_object"},
    )
    return _object(response.choices[0].message.content)


def client_for(args):
    from openai import OpenAI

    return OpenAI(base_url=args.base_url, api_key=os.environ[args.key_env])


def plan(args):
    client = client_for(args) if args.role != "keep" else None
    counts = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        for parent in read_rows(args.parents)[args.shard_index::args.shard_count][:args.limit]:
            raw = parent["parent_condition"]
            source = parent["source_image"]
            state = parent["states"][f"{args.time:.6f}"]
            uid = str(parent["uid"])
            digest = hashlib.sha256(uid.encode()).digest()
            task = args.task or ("count_increase", "count_reduce", "spatial", "attribute")[digest[0] % 4]
            row = dict(
                uid=f"{uid}:{args.role}:{task}" if args.role != "keep" else f"{uid}:keep",
                source_kind="native_parent", source_family=parent.get("source_family") or uid,
                prompt_family=parent.get("prompt_family") or raw,
                split=parent.get("split", "train"), seed=parent["seed"],
                condition_protocol=CONDITION_PROTOCOL, source_image=source,
                parent_condition=raw, tail_condition=raw, preserve="",
                phase_grid=parent["phase_grid"],
                states={f"{args.time:.6f}": dict(state, accepted=False)},
            )
            if args.role == "keep":
                row.update(role="R_keep", target_image=source,
                           edit_instruction="Keep the current scene unchanged.")
            else:
                requirements = {
                    "active": "Design a single visible change to the original scene. Return goal, the original description minimally updated to express the change.",
                    "repair": "Find one violation of the original prompt visible in BOTH images. Use none when no such violation exists.",
                    "satisfied": "Find one requirement of the original prompt ALREADY satisfied in BOTH images. Express it as a modification instruction that would leave both images unchanged. Use none if ambiguous.",
                }
                required_task = "count" if args.role == "satisfied" and task.startswith("count_") else task
                response = ask(
                    client, args.model,
                    f"{requirements[args.role]}\nOriginal prompt: {raw}\nRequired task: {required_task}. "
                    "For any count task, use task=count and provide source_count "
                    "as the exact visible count in the original continuation. For satisfied count "
                    "examples target equals source_count. Include preserve, a short description of "
                    "what must remain intact. Use none if the required task cannot be grounded. "
                    + SCHEMA,
                    [("Draft", state["preview_image"]), ("Original continuation", source)],
                )
                try:
                    edit = parse_edit(response)
                except ValueError:
                    counts["invalid_edit"] += 1
                    continue
                expected_task = "count" if task.startswith("count_") else task
                if edit is None or edit["task"] != expected_task:
                    counts["no_matching_edit"] += 1
                    continue
                direction = source_count = None
                if expected_task == "count":
                    source_count = response.get("source_count")
                    target = edit["target"]
                    if type(source_count) is not int or source_count < 0:
                        counts["invalid_source_count"] += 1
                        continue
                    if target != source_count:
                        direction = "count_increase" if target > source_count else "count_reduce"
                    valid_direction = direction is None if args.role == "satisfied" else direction == task
                    if not valid_direction:
                        counts["wrong_count_direction"] += 1
                        continue
                if args.role == "active":
                    goal = response.get("goal")
                    if not isinstance(goal, str) or not goal.strip() or goal.strip() == raw.strip():
                        counts["invalid_goal"] += 1
                        continue
                    row["tail_condition"] = goal.strip()
                row.update(role=f"R_{args.role}", semantic_task=expected_task,
                           count_direction=direction, source_count=source_count, atomic_edit=edit,
                           edit_instruction=edit["instruction"],
                           preserve=edit["preserve"])
                if args.role == "satisfied":
                    row["target_image"] = source
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[row["role"]] += 1
    print(json.dumps(counts))


def render(args):
    from PIL import Image
    from qwen_latent_cot.u15_target_correction.accelerator import set_device
    from qwen_latent_cot.u15_target_correction.config import load_config
    from qwen_latent_cot.u15_target_correction.field import phase_grid, protocol, time_key
    from qwen_latent_cot.u15_target_correction.images import save_image
    from qwen_latent_cot.u15_target_correction.loading import load_native
    from qwen_latent_cot.u15_target_correction.native import it2i

    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, set_device(args.device), distributed=False)
    grid = phase_grid(rt, cfg)
    key = time_key(cfg["read_time"])
    directory = args.output.resolve()
    (directory / "progress").mkdir(parents=True, exist_ok=True)
    (directory / "targets").mkdir(exist_ok=True)
    records = []
    for row in read_rows(args.requests)[args.shard_index::args.shard_count]:
        if row["phase_grid"] != grid or key not in row["states"]:
            raise ValueError(f"Parent trajectory does not match the training grid/time: {row['uid']}")
        stem = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:24]
        marker = directory / "progress" / f"{stem}.json"
        if marker.exists():
            records.append(json.loads(marker.read_text()))
            continue
        if row["role"] in ("R_active", "R_repair"):
            prompt = (row["edit_instruction"] + "\nTarget: " + row["tail_condition"]
                      + "\nPreserve: " + row["preserve"])
            with Image.open(row["source_image"]) as source:
                image = it2i(rt, cfg, source.convert("RGB"), prompt, row["seed"])
            target = directory / "targets" / f"{stem}.png"
            save_image(target, image)
            row["target_image"] = str(target)
        write_json(marker, row)
        records.append(row)
    name = f"manifest.shard_{args.shard_index:03d}.json"
    write_json(directory / name, dict(kind=CACHE_KIND, protocol=protocol(rt, cfg), records=records))


def normalized_regions(value):
    if not isinstance(value, list) or not value:
        return None
    for box in value:
        if (not isinstance(box, list) or len(box) != 4
                or any(type(x) not in (float, int) for x in box)):
            return None
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            return None
    return value


def accepted_row(row, regions, verdict=None):
    states = {key: dict(value, accepted=True) for key, value in row["states"].items()}
    return dict(row, states=states, edit_regions=regions, acceptance=verdict)


def accept(args):
    client = client_for(args)
    manifest = json.loads(args.rendered.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    approved, rejected = [], []
    for row in manifest["records"]:
        role = row["role"]
        if role == "R_keep":
            approved.append(accepted_row(row, []))
            continue
        if role == "R_satisfied":
            state = next(iter(row["states"].values()))
            instruction = (
                f"Does the requested final state already hold in both images? Edit: {row['edit_instruction']} "
                'Return {"already_satisfied": true} only if both clearly satisfy it.'
            )
            verdict = ask(client, args.model, instruction,
                          [("Draft", state["preview_image"]), ("Original continuation", row["source_image"])])
            if verdict.get("already_satisfied") is True:
                approved.append(accepted_row(row, [], verdict))
            else:
                rejected.append(dict(uid=row["uid"], verdict=verdict))
            continue
        instruction = (
            f"Evaluate one edit. Instruction: {row['edit_instruction']} "
            f"Target: {row['tail_condition']} Preserve: {row['preserve']}. "
            "Return a JSON object with booleans edit_needed, edit_done, keep_ok, "
            "context_keep_ok, quality_ok and edit_regions. edit_needed means A fails "
            "the exact requested edit target; edit_done means B satisfies it. "
            "For counts report integer source_count and target_count measured in A and B. "
            "For spatial tasks verify the named relation. "
            "edit_regions is an array of normalized [x0,y0,x1,y1] boxes covering only "
            "the required edit, including old and new positions and affected occlusions. "
            "keep_ok checks unrelated content outside all boxes. context_keep_ok "
            "checks the surrounding scene and the gaps between boxes inside their enclosing "
            "rectangle, including the immediate context around the boxes. "
            "Reject visible redraw or drift in this context. quality_ok rejects new "
            "ghosting, fusion, tearing, malformed or duplicated objects."
        )
        verdict = ask(client, args.model, instruction,
                      [("A", row["source_image"]), ("B", row["target_image"])])
        regions = normalized_regions(verdict.get("edit_regions"))
        gates = ("edit_needed", "edit_done", "keep_ok", "context_keep_ok", "quality_ok")
        counts_match = True
        if row["semantic_task"] == "count":
            before, after = verdict.get("source_count"), verdict.get("target_count")
            counts_match = (
                type(before) is int and type(after) is int
                and before == row["source_count"] and after == row["atomic_edit"]["target"]
            )
        if regions and counts_match and all(verdict.get(gate) is True for gate in gates):
            approved.append(accepted_row(row, regions, verdict))
        else:
            rejected.append(dict(uid=row["uid"], verdict=verdict))
    write_json(args.output, dict(manifest, records=approved))
    write_json(args.output.with_suffix(".rejected.json"), rejected)
    print(json.dumps(dict(approved=len(approved), rejected=len(rejected))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    planning = stages.add_parser("plan")
    planning.add_argument("--parents", required=True, type=Path)
    planning.add_argument("--role", required=True, choices=("active", "repair", "satisfied", "keep"))
    planning.add_argument("--task", choices=("count_increase", "count_reduce", "spatial", "attribute"))
    planning.add_argument("--time", type=float, default=0.36)
    planning.add_argument("--limit", type=int)
    rendering = stages.add_parser("render")
    rendering.add_argument("--config", required=True)
    rendering.add_argument("--source", required=True)
    rendering.add_argument("--requests", required=True, type=Path)
    rendering.add_argument("--device", default="npu:0")
    accepting = stages.add_parser("accept")
    accepting.add_argument("--rendered", required=True, type=Path)
    for stage in (planning, accepting):
        stage.add_argument("--base-url", required=True)
        stage.add_argument("--key-env", required=True)
    for stage in (planning, rendering):
        stage.add_argument("--shard-index", type=int, default=0)
        stage.add_argument("--shard-count", type=int, default=1)
    for stage in (planning, rendering, accepting):
        stage.add_argument("--model", required=True)
        stage.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.stage in ("plan", "render") and not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be within shard-count")
    {"plan": plan, "render": render, "accept": accept}[args.stage](args)


if __name__ == "__main__":
    main()
