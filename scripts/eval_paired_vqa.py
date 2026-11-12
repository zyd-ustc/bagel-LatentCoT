"""Score paired sample outputs with independent VQA and optional count cross-check."""

import argparse
import base64
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from qwen_latent_cot.u15_target_correction.judge import _object

NUMBERS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve".split(),
    range(13),
))


def normalize(answer):
    value = str(answer).strip().lower().rstrip(".")
    return str(NUMBERS.get(value, value))


def is_count(question, expected):
    return expected.isdecimal() or question.casefold().startswith("how many ")


def ask(client, model, path, questions):
    encoded = base64.b64encode(path.read_bytes()).decode()
    response = client.chat.completions.create(
        model=model,
        messages=[dict(role="user", content=[
            dict(type="image_url", image_url=dict(url=f"data:image/png;base64,{encoded}")),
            dict(type="text", text=(
                'Answer each question using only the image. Return {"answers": [...]} in order. '
                'Use exact integers for counts, and Yes or No for binary questions. '
                'Use "unknown" when an answer cannot be determined.\n'
                + json.dumps(questions, ensure_ascii=False)
            )),
        ])],
        temperature=0,
        response_format={"type": "json_object"},
    )
    answers = _object(response.choices[0].message.content)["answers"]
    if not isinstance(answers, list) or len(answers) != len(questions):
        raise ValueError(f"Wrong number of answers for {path}")
    return [normalize(answer) for answer in answers]


def pixels(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255


def high_frequency(image):
    gray = image.mean(axis=-1)
    laplacian = (
        4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1]
        - gray[1:-1, :-2] - gray[1:-1, 2:]
    )
    return float(np.abs(laplacian).mean())


def score(client, model, counter, counter_model, directory, prefix, record, score_preview):
    qa = record["vqa_list"]
    questions = [question for question, _ in qa]
    expected = [normalize(answer) for _, answer in qa]
    counts = [i for i, q in enumerate(questions) if is_count(q, expected[i])]
    paths = {arm: directory / f"{prefix}_{arm}.png" for arm in ("baseline", "result")}
    for arm in ("post_edit", "source"):
        path = directory / f"{prefix}_{arm}.png"
        if path.is_file():
            paths[arm] = path
    if score_preview:
        paths["preview"] = directory / f"{prefix}_preview.png"
    answers = {arm: ask(client, model, path, questions) for arm, path in paths.items()}
    checks = (
        {arm: ask(counter, counter_model, path, [questions[i] for i in counts])
         for arm, path in paths.items()}
        if counter is not None and counts else {}
    )
    nfe = record["nfe"]
    shared_nfe = nfe["parent_trajectory"]
    image_calls = {
        "baseline": shared_nfe + record["baseline_nfe"],
        "result": shared_nfe + sum(nfe[key] for key in ("preview", "baseline", "state_edit", "tail")),
    }
    metrics = {}
    base_image = pixels(paths.get("source", paths["baseline"]))
    base_hf = high_frequency(base_image)
    for arm, values in answers.items():
        path = paths[arm]
        image = pixels(path)
        ratio = high_frequency(image) / base_hf if base_hf else None
        metrics[arm] = dict(
            answers=values,
            correct=[a == b for a, b in zip(values, expected)],
            count_crosscheck=checks.get(arm),
            hf_ratio=ratio,
            image_mae=float(np.abs(image - base_image).mean()),
            image_nfe=image_calls.get(arm),
        )
    skills = record.get("skills", [])
    atomic = []
    for i, (question, target) in enumerate(zip(questions, expected)):
        base = answers["baseline"][i]
        direction = None
        if i in counts and base.isdecimal() and target.isdecimal() and base != target:
            direction = "count_reduce" if int(base) > int(target) else "count_increase"
        atomic.append(dict(question=question, expected=target,
                           skill=skills[i] if i < len(skills) else ("count" if i in counts else "other"),
                           direction=direction))
    return dict(uid=record["uid"], seed=record["seed"], route=record["route"],
                atomic=atomic, arms=metrics, count_indices=counts,
                nfe=record["nfe"], baseline_nfe=record["baseline_nfe"],
                post_edit_steps=record.get("post_edit_steps"))


def summarize(rows):
    by_skill = defaultdict(lambda: dict(atoms=0, baseline_correct=0, final_correct=0,
                                        repaired=0, broken=0))
    agreements = []
    for row in rows:
        before = row["arms"]["baseline"]["correct"]
        after = row["arms"]["result"]["correct"]
        for atom, base, result in zip(row["atomic"], before, after):
            for group in filter(None, ("all", atom["skill"], atom["direction"])):
                values = by_skill[group]
                values["atoms"] += 1
                values["baseline_correct"] += int(base)
                values["final_correct"] += int(result)
                values["repaired"] += int(not base and result)
                values["broken"] += int(base and not result)
        for arm in row["arms"].values():
            if arm["count_crosscheck"] is not None:
                agreements.extend(arm["answers"][i] == count
                                  for i, count in zip(row["count_indices"], arm["count_crosscheck"]))
    for values in by_skill.values():
        values["net"] = values["repaired"] - values["broken"]
    arms = {}
    for name in sorted({name for row in rows for name in row["arms"]}):
        group = [row["arms"][name] for row in rows if name in row["arms"]]
        ratios = [arm["hf_ratio"] for arm in group if arm["hf_ratio"] is not None]
        image_nfe = [arm["image_nfe"] for arm in group if arm["image_nfe"] is not None]
        arms[name] = dict(samples=len(group), full_pass=sum(all(arm["correct"]) for arm in group),
                          mean_image_nfe=float(np.mean(image_nfe)) if image_nfe else None,
                          median_hf_ratio=float(np.median(ratios)) if ratios else None,
                          mean_image_mae=float(np.mean([arm["image_mae"] for arm in group])))
    return dict(samples=len(rows), by_skill=dict(by_skill), arms=arms,
                count_agreement=dict(agree=sum(agreements), compared=len(agreements)))


def main():
    from openai import OpenAI

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-env", required=True)
    parser.add_argument("--count-model")
    parser.add_argument("--count-base-url")
    parser.add_argument("--count-key-env")
    parser.add_argument("--score-preview", action="store_true")
    args = parser.parse_args()
    client = OpenAI(base_url=args.base_url, api_key=os.environ[args.key_env])
    counter = (
        OpenAI(base_url=args.count_base_url or args.base_url,
               api_key=os.environ[args.count_key_env or args.key_env])
        if args.count_model else None
    )
    if args.count_model and args.count_model == args.model and (
        not args.count_base_url or args.count_base_url == args.base_url
    ):
        parser.error("Select an independent counting model")
    records = [(path, json.loads(path.read_text())) for path in sorted(args.samples.glob("[0-9]*.json"))]
    if not records or any(not row.get("vqa_list") for _, row in records):
        parser.error("Every sample must carry vqa_list")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    with (args.output / "pairs.jsonl").open("w") as handle:
        for path, record in records:
            row = score(client, args.model, counter, args.count_model,
                        args.samples, path.stem, record, args.score_preview)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    result = dict(summarize(rows), model=args.model, count_model=args.count_model)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
