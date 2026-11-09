"""Targeted prompts whose failure mode is known before generation.

The existing request builder only asks for "one change this image does not
show", and add-a-subject is the easiest such change to state and to verify. The
result is a training pool that is 98% additions, so the model is never asked to
undo anything. These prompts invert that: each one states a constraint that a
text-to-image model actually breaks, in four families that need the existing
content reorganised rather than extended.

Families
--------
``over``    too many instances to be plausible (count N >= 7); the usual failure
            is drawing fewer, so the repair is "add".
``under``   task text pinned to a count the model undershoots ("exactly N").
``space``   an explicit left/right or above/below relation; the usual failure
            is swapping it, so the repair is "swap".
``attr``    two same-kind subjects with crossing attributes; the usual failure
            is binding both attributes to one subject.

Usage::

    python scripts/build_targeted_prompts.py --per-family 1500 --out prompts.jsonl
"""

import argparse
import itertools
import json
import random

COUNTS = [6, 7, 8, 9, 10, 12]

OBJECTS = {
    "count": ["red apples", "green croissants", "blue coffee mugs", "yellow ducks",
              "wooden pencils", "white sheep", "orange balloons", "purple candles",
              "silver spoons", "brown ceramic cups", "pink peonies", "green watermelons"],
    "scene": ["on a wooden table", "on a white kitchen counter", "in a wicker basket",
              "on a bookshelf", "on a stone terrace", "on a picnic blanket",
              "inside a glass display case", "on a marble counter"],
    "light": ["in soft daylight", "in warm afternoon light", "in cool studio lighting",
              "under overcast sky", "in golden hour light"],
}

SPATIAL = [
    ("a red bicycle", "a parked motorcycle", "left", "right"),
    ("a large tree", "a wooden bench", "left", "right"),
    ("a black cat", "a white dog", "left", "right"),
    ("a green plant pot", "a blue watering can", "left", "right"),
    ("a hanging lantern", "a framed picture", "above", "below"),
    ("a bird", "a flower vase", "above", "below"),
]

ATTR_SUBJECTS = [
    ("a rabbit", "a fox", "red", "blue"),
    ("a tall glass", "a short mug", "blue", "green"),
    ("a small ladder", "a large ladder", "red", "yellow"),
]
ATTR_PROPS = ["a wool scarf", "a knitted hat", "a leather collar"]


def _count_prompts(count, rng, limit):
    """Exactly-`count` statements; `count` is large enough that models under-draw."""
    rows = []
    for number in range(count, count + int(limit)):
        n = COUNTS[number % len(COUNTS)]
        obj = rng.choice(OBJECTS["count"])
        scene = rng.choice(OBJECTS["scene"])
        light = rng.choice(OBJECTS["light"])
        uid = f"tgt_count_{number:05d}"
        rows.append(dict(
            uid=uid, family="count", target_count=n,
            raw_prompt=f"A photo of exactly {n} {obj} {scene}, {light}.",
        ))
    return rows


def _over_prompts(start, limit, rng):
    """Deliberately crowded scenes: models habitually render fewer than asked."""
    rows = []
    for offset in range(int(limit)):
        number = start + offset
        n = COUNTS[(number + 2) % len(COUNTS)]
        obj = rng.choice(OBJECTS["count"])
        scene = rng.choice(OBJECTS["scene"])
        uid = f"tgt_over_{number:05d}"
        rows.append(dict(
            uid=uid, family="over", target_count=n,
            raw_prompt=(f"A close-up photo of {n} {obj} arranged tightly together "
                        f"{scene}, every one fully visible, {rng.choice(OBJECTS['light'])}."),
        ))
    return rows


def _under_prompts(start, limit, rng):
    """Scenes where a small count must be exact, so over-drawing also fails."""
    rows = []
    for offset in range(int(limit)):
        number = start + offset
        n = [2, 3, 4, 5][number % 4]
        obj = rng.choice(OBJECTS["count"])
        uid = f"tgt_under_{number:05d}"
        rows.append(dict(
            uid=uid, family="under", target_count=n,
            raw_prompt=(f"A photo containing exactly {n} {obj}, no more and no fewer, "
                        f"{rng.choice(OBJECTS['scene'])}, {rng.choice(OBJECTS['light'])}."),
        ))
    return rows


def _space_prompts(start, limit, rng):
    rows = []
    combos = list(itertools.product(SPATIAL, range(2)))
    for offset in range(int(limit)):
        number = start + offset
        (left, right, a, b), flip = combos[number % len(combos)]
        near, far = (left, right) if flip == 0 else (right, left)
        uid = f"tgt_space_{number:05d}"
        rows.append(dict(
            uid=uid, family="space",
            raw_prompt=(f"A photo where {near} is clearly on the {a} and {far} is on the "
                        f"{b}, both visible at the same distance from the camera."),
        ))
    return rows


def _attr_prompts(start, limit, rng):
    rows = []
    for offset in range(int(limit)):
        number = start + offset
        (s1, s2, c1, c2) = ATTR_SUBJECTS[number % len(ATTR_SUBJECTS)]
        prop = ATTR_PROPS[(number // len(ATTR_SUBJECTS)) % len(ATTR_PROPS)]
        if number % 2 == 0:
            text = (f"A photo of {s1} wearing {prop} coloured {c1} and {s2} wearing the "
                    f"same kind of {prop} coloured {c2}.")
        else:
            text = (f"A photo of {s1} in {c1} and {s2} in {c2}, standing side by side, "
                    f"only one of each present.")
        uid = f"tgt_attr_{number:05d}"
        rows.append(dict(uid=uid, family="attr", raw_prompt=text))
    return rows


BUILDERS = {
    "over": lambda start, limit, rng: _over_prompts(start, limit, rng),
    "count": lambda start, limit, rng: _count_prompts(start, limit, rng),
    "under": lambda start, limit, rng: _under_prompts(start, limit, rng),
    "space": lambda start, limit, rng: _space_prompts(start, limit, rng),
    "attr": lambda start, limit, rng: _attr_prompts(start, limit, rng),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default="over,under,space,attr")
    parser.add_argument("--per-family", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows, offset = [], 0
    for family in args.families.split(","):
        family = family.strip()
        if family not in BUILDERS:
            raise SystemExit("unknown family: " + family)
        produced = BUILDERS[family](offset, args.per_family, rng)
        offset += len(produced)
        rows.extend(produced)
    for row in rows:
        row["split"] = "train"
        row["seed"] = abs(hash(row["uid"])) % (2**31)
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for row in rows:
        counts[row["family"]] = counts.get(row["family"], 0) + 1
    print(json.dumps(dict(written=len(rows), by_family=counts, out=args.out), indent=1))


if __name__ == "__main__":
    main()
