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


# Failure shapes that a text-to-image model actually makes on a crowded or
# relational scene. Each one states a requirement the picture can visibly break,
# so a completion-side audit has something concrete to find.
EXTRA_OBJECTS = ["red apples", "green croissants", "blue coffee mugs", "yellow ducks",
                 "wooden pencils", "white sheep", "orange balloons", "purple candles",
                 "silver spoons", "brown ceramic cups", "pink peonies", "green watermelons",
                 "red toy cars", "brass keys", "white eggs", "blue marbles"]
EXTRA_SCENES = ["on a weathered wooden table", "on a white kitchen counter",
                "inside a wicker basket", "on a bookshelf", "on a stone terrace",
                "on a red picnic blanket", "inside a glass display case",
                "on a marble counter", "on a wooden floor", "on a metal tray"]
EXTRA_LEVELS = ["8", "9", "10", "11", "12", "14"]
EXTRA_COLOURS = ["red", "orange", "yellow", "green", "blue", "purple",
                 "pink", "brown", "grey", "black", "white", "teal"]
EXTRA_MATERIALS = ["wooden", "ceramic", "glass", "metal", "woven", "stone"]


def _crowd_prompts(start, limit, rng):
    """Many instances, all entirely inside the frame: the usual failure is under-drawing."""
    rows = []
    for offset in range(int(limit)):
        number = start + offset
        n = EXTRA_LEVELS[number % len(EXTRA_LEVELS)]
        obj = f"{EXTRA_COLOURS[number % 12]} {EXTRA_MATERIALS[(number // 12) % 6]} {rng.choice(EXTRA_OBJECTS)}"
        scene = rng.choice(EXTRA_SCENES)
        uid = f"tgt_crowd_{number:05d}"
        rows.append(dict(uid=uid, family="over", target_count=int(n),
                         raw_prompt=f"An overhead photo of exactly {n} {obj} laid out side by side {scene}, "
                                    f"every object entirely inside the frame."))
    return rows


def _pair_prompts(start, limit, rng):
    """Two same-kind subjects with crossing attributes: the usual failure is swapping them."""
    rows = []
    shapes = [("mug", "plate"), ("chair", "stool"), ("bottle", "jar"), ("hat", "scarf"),
              ("cup", "saucer"), ("box", "crate"), ("vase", "bowl"), ("spoon", "fork")]
    relations = ["on the left", "in the foreground", "near the window", "on the shelf"]
    for offset in range(int(limit)):
        number = start + offset
        colour = EXTRA_COLOURS[number % len(EXTRA_COLOURS)]
        material = EXTRA_MATERIALS[(number // len(EXTRA_COLOURS)) % len(EXTRA_MATERIALS)]
        shape = shapes[(number // (len(EXTRA_COLOURS) * len(EXTRA_MATERIALS))) % len(shapes)]
        relation = relations[(number // (len(EXTRA_COLOURS) * len(EXTRA_MATERIALS) * len(shapes)))
                             % len(relations)]
        left, right = shape
        uid = f"tgt_pair_{number:05d}"
        rows.append(dict(uid=uid, family="attr",
                         raw_prompt=f"A photo of a {colour} {material} {left} {relation} and a "
                                    f"{colour} {material} {right} next to it, "
                                    f"only one of each, both clearly visible."))
    return rows


def _relation_prompts(start, limit, rng):
    """Explicit spatial relations; each row picks a fresh pair/axis/colour combination.

    The first version indexed a five-item list with ``number % len(pairs)``, so
    1,500 prompts collapsed to five distinct sentences and the family was spent
    after a handful of images. Combinations are drawn without replacement here.
    """
    rows = []
    shapes = ["bicycle", "ladder", "cactus", "rock", "lamp", "crate", "cat", "dog",
              "jar", "can", "stool", "bucket", "sign", "rope", "fruit bowl", "kettle"]
    seen = set()
    number = start
    while len(rows) < int(limit):
        first = shapes[(number + rng.randrange(len(shapes))) % len(shapes)]
        second = shapes[(number * 3 + rng.randrange(len(shapes)) + 1) % len(shapes)]
        if first == second:
            number += 1
            continue
        c1, c2 = rng.sample(EXTRA_COLOURS, 2)
        a, b = rng.choice([("left", "right"), ("above", "below"),
                           ("nearer the camera than", "farther from the camera than")])
        key = (first, second, c1, c2, a)
        number += 1
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(uid=f"tgt_rel_{start + len(rows):05d}", family="space",
                         raw_prompt=f"A photo where the {c1} {first} is {a} the {c2} {second}, "
                                    f"both the same size and both fully visible."))
    return rows


def _text_prompts(start, limit, rng):
    """Legible text in the frame: models frequently drop or garble part of it."""
    rows = []
    words = ["OPEN", "SALE", "KITCHEN", "EXIT", "FRESH", "BAKERY", "CAFE", "STUDIO",
             "ENTRANCE", "CLOSED", "MARKET", "GALLERY"]
    where = ["on a wooden sign", "on a shop window", "on a metal plate",
             "on a chalkboard", "on a white wall", "on a brick pillar",
             "on a hanging board", "on a glass door"]
    shade = ["in daylight", "at dusk", "under warm light", "in overcast light",
             "under a single spotlight", "in soft morning light"]
    for offset in range(int(limit)):
        number = start + offset
        word = words[number % len(words)]
        where_i = (number // len(words)) % len(where)
        shade_i = (number // (len(words) * len(where))) % len(shade)
        extra = (number // (len(words) * len(where) * len(shade))) % len(EXTRA_COLOURS)
        uid = f"tgt_text_{number:05d}"
        rows.append(dict(uid=uid, family="text",
                         raw_prompt=f"A photo of the {EXTRA_COLOURS[extra]} word \"{word}\" written "
                                    f"clearly {where[where_i]} {shade[shade_i]}, "
                                    f"the whole word legible and centred."))
    return rows


# Failure shapes that a text-to-image model actually makes on a crowded or
# relational scene. Each one states a requirement the picture can visibly break,
# so a completion-side audit has something concrete to find.
EXTRA_OBJECTS = ["red apples", "green croissants", "blue coffee mugs", "yellow ducks",
                 "wooden pencils", "white sheep", "orange balloons", "purple candles",
                 "silver spoons", "brown ceramic cups", "pink peonies", "green watermelons",
                 "red toy cars", "brass keys", "white eggs", "blue marbles"]
EXTRA_SCENES = ["on a weathered wooden table", "on a white kitchen counter",
                "inside a wicker basket", "on a bookshelf", "on a stone terrace",
                "on a red picnic blanket", "inside a glass display case",
                "on a marble counter", "on a wooden floor", "on a metal tray"]
EXTRA_LEVELS = ["8", "9", "10", "11", "12", "14"]
EXTRA_COLOURS = ["red", "orange", "yellow", "green", "blue", "purple",
                 "pink", "brown", "grey", "black", "white", "teal"]
EXTRA_MATERIALS = ["wooden", "ceramic", "glass", "metal", "woven", "stone"]


def _crowd_prompts(start, limit, rng):
    """Many instances, all entirely inside the frame: the usual failure is under-drawing."""
    rows = []
    for offset in range(int(limit)):
        number = start + offset
        n = EXTRA_LEVELS[number % len(EXTRA_LEVELS)]
        obj = f"{EXTRA_COLOURS[number % 12]} {EXTRA_MATERIALS[(number // 12) % 6]} {rng.choice(EXTRA_OBJECTS)}"
        scene = rng.choice(EXTRA_SCENES)
        uid = f"tgt_crowd_{number:05d}"
        rows.append(dict(uid=uid, family="over", target_count=int(n),
                         raw_prompt=f"An overhead photo of exactly {n} {obj} laid out side by side {scene}, "
                                    f"every object entirely inside the frame."))
    return rows


def _pair_prompts(start, limit, rng):
    """Two same-kind subjects with crossing attributes: the usual failure is swapping them."""
    rows = []
    shapes = [("mug", "plate"), ("chair", "stool"), ("bottle", "jar"), ("hat", "scarf"),
              ("cup", "saucer"), ("box", "crate"), ("vase", "bowl"), ("spoon", "fork")]
    relations = ["on the left", "in the foreground", "near the window", "on the shelf"]
    for offset in range(int(limit)):
        number = start + offset
        colour = EXTRA_COLOURS[number % len(EXTRA_COLOURS)]
        material = EXTRA_MATERIALS[(number // len(EXTRA_COLOURS)) % len(EXTRA_MATERIALS)]
        shape = shapes[(number // (len(EXTRA_COLOURS) * len(EXTRA_MATERIALS))) % len(shapes)]
        relation = relations[(number // (len(EXTRA_COLOURS) * len(EXTRA_MATERIALS) * len(shapes)))
                             % len(relations)]
        left, right = shape
        uid = f"tgt_pair_{number:05d}"
        rows.append(dict(uid=uid, family="attr",
                         raw_prompt=f"A photo of a {colour} {material} {left} {relation} and a "
                                    f"{colour} {material} {right} next to it, "
                                    f"only one of each, both clearly visible."))
    return rows


def _relation_prompts(start, limit, rng):
    """Explicit spatial relations; each row picks a fresh pair/axis/colour combination.

    The first version indexed a five-item list with ``number % len(pairs)``, so
    1,500 prompts collapsed to five distinct sentences and the family was spent
    after a handful of images. Combinations are drawn without replacement here.
    """
    rows = []
    shapes = ["bicycle", "ladder", "cactus", "rock", "lamp", "crate", "cat", "dog",
              "jar", "can", "stool", "bucket", "sign", "rope", "fruit bowl", "kettle"]
    seen = set()
    number = start
    while len(rows) < int(limit):
        first = shapes[(number + rng.randrange(len(shapes))) % len(shapes)]
        second = shapes[(number * 3 + rng.randrange(len(shapes)) + 1) % len(shapes)]
        if first == second:
            number += 1
            continue
        c1, c2 = rng.sample(EXTRA_COLOURS, 2)
        a, b = rng.choice([("left", "right"), ("above", "below"),
                           ("nearer the camera than", "farther from the camera than")])
        key = (first, second, c1, c2, a)
        number += 1
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(uid=f"tgt_rel_{start + len(rows):05d}", family="space",
                         raw_prompt=f"A photo where the {c1} {first} is {a} the {c2} {second}, "
                                    f"both the same size and both fully visible."))
    return rows


def _occlusion_prompts(start, limit, rng):
    """Two objects where one must stay behind the other: models merge or swap them."""
    rows = []
    fronts = ["water bottle", "cardboard box", "potted palm", "framed painting",
              "tall vase", "lamp", "crate", "ladder", "book stack", "bowl", "clock", "basket"]
    backs = ["glass cup", "red ball", "stool", "candle", "small bowl", "book", "mug", "pencil"]
    for offset in range(int(limit)):
        number = start + offset
        f_i = number % 12
        b_i = (f_i + 1 + (number // 12) % 11) % 12
        front = f"a {EXTRA_COLOURS[f_i]} {fronts[(number // 12) % len(fronts)]}"
        back = f"a {EXTRA_COLOURS[b_i]} {backs[(number // (12 * len(fronts))) % len(backs)]}"
        uid = f"tgt_occ_{number:05d}"
        rows.append(dict(uid=uid, family="occlusion",
                         raw_prompt=f"A photo where {front} stands fully in front of {back}, "
                                    f"both clearly distinct and neither merged with the other."))
    return rows


def _colour_bind_prompts(start, limit, rng):
    """Same shape, different colours: the usual failure is one colour for both."""
    rows = []
    objects = ["ceramic bowls", "wooden crates", "glass vases", "clay pots", "metal tins",
               "paper cups", "linen cushions", "stone blocks"]
    placements = ["side by side", "on a bright surface", "in a row", "near the window",
                  "on a shelf", "on the floor"]
    for offset in range(int(limit)):
        number = start + offset
        obj = objects[(number // 132) % len(objects)]
        placement = placements[(number // (132 * len(objects))) % len(placements)]
        c1 = EXTRA_COLOURS[number % 12]
        c2 = EXTRA_COLOURS[(number % 12 + 1 + ((number // 12) % 11)) % 12]
        uid = f"tgt_col_{number:05d}"
        rows.append(dict(uid=uid, family="colour",
                         raw_prompt=f"A photo of two identical {obj} {placement}, the left one {c1} "
                                    f"and the right one {c2}, no other colours."))
    return rows


def _count_cap_prompts(start, limit, rng):
    """A small exact count in a cluttered context: models over-draw."""
    rows = []
    for offset in range(int(limit)):
        number = start + offset
        n = [2, 3, 4][number % 3]
        obj = (f"{EXTRA_COLOURS[number % 12]} {EXTRA_MATERIALS[(number // 12) % 6]} "
               f"{EXTRA_OBJECTS[(number // 72) % len(EXTRA_OBJECTS)]}")
        scene = EXTRA_SCENES[(number // 3) % len(EXTRA_SCENES)]
        uid = f"tgt_cap_{number:05d}"
        rows.append(dict(uid=uid, family="under", target_count=n,
                         raw_prompt=f"A sparse photo with exactly {n} {obj} {scene} and nothing else "
                                    f"resting on the same surface."))
    return rows

BUILDERS = {
    "occlusion": lambda start, limit, rng: _occlusion_prompts(start, limit, rng),
    "colour": lambda start, limit, rng: _colour_bind_prompts(start, limit, rng),
    "cap": lambda start, limit, rng: _count_cap_prompts(start, limit, rng),
    "crowd": lambda start, limit, rng: _crowd_prompts(start, limit, rng),
    "pair": lambda start, limit, rng: _pair_prompts(start, limit, rng),
    "relation": lambda start, limit, rng: _relation_prompts(start, limit, rng),
    "text": lambda start, limit, rng: _text_prompts(start, limit, rng),
    "over": lambda start, limit, rng: _over_prompts(start, limit, rng),
    "count": lambda start, limit, rng: _count_prompts(start, limit, rng),
    "under": lambda start, limit, rng: _under_prompts(start, limit, rng),
    "space": lambda start, limit, rng: _space_prompts(start, limit, rng),
    "attr": lambda start, limit, rng: _attr_prompts(start, limit, rng),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default="crowd,pair,relation,text,occlusion,colour,cap")
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
