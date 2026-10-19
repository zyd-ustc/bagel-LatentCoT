"""GEN-only pairs from a caption JSONL or the annotation-index generation split.

The index path reads one generation split. Training uses ``train``. The
curriculum probe uses ``val``. The text condition is the prompt stored with
the target image (`goal_prompt`), not `target_caption`.
"""
from __future__ import annotations

import json
import math
import mmap
import os
import random
from collections import OrderedDict
from pathlib import Path

from ..diagnostics.corpus import REF, resolve_path
from ..interaction.data import PairSampler, load_pairs


ACCEPTED_FORMATS = ("annotation_index_v1", "annotation_index_v2")
GENERATION_ROLE = 2


class JsonlPairSource:
    """Existing caption-image JSONL. `caption` is the text condition."""

    source_kind = "caption_image_jsonl"
    text_field = "caption"
    fingerprint = None

    def __init__(self, path, seed):
        self.rows = load_pairs(path)
        self.sampler = PairSampler(self.rows, seed)
        self.pair_count = len(self.rows)

    @property
    def rng(self):
        return self.sampler.rng

    def take(self, count):
        return self.sampler.take(count)

    def state_dict(self):
        state = self.sampler.state_dict()
        state["source_kind"] = self.source_kind
        return state

    def load_state_dict(self, state):
        if "order" not in state:
            raise ValueError("resume source is not the caption-image jsonl")
        kind = state.get("source_kind", self.source_kind)
        if kind != self.source_kind:
            raise ValueError("resume source is not the caption-image jsonl")
        self.sampler.load_state_dict({key: state[key] for key in ("order", "cursor", "rng")})

    def close(self):
        return None


class _GenerationView:
    def __init__(self, source: "GenerationPromptSource", info: dict):
        self.source = source
        self.info = info
        self.path = source.root / "views" / info["path"]
        self.stream = self.path.open("rb")
        self.index = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return int(self.info["count"])

    def pair(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard, offset, role = REF.unpack_from(self.index, index * REF.size)
        if role != GENERATION_ROLE:
            raise ValueError(f"{self.info['path']} contains role {role}, expected generation role 2")
        row = self.source.record(int(shard), int(offset))
        prompt = str(row.get("goal_prompt") or "").strip()
        original = row.get("original") or {}
        image = resolve_path(
            row.get("target_image"),
            original.get("path", ""),
            self.source.manifest.get("registry", {}).get("roots", {}),
        )
        if not prompt or not image or not os.path.isabs(image):
            raise ValueError(
                f"{self.info['path']} index {index} needs a nonempty goal_prompt and an absolute target_image"
            )
        return {"caption": prompt, "image": image, "prompt": prompt}

    def close(self):
        self.index.close()
        self.stream.close()


class GenerationPromptSource:
    """Sqrt-weighted views for one generation split. Only `take` materializes a row."""

    source_kind = "annotation_index"
    text_field = "goal_prompt"

    def __init__(self, root, seed, split="train"):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"annotation index has no manifest.json: {self.root}")
        self.manifest = json.loads(manifest_path.read_text())
        fmt = self.manifest.get("format")
        if fmt not in ACCEPTED_FORMATS:
            raise ValueError(f"unsupported annotation index format: {fmt}")
        sampling = self.manifest.get("source_sampling", "sqrt_task_records")
        if sampling != "sqrt_task_records":
            raise ValueError(f"unsupported source sampling: {sampling}")
        fingerprint = self.manifest.get("fingerprint")
        if not fingerprint:
            raise ValueError("annotation index is missing a fingerprint")
        self.fingerprint = fingerprint
        self.split = str(split)
        selected = [
            info for info in self.manifest.get("views", [])
            if info.get("task") == "generation" and info.get("split") == self.split and info.get("count")
        ]
        if not selected:
            raise ValueError(f"generation {self.split} split is empty")
        self.views = [_GenerationView(self, info) for info in selected]
        self.pair_count = sum(len(view) for view in self.views)
        self.weights = [len(view) ** 0.5 for view in self.views]
        self.rng = random.Random(seed)
        self.cursors = [0] * len(self.views)
        self.epochs = [0] * len(self.views)
        self.transforms = [self._coprime(len(view)) for view in self.views]
        self.handles: OrderedDict[int, object] = OrderedDict()

    def _coprime(self, count: int) -> tuple[int, int]:
        scale = self.rng.randrange(1, max(2, count))
        while math.gcd(scale, count) != 1:
            scale = (scale + 1) % count or 1
        return scale, self.rng.randrange(count)

    def record(self, shard: int, offset: int) -> dict:
        if shard not in self.handles:
            path = self.root / "records" / self.manifest["shards"][shard]["path"]
            self.handles[shard] = path.open("rb")
            if len(self.handles) > 8:
                self.handles.popitem(last=False)[1].close()
        stream = self.handles[shard]
        self.handles.move_to_end(shard)
        stream.seek(offset)
        return json.loads(stream.readline())

    def take(self, count: int) -> list[dict]:
        result = []
        for _ in range(count):
            index = self.rng.choices(range(len(self.views)), weights=self.weights, k=1)[0]
            view = self.views[index]
            cursor = self.cursors[index]
            scale, shift = self.transforms[index]
            position = (scale * cursor + shift) % len(view)
            result.append(view.pair(position))
            self.cursors[index] += 1
            if self.cursors[index] == len(view):
                self.cursors[index] = 0
                self.epochs[index] += 1
                self.transforms[index] = self._coprime(len(view))
        return result

    def state_dict(self):
        return {
            "source_kind": self.source_kind,
            "fingerprint": self.fingerprint,
            "views": [view.info["path"] for view in self.views],
            "rng": self.rng.getstate(),
            "cursors": list(self.cursors),
            "epochs": list(self.epochs),
            "transforms": list(self.transforms),
        }

    def load_state_dict(self, state):
        if state.get("source_kind") != self.source_kind:
            raise ValueError("resume source is not the annotation index")
        paths = [view.info["path"] for view in self.views]
        if state.get("fingerprint") != self.fingerprint or state.get("views") != paths:
            raise ValueError("resume corpus views changed")
        self.rng.setstate(state["rng"])
        self.cursors = list(state["cursors"])
        self.epochs = list(state["epochs"])
        self.transforms = list(state["transforms"])

    def close(self):
        for view in self.views:
            view.close()
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def open_gen_only_source(path, seed, split="train"):
    """A directory is the annotation index. A file is caption-image JSONL."""
    source = Path(path)
    if source.is_dir():
        return GenerationPromptSource(source, seed, split=split)
    if split != "train":
        raise ValueError("a caption JSONL has no held-out split")
    if not source.is_file():
        raise ValueError(f"GEN-only train source is missing: {source}")
    return JsonlPairSource(source, seed)


def open_curriculum_val(train_path, val_path, seed):
    """Validation rows for the curriculum probe. Observe views stay closed.

    An annotation index supplies its generation val split. A train JSONL has
    no split, so the caller must pass a separate val JSONL.
    """
    if val_path:
        source = Path(val_path)
        if source.is_dir():
            return GenerationPromptSource(source, seed, split="val")
        if not source.is_file():
            raise ValueError(f"curriculum val source is missing: {source}")
        return JsonlPairSource(source, seed)
    source = Path(train_path)
    if source.is_dir():
        return GenerationPromptSource(source, seed, split="val")
    raise ValueError("curriculum validation needs an annotation index or a separate val JSONL")
