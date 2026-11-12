"""Linear image pairs and sampling for the Stage 1 recipe."""

import hashlib
import json
import random
from pathlib import Path

import torch

from . import LINEAR_PAIR, ROLE_L_EDIT, ROLE_L_REVERSE
from .conditions import compact
from .field import protocol, sample_state_time, weighted_choice
from .images import load_canvas, load_pair


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    """Prompt rows from JSONL, a JSON list, or a ``records`` object."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    raise ValueError("Prompt file must be a list, records object, or JSONL")


def _approved(row):
    value = row.get("semantic_approved", row.get("approved", True))
    return bool(value)


def _remember(groups, key, split):
    if key is None:
        return
    previous = groups.setdefault(str(key), split)
    if previous != split:
        raise ValueError("A prompt or source family crosses splits")


def _family(row, key, fallback):
    return row.get(key) or fallback


def _keep_or_edit(row, cfg, keep):
    """The instruction triple for a keep draw or for the record's own edit."""
    if keep:
        return cfg["keep_instruction"], row["parent_condition"], ""
    return row["edit_instruction"], row["tail_condition"], row["preserve"]


# Keys early caches carried that describe the draw, not the content.
RETIRED_PROTOCOL_KEYS = frozenset({"state_times", "state_time_probs"})


def protocol_matches(cached, bound):
    """Whether a cached manifest was produced by this run's discretisation.

    Caches written before the grid/sampling split stored ``state_times`` and
    ``state_time_probs`` and no ``phase_grid_times``: the drawn times *were* the
    grid back then. Those keys describe how times were drawn, not what the cache
    contains, so they are tolerated on either side; and a cache that predates
    the split is read as having walked exactly the nodes it stored. Everything
    else, the grid included, is compared exactly.
    """
    if not isinstance(cached, dict):
        return False
    compare = {k: v for k, v in cached.items() if k not in RETIRED_PROTOCOL_KEYS}
    if "phase_grid_times" not in compare:
        compare["phase_grid_times"] = [float(t) for t in cached.get("state_times", [])]
    return compare == bound


class LinearManifest:
    """Existing high-quality edit pairs consumed directly as L data."""

    def __init__(self, path, rt, cfg):
        self.path = Path(path).resolve()
        self.root = self.path.parent
        self.rt, self.cfg = rt, cfg
        manifest = _read(self.path)
        self.manifest = manifest
        if "records" not in manifest:
            raise ValueError("A linear manifest needs a records list")
        self.bound = protocol(rt, cfg)
        if manifest.get("protocol") is not None and not protocol_matches(manifest["protocol"], self.bound):
            raise ValueError("Linear manifest protocol differs from this run")
        self.rows = []
        prompt_splits, source_splits = {}, {}
        for index, raw in enumerate(manifest["records"]):
            row = self._normalize(raw)
            _remember(prompt_splits, row["prompt_family"], row["split"])
            _remember(source_splits, row["source_family"], row["split"])
            self.rows.append(row)
        self.train_indices = [i for i, row in enumerate(self.rows) if row["split"] == "train"]
        if not self.train_indices:
            raise ValueError("The linear manifest has no training rows")

    def _normalize(self, row):
        role = row.get("role")
        allowed_linear = (None, "native_edit", LINEAR_PAIR, ROLE_L_EDIT, ROLE_L_REVERSE)
        if role not in allowed_linear:
            raise ValueError("A linear manifest only carries edit pairs")
        if role != ROLE_L_REVERSE:
            role = ROLE_L_EDIT
        split = row.get("split")
        if split not in ("train", "val", "test"):
            raise ValueError("Split must be train, val, or test")
        if not _approved(row):
            raise ValueError("Unapproved edit pairs cannot enter training")
        uid = str(row.get("uid") or "").strip()
        if not uid:
            raise ValueError("Every linear pair needs a uid")
        source_image = row.get("source_image")
        target_image = row.get("target_image")
        if not source_image or not target_image:
            raise ValueError("Every linear pair needs source and target images")
        if not _resolve(self.root, source_image).exists() or not _resolve(self.root, target_image).exists():
            raise ValueError("A linear pair image is missing: " + uid)
        edit = row.get("edit_instruction") or row.get("edit_condition") or row.get("instruction")
        if not str(edit or "").strip():
            raise ValueError("A linear pair needs a short edit instruction: " + uid)
        edit = compact(edit)
        # Stage 1 reads the state pair and the instruction; the parent caption
        # conditioned a lookahead preview the editor no longer has, and the tail
        # caption only matters once the frozen tail is supervised. Both are
        # therefore optional here and carried through untouched when present.
        parent = compact(row.get("parent_condition") or row.get("raw_prompt") or row.get("source_prompt"))
        tail = compact(row.get("tail_condition"))
        tail_source = str(row.get("tail_source") or "declared")
        frozen = row.get("condition_protocol") or self.cfg["condition_protocol"]
        if frozen == "derived_parent_plus_edit":
            tail_source = "derived_parent_plus_edit"
            frozen = self.cfg["condition_protocol"]
        elif frozen != self.cfg["condition_protocol"]:
            raise ValueError("Linear condition protocol differs from this run: " + uid)
        return dict(
            uid=uid,
            split=split,
            source_image=str(_resolve(self.root, source_image)),
            target_image=str(_resolve(self.root, target_image)),
            source_family=_family(row, "source_family", hashlib.sha1(str(source_image).encode()).hexdigest()[:16]),
            prompt_family=_family(row, "prompt_family", uid),
            edit_instruction=edit,
            parent_condition=parent,
            tail_condition=tail,
            preserve=compact(row.get("preserve")),
            condition_protocol=frozen,
            tail_source=tail_source,
            semantic_quality_source=row.get("semantic_quality_source", "legacy_semantic_approved"),
            target_quality_source=row.get("target_quality_source", "legacy_semantic_approved"),
            task=row.get("task"),
            geometry=row.get("geometry"),
            role=role,
        )

    @property
    def count(self):
        return len(self.train_indices)

    def role_pools(self):
        """Forward and reversed pairs are sampled apart even though both are linear."""
        grouped = {}
        for index in self.train_indices:
            grouped.setdefault(self.rows[index]["role"], []).append(index)
        return {
            "linear": self,
            ROLE_L_EDIT: LinearPool(self, grouped.get(ROLE_L_EDIT, [])),
            ROLE_L_REVERSE: LinearPool(self, grouped.get(ROLE_L_REVERSE, [])),
        }

    def available_times(self):
        return [float(t) for t in self.cfg["state_times"]]

    def choose_time(self, rng, time=None):
        if time is not None:
            if float(time) not in self.available_times():
                raise ValueError("Unsupported linear state time")
            return float(time)
        return sample_state_time(self.cfg, rng)

    def choose_index(self, time, rng):
        return rng.choice(self.train_indices)

    def sample(self, index, time, keep=False):
        row = self.rows[index]
        if keep:
            declared = row.get("geometry") or {}
            crop = declared.get("source") if isinstance(declared, dict) else None
            source, valid, _ = load_canvas(
                row["source_image"], self.rt.width, self.rt.height, self.rt.device,
                crop=None if crop in (None, "full") else crop,
            )
            target = source.clone()
            frozen = self.cfg["condition_protocol"]
        else:
            source, target, valid, _ = load_pair(
                row, self.rt.width, self.rt.height, self.rt.device
            )
            frozen = row["condition_protocol"]
        edit, tail, preserve = _keep_or_edit(row, self.cfg, keep)
        return dict(
            uid=row["uid"],
            source_kind=LINEAR_PAIR,
            keep=bool(keep),
            source=source,
            target=target,
            state_a=None,
            state_b=None,
            preview=None,
            base_prediction=None,
            valid=valid,
            time=float(time),
            edit_request=edit,
            parent_condition=row["parent_condition"],
            tail_condition=tail,
            preserve=preserve,
            condition_protocol=frozen,
            tail_source=row["tail_source"],
            source_family=row["source_family"],
            prompt_family=row["prompt_family"],
            semantic_quality_source=row["semantic_quality_source"],
            target_quality_source=row["target_quality_source"],
            role=row["role"],
        )



class LinearPool:
    """One sub-role of the linear manifest, with the same interface as a state pool."""

    def __init__(self, manifest, indices):
        self.manifest = manifest
        self.indices = list(indices)
        self.cfg = manifest.cfg

    @property
    def count(self):
        return len(self.indices)

    def available_times(self):
        return self.manifest.available_times()

    def choose_time(self, rng, time=None):
        return self.manifest.choose_time(rng, time)

    def choose_index(self, time, rng):
        if not self.indices:
            raise ValueError("An empty linear sub-role pool was sampled")
        return rng.choice(self.indices)

    def sample(self, index, time, keep=False):
        return self.manifest.sample(index, time, keep=keep)


class TrainingSampler:
    """Sample L_edit pairs with the original Stage 1 RNG sequence."""

    def __init__(self, linear, cfg, rank=0):
        self.cfg = cfg
        self.rng = random.Random(int(cfg["seed"]) + int(rank))
        self.keep_draws = 0
        self.draws = 0
        pools = linear.role_pools()
        self.pools = {ROLE_L_EDIT: pools[ROLE_L_EDIT] if pools[ROLE_L_EDIT].count else linear}
        self.weights = {ROLE_L_EDIT: 1.0}
        self.configured = dict(cfg["role_weights"])
        self.keep_fraction = 0.0
        self.seen = set()

    @property
    def configured_weights(self):
        return dict(self.configured)

    @property
    def actual_weights(self):
        return dict(self.weights)

    def input_domains(self):
        return ["linear"] if self.seen else []

    def _role(self):
        keys = list(self.weights)
        return weighted_choice(self.rng, keys, [self.weights[key] for key in keys])

    def draw(self, time=None):
        role = self._role()
        pool = self.pools[role]
        keep = self.keep_fraction > 0 and self.rng.random() < self.keep_fraction
        chosen_time = pool.choose_time(self.rng, time)
        index = pool.choose_index(chosen_time, self.rng)
        sample = pool.sample(index, chosen_time, keep=keep)
        sample["role"] = role
        self.seen.add(sample["role"])
        self.draws += 1
        self.keep_draws += int(bool(sample.get("keep")))
        return sample

    def state_dict(self):
        return dict(rng=self.rng.getstate(), seen=sorted(self.seen), draws=self.draws, keep=self.keep_draws)

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        self.seen = set(state.get("seen", []))
        self.draws = int(state.get("draws", 0))
        self.keep_draws = int(state.get("keep", 0))


def data_version(*paths):
    digest = hashlib.sha256()
    for path in sorted(Path(item).resolve() for item in paths if item is not None):
        digest.update(str(path).encode())
        files = sorted(path.glob("manifest*.json")) if path.is_dir() else [path]
        for file in files:
            digest.update(str(file).encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()
