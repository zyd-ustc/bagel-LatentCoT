"""Linear, inverted, and native-history records under one state-flow contract."""

import hashlib
import json
import random
from pathlib import Path

import torch

from . import (
    CACHE_KIND, INVERTED_PAIR, LINEAR_PAIR, NATIVE_PARENT,
    ROLE_I_EDIT, ROLE_L_EDIT, ROLE_L_REVERSE, ROLE_R_EDIT, ROLE_R_KEEP,
    ROLE_R_ACTIVE, ROLE_R_REPAIR, ROLES,
)
from .conditions import compact, derived_tail_condition
from .field import protocol, sample_state_time, time_key, weighted_choice
from .images import load_canvas, load_pair, valid_mask

DOMAIN = {
    LINEAR_PAIR: "linear",
    INVERTED_PAIR: "inverted",
    NATIVE_PARENT: "native_history",
}

ROLE_DOMAIN = {
    ROLE_L_EDIT: "linear",
    ROLE_L_REVERSE: "linear",
    ROLE_I_EDIT: "inverted",
    ROLE_R_EDIT: "native_history",
    ROLE_R_REPAIR: "native_history",
    ROLE_R_ACTIVE: "native_history",
    ROLE_R_KEEP: "native_history",
}


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
        if manifest.get("protocol") is not None and manifest["protocol"] != self.bound:
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
        parent = row.get("parent_condition") or row.get("raw_prompt") or row.get("source_prompt")
        if not str(edit or "").strip():
            raise ValueError("A linear pair needs a short edit instruction: " + uid)
        if not str(parent or "").strip():
            raise ValueError("A linear pair needs its parent scene description: " + uid)
        edit = compact(edit)
        parent = compact(parent)
        tail = row.get("tail_condition")
        if str(tail or "").strip():
            tail = compact(tail)
            tail_source = str(row.get("tail_source") or "declared")
        else:
            if not (bool(row.get("allow_derived_tail")) or bool(self.cfg.get("allow_derived_tail"))):
                raise ValueError(
                    "A linear pair needs an independent full tail_condition: " + uid
                )
            tail = derived_tail_condition(parent, edit)
            tail_source = "derived_parent_plus_edit"
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


class StatePairManifest:
    """I or R records produced by prepare_state_pairs.py."""

    def __init__(self, path, rt, cfg, expected_kind):
        if expected_kind not in (INVERTED_PAIR, NATIVE_PARENT):
            raise ValueError("Unknown state-pair source kind")
        self.path = Path(path).resolve()
        if self.path.is_dir():
            singles = sorted(self.path.glob("manifest.json"))
            shards = sorted(self.path.glob("manifest.shard_*.json"))
            if singles and shards:
                raise ValueError("A cache directory cannot mix a single and sharded manifests")
            manifests = shards or singles
        else:
            manifests = [self.path]
        if not manifests:
            raise ValueError("No state-pair manifest found")
        self.root = manifests[0].parent
        self.rt, self.cfg = rt, cfg
        self.expected_kind = expected_kind
        self.bound = protocol(rt, cfg)
        records = []
        for manifest_path in manifests:
            manifest = _read(manifest_path)
            if manifest.get("kind") != CACHE_KIND:
                raise ValueError("State-pair cache kind differs")
            if manifest.get("protocol") != self.bound:
                raise ValueError("State-pair cache protocol differs from this run")
            if "records" not in manifest:
                raise ValueError("A state-pair cache needs a records list")
            if manifest_path.parent != self.root:
                raise ValueError("Sharded state-pair manifests must share one directory")
            records.extend(manifest["records"])
        self.manifest = dict(kind=CACHE_KIND, protocol=self.bound, records=records)
        self.rows = []
        prompt_splits, source_splits = {}, {}
        for raw in records:
            row = self._normalize(raw)
            _remember(prompt_splits, row["prompt_family"], row["split"])
            _remember(source_splits, row["source_family"], row["split"])
            self.rows.append(row)
        self.by_time = {time_key(t): [] for t in cfg["state_times"]}
        for index, row in enumerate(self.rows):
            if row["split"] != "train":
                continue
            for key, item in row["states"].items():
                if item["accepted"] and key in self.by_time:
                    self.by_time[key].append(index)
        self.train_indices = sorted({index for values in self.by_time.values() for index in values})
        if not self.train_indices:
            raise ValueError("The state-pair cache has no accepted training times")

    def _normalize(self, row):
        if row.get("source_kind") != self.expected_kind:
            raise ValueError("State-pair source kind differs from the requested cache")
        split = row.get("split")
        if split not in ("train", "val", "test"):
            raise ValueError("Split must be train, val, or test")
        uid = str(row.get("uid") or "").strip()
        if not uid:
            raise ValueError("Every state-pair record needs a uid")
        for key in ("parent_condition", "tail_condition", "edit_instruction"):
            if not str(row.get(key) or "").strip():
                raise ValueError("State-pair record is missing " + key + ": " + uid)
        if row.get("condition_protocol") != self.cfg["condition_protocol"]:
            raise ValueError("State-pair condition protocol differs: " + uid)
        if not isinstance(row.get("states"), dict) or not row["states"]:
            raise ValueError("State-pair record has no state times: " + uid)
        supported = {time_key(t) for t in self.cfg["state_times"]}
        states = {}
        for key, item in row["states"].items():
            if time_key(float(key)) != key:
                raise ValueError("State time keys must use six decimals: " + key)
            if key not in supported:
                raise ValueError("Cache contains an unsupported state time: " + key)
            path = _resolve(self.root, item["path"])
            if not path.exists():
                raise ValueError("Cached state file is missing: " + str(path))
            states[key] = dict(item, path=str(path), accepted=bool(item.get("accepted")))
        return dict(
            uid=uid,
            source_kind=row["source_kind"],
            split=split,
            source_family=_family(row, "source_family", uid),
            prompt_family=_family(row, "prompt_family", uid),
            source_image=row.get("source_image"),
            target_image=row.get("target_image"),
            edit_instruction=compact(row["edit_instruction"]),
            parent_condition=compact(row["parent_condition"]),
            tail_condition=compact(row["tail_condition"]),
            preserve=compact(row.get("preserve")),
            condition_protocol=row["condition_protocol"],
            states=states,
            valid_region=row.get("valid_region"),
            semantic_quality_source=row.get("semantic_quality_source"),
            target_quality_source=row.get("target_quality_source"),
            role=row.get("role") or self._default_role(row),
        )

    @property
    def count(self):
        return len(self.train_indices)

    def available_times(self):
        return [
            float(t) for t in self.cfg["state_times"]
            if self.by_time[time_key(t)]
        ]

    def choose_time(self, rng, time=None):
        available = self.available_times()
        if not available:
            raise ValueError("No accepted state time in " + self.expected_kind)
        if time is not None:
            if float(time) not in available:
                raise ValueError("The selected state time has no accepted label")
            return float(time)
        configured = [float(t) for t in self.cfg["state_times"]]
        probs = [
            float(self.cfg["state_time_probs"][configured.index(value)])
            for value in available
        ]
        return weighted_choice(rng, available, probs, total=sum(probs))

    def choose_index(self, time, rng):
        indices = self.by_time[time_key(time)]
        if not indices:
            raise ValueError("No record at the requested state time")
        return rng.choice(indices)

    def sample(self, index, time, keep=False):
        row = self.rows[index]
        key = time_key(time)
        item = row["states"].get(key)
        if item is None or not item["accepted"]:
            raise ValueError("Requested state time is not accepted for " + row["uid"])
        payload = torch.load(item["path"], map_location="cpu", weights_only=False)
        if time_key(payload.get("time", time)) != key:
            raise ValueError("Cached state file time differs: " + item["path"])
        state_a = payload["state_a"].float().to(self.rt.device)
        state_b = payload["state_b"].float().to(self.rt.device)
        preview = payload["preview"].float().to(self.rt.device)
        base = payload.get("base_prediction")
        if base is not None:
            base = base.float().to(self.rt.device)
            if base.shape != state_a.shape or not torch.isfinite(base).all():
                raise ValueError("Cached base prediction differs: " + item["path"])
        valid = payload.get("valid")
        if valid is not None:
            valid = valid.float().to(self.rt.device)
        elif row.get("valid_region") is not None:
            valid = valid_mask(row["valid_region"], self.rt.width, self.rt.height, self.rt.device)
        else:
            valid = torch.ones(1, 1, self.rt.height, self.rt.width, device=self.rt.device)
        if preview.shape != state_a.shape or valid.shape != (1, 1, self.rt.height, self.rt.width):
            raise ValueError("Cached state/preview/valid shape differs: " + item["path"])
        if not (
            torch.isfinite(state_a).all()
            and torch.isfinite(state_b).all()
            and torch.isfinite(preview).all()
        ):
            raise ValueError("Cached state contains a nonfinite value: " + item["path"])
        if keep:
            state_b = state_a.clone()
        edit, tail, preserve = _keep_or_edit(row, self.cfg, keep)
        terminal = bool((self.cfg.get("terminal") or {}).get("enabled", False))
        x_source = x_teacher = None
        if terminal:
            x_source, _, _ = load_canvas(
                _resolve(self.root, row["source_image"]),
                self.rt.width, self.rt.height, self.rt.device,
                crop=(row.get("geometry") or {}).get("source"),
            )
            target_path = row.get("target_image")
            if keep or not target_path:
                x_teacher = x_source
            else:
                x_teacher, _, _ = load_canvas(
                    _resolve(self.root, target_path),
                    self.rt.width, self.rt.height, self.rt.device,
                    crop=(row.get("geometry") or {}).get("target"),
                )
        return dict(
            uid=row["uid"],
            source_kind=row["source_kind"],
            keep=bool(keep),
            source=None,
            target=None,
            x_source=x_source,
            x_teacher=x_teacher,
            state_a=state_a,
            state_b=state_b,
            preview=preview,
            base_prediction=base,
            valid=valid,
            time=float(time),
            edit_request=edit,
            parent_condition=row["parent_condition"],
            tail_condition=tail,
            preserve=preserve,
            condition_protocol=self.cfg["condition_protocol"],
            source_family=row["source_family"],
            prompt_family=row["prompt_family"],
            semantic_quality_source=row.get("semantic_quality_source"),
            target_quality_source=row.get("target_quality_source"),
            role=row["role"],
        )

    def _default_role(self, row):
        """Older caches carry no role; keep is derivable from how it was made."""
        if self.expected_kind == INVERTED_PAIR:
            return ROLE_I_EDIT
        kept = (
            row.get("target_quality_source") == "native_keep"
            or row.get("diagnosis_status") == "passed"
        )
        return ROLE_R_KEEP if kept else ROLE_R_EDIT

    def role_pools(self):
        """One sampling pool per role present in this cache."""
        grouped = {}
        for index in self.train_indices:
            grouped.setdefault(self.rows[index]["role"], []).append(index)
        return {
            role: StatePairPool(self, role, grouped[role])
            for role in (ROLE_I_EDIT, ROLE_R_EDIT, ROLE_R_KEEP, ROLE_R_REPAIR, ROLE_R_ACTIVE)
            if grouped.get(role)
        }


class StatePairPool:
    """A role restricted to one cache: its indices, accepted times, and draws."""

    def __init__(self, manifest, role, indices):
        self.manifest = manifest
        self.role = role
        self.indices = list(indices)
        self.by_time = {key: [] for key in manifest.by_time}
        for index in self.indices:
            for key, item in manifest.rows[index]["states"].items():
                if item["accepted"] and key in self.by_time:
                    self.by_time[key].append(index)

    @property
    def count(self):
        return len(self.indices)

    def available_times(self):
        return [
            float(t) for t in self.manifest.cfg["state_times"]
            if self.by_time[time_key(t)]
        ]

    def choose_time(self, rng, time=None):
        available = self.available_times()
        if not available:
            raise ValueError("No accepted state time in " + self.role)
        if time is not None:
            if float(time) not in available:
                raise ValueError("The selected state time has no accepted label")
            return float(time)
        configured = [float(t) for t in self.manifest.cfg["state_times"]]
        probs = [
            float(self.manifest.cfg["state_time_probs"][configured.index(value)])
            for value in available
        ]
        return weighted_choice(rng, available, probs, total=sum(probs))

    def choose_index(self, time, rng):
        indices = self.by_time[time_key(time)]
        if not indices:
            raise ValueError("No " + self.role + " record at the requested state time")
        return rng.choice(indices)

    def sample(self, index, time, keep=False):
        return self.manifest.sample(index, time, keep=keep)


class TrainingSampler:
    """Role-aware source, time, and keep sampling.

    ``role_weights`` draws the four roles explicitly, so a changing cache
    acceptance rate cannot silently change the training task; keep records then
    come only from the R-keep pool. Without it, the legacy source-kind weights
    and ``keep_fraction`` apply unchanged.
    """

    def __init__(self, linear, inverted, native, cfg, rank=0):
        self.cfg = cfg
        self.rng = random.Random(int(cfg["seed"]) + int(rank))
        self.keep_draws = 0
        self.draws = 0
        roles = {key: float(value) for key, value in (cfg.get("role_weights") or {}).items()}
        if roles:
            self.pools = {}
            if linear is not None and linear.count:
                for role, pool in linear.role_pools().items():
                    if role == "linear" or not pool.count:
                        continue
                    self.pools[role] = pool
                # Legacy manifests carry no sub-role; keep them usable as L-edit.
                if ROLE_L_EDIT not in self.pools:
                    self.pools[ROLE_L_EDIT] = linear
            for manifest in (inverted, native):
                if manifest is not None:
                    self.pools.update(manifest.role_pools())
            missing = sorted(
                role for role, weight in roles.items() if weight > 0 and role not in self.pools
            )
            if missing:
                raise ValueError(
                    "Role weights ask for records this run does not have: " + ", ".join(missing)
                )
            weights = {role: weight for role, weight in roles.items() if weight > 0}
            total = sum(weights.values())
            self.weights = {role: weight / total for role, weight in weights.items()}
            self.configured = dict(roles)
            self.keep_fraction = 0.0
        else:
            self.pools = {
                LINEAR_PAIR: linear,
                INVERTED_PAIR: inverted,
                NATIVE_PARENT: native,
            }
            configured = {k: float(v) for k, v in cfg["source_weights"].items()}
            weights = {
                kind: value for kind, value in configured.items()
                if value > 0 and self.pools.get(kind) is not None and self.pools[kind].count > 0
            }
            if not weights:
                raise ValueError("No configured source has a usable training record")
            if int(cfg["stage"]) == 2 and INVERTED_PAIR not in weights:
                raise ValueError("Stage 2 needs inverted pairs for the requested mix")
            if int(cfg["stage"]) == 2 and configured.get(NATIVE_PARENT, 0) > 0 and NATIVE_PARENT not in weights:
                # The plan's explicit fallback when R is not ready.
                weights = {LINEAR_PAIR: 0.2, INVERTED_PAIR: 0.8}
            total = sum(weights.values())
            self.weights = {kind: value / total for kind, value in weights.items()}
            self.configured = dict(configured)
            self.keep_fraction = float(cfg["keep_fraction"])
        self.seen = set()

    @property
    def configured_weights(self):
        return dict(self.configured)

    @property
    def actual_weights(self):
        return dict(self.weights)

    def input_domains(self):
        domains = []
        for key in (*ROLES, LINEAR_PAIR, INVERTED_PAIR, NATIVE_PARENT):
            if key in self.seen:
                domain = ROLE_DOMAIN.get(key) or DOMAIN[key]
                if domain not in domains:
                    domains.append(domain)
        return domains

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
        self.seen.add(role)
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
