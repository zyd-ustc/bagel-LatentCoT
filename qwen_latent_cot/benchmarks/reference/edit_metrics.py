"""Offline MagicBrush native-tool metrics / explicitly INTERNAL Emu reproduction.

Default dry-run reads bindings/coverage but imports no torch and writes nothing.
Example (add explicit local code/weight paths and hashes to actually execute):
  python tool_metrics.py --prepared .../magicbrush/prepared_single_v1 \
    --arm-root native=/sfs/native --arm-root pe_edit=/sfs/threeway \
    --arm-root unimoe_decode=/sfs/threeway \
    --captions /sfs/magicbrush/source/test_metadata/local_descriptions.json \
    --caption-kind local --out /sfs/new_scoring_namespace

No network, torch.hub, paid API, model downloads or generation. Reference roles:
MagicBrush generated vs GT TARGET; Emu generated vs INPUT SOURCE, plus captions.
Only independent single-turn MagicBrush is implemented; no fake sequential score.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

SCHEMA = "editing7_tool_metrics_v1"
ARMS = ("native", "pe_edit", "unimoe_decode")
COUNTS = {"magicbrush": 1053, "emu_edit": 3589}
MAGIC_REV = "d2d7f84d71ff23154e3f16f0c8f19392782bd201"
MAGIC_CODE_SHA = "6e14b77173901340b0ea7dc9a20e13724e20753dc435e8e7c7d13103172d3984"
METRICS = {
    "magicbrush": ("l1", "l2_mse", "clip_i", "dino", "clip_t", "clip_t_oracle"),
    "emu_edit": ("l1", "clip_im", "dino", "clip_out", "clip_dir"),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def freeze(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value) + b"\n"
    if path.exists():
        require(path.read_bytes() == data, "Refuse changed immutable output: " + str(path))
        return
    fd, tmp = tempfile.mkstemp(prefix=".metrics-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def task_key(row):
    return digest({k: row[k] for k in ("benchmark", "split", "id")})


def code_tree_hash(root):
    root = Path(root).resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".py", ".gz"))
    require(files, "Empty local encoder source tree")
    return digest([{ "path": p.relative_to(root).as_posix(), "sha256": sha(p)} for p in files])


def pixel_metrics(generated, reference):
    """Official MagicBrush L1Loss / MSELoss; RGB PIL default resize=BICUBIC."""
    with Image.open(generated) as g, Image.open(reference) as r:
        r = r.convert("RGB")
        g = g.convert("RGB").resize(r.size, Image.Resampling.BICUBIC)
        a, b = np.asarray(g, dtype=np.float32) / 255, np.asarray(r, dtype=np.float32) / 255
    difference = a - b
    return {"l1": float(np.mean(np.abs(difference), dtype=np.float32)),
            "l2_mse": float(np.mean(difference * difference, dtype=np.float32))}


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    require(vector.size > 0 and np.isfinite(vector).all(), "Invalid encoder features")
    length = float(np.linalg.norm(vector))
    return None if length < 1e-12 else vector / length


def cosine(a, b):
    a, b = unit(a), unit(b)
    if a is None or b is None:
        return None
    require(a.shape == b.shape, "Feature dimensionality mismatch")
    return float(np.clip(np.dot(a, b), -1, 1))


def direction_score(source_image, generated_image, input_text, output_text):
    """Explicit INTERNAL Emu choice: zero deltas are 0 + flag, not missing=0."""
    vectors = [unit(v) for v in (source_image, generated_image, input_text, output_text)]
    if any(v is None for v in vectors):
        return None, "invalid_zero_encoder_feature"
    require(len({v.shape for v in vectors}) == 1, "Direction feature dimensions differ")
    value = cosine(vectors[1] - vectors[0], vectors[3] - vectors[2])
    return (0.0, "degenerate_direction_zero_delta") if value is None else (value, None)


def asset_path(value):
    if isinstance(value, dict) and "bytes" in value:
        value = value["bytes"]
    require(isinstance(value, dict) and "embedded_asset" in value, "Missing preserved reference image")
    asset = value["embedded_asset"]
    path = Path(asset["path"])
    require(path.is_file() and sha(path) == asset["sha256"], "Reference image SHA mismatch")
    return str(path), asset["sha256"]


def reference_for(row, caption_dict=None):
    path = Path(row["evaluation_metadata"])
    require(sha(path) == row["evaluation_metadata_sha256"], "Evaluation metadata SHA mismatch")
    metadata = read(path)
    require(metadata["record_id"] == row["record_id"] and metadata["benchmark"] == row["benchmark"],
            "Evaluation metadata belongs to another case")
    data = metadata["data"]
    if row["benchmark"] == "magicbrush":
        reference, reference_sha = asset_path(data["target_img"])
        session, turn = str(row["session_id"]), int(row["turn_index"])
        name = f"{session}-output{turn}.png"
        caption = (caption_dict or {}).get(session, {}).get(name)
        return {"reference_image": reference, "reference_sha256": reference_sha, "role": "GT_target",
                "output_caption": caption if isinstance(caption, str) and caption.strip() else None,
                "caption_key": [session, name]}
    require(row["benchmark"] == "emu_edit", "Unsupported benchmark")
    require(sha(row["source_image"]) == row["source_sha256"], "Emu original source SHA mismatch")
    return {"reference_image": row["source_image"], "reference_sha256": row["source_sha256"],
            "role": "input_source_NOT_edited_GT", "input_caption": data.get("input_caption"),
            "output_caption": data.get("output_caption")}


def prediction_for(row, arm, root):
    """Consume exact worker envelopes, not arbitrary loose generated PNG files."""
    require(arm in ARMS, "Unsupported arm")
    folder = Path(root).resolve() / "cases" / task_key(row)
    if arm != "native":
        folder /= arm
    path = folder / "SUCCESS.json"
    if not path.exists():
        return {"status": "pending", "reason": "no_success_receipt"}
    try:
        value = read(path)
        expected_schema = "editing7_single_native_worker_v1" if arm == "native" else "editing7_three_arm_stage_worker_v1"
        # Match schemas via the worker's stable semantic envelope, never a filename-only success.
        require(value.get("status") == "ok" and value.get("schema") == expected_schema, "Wrong success schema/status")
        require(value.get("arm") == "native" if arm == "native" else value.get("stage") == arm, "Wrong arm/stage")
        require(value.get("task_key") == task_key(row) and value.get("row_sha256") == digest(row), "Receipt row binding changed")
        require(value.get("source_sha256") == row["source_sha256"] and
                value.get("condition_sha256") == row["condition_sha256"], "Actual source/condition binding changed")
        science_path = folder / "execution/SUCCESS.json"
        expected_science_sha = value.get("adapter_receipt_sha256") if arm == "native" else value.get("science_receipt", {}).get("sha256")
        require(sha(science_path) == expected_science_sha, "Scientific receipt changed")
        science = read(science_path)
        require(value["output"] == science["output"], "Output envelope differs from scientific record")
        output = value["output"]
        image = folder / "execution/image.png"
        require(Path(output["path"]) == image and not image.is_symlink() and sha(image) == output["sha256"], "Generated PNG SHA/path mismatch")
        with Image.open(image) as opened:
            opened.verify()
        return {"status": "ok", "image": str(image), "sha256": output["sha256"],
                "receipt": str(path), "receipt_sha256": sha(path), "science_receipt_sha256": expected_science_sha}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"status": "held_integrity", "reason": str(error)}


class LocalEncoders:
    """Pinned local OpenAI CLIP and DINO source/weights. Never torch.hub/download."""
    def __init__(self, args):
        for name in ("clip", "dino"):
            root, weights = Path(getattr(args, name + "_code")).resolve(), Path(getattr(args, name + "_weights")).resolve()
            require(root.is_dir() and weights.is_file(), "Explicit local source/weight paths required")
            require(code_tree_hash(root) == getattr(args, name + "_code_sha256"), "Encoder code tree drift")
            require(sha(weights) == getattr(args, name + "_weights_sha256"), "Encoder checkpoint SHA drift")
        import torch
        import torchvision
        from torchvision import transforms
        self.torch, self.device = torch, args.device
        if args.device.startswith("npu"):
            import torch_npu  # noqa: F401; only explicit execute can reach this branch.
        clip_root = Path(args.clip_code).resolve()
        sys.path.insert(0, str(clip_root))
        import clip
        require(Path(clip.__file__).resolve().is_relative_to(clip_root), "Foreign CLIP module already loaded")
        self.clip_module = clip
        self.clip, self.clip_transform = clip.load(str(Path(args.clip_weights).resolve()), device=args.device, jit=False)
        require(tuple(self.clip.visual.conv1.kernel_size) == (32, 32) and
                self.clip.visual.conv1.out_channels == 768 and self.clip.visual.input_resolution == 224,
                "Checkpoint is not CLIP ViT-B/32")
        self.clip.eval()
        dino_root = Path(args.dino_code).resolve()
        if "utils" in sys.modules:
            require(Path(sys.modules["utils"].__file__).resolve().is_relative_to(dino_root), "Foreign DINO utils module loaded")
        sys.path.insert(0, str(dino_root))
        dino_file = dino_root / "vision_transformer.py"
        spec = importlib.util.spec_from_file_location("_editing7_pinned_dino_vit", dino_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.dino = module.vit_small(patch_size=16, num_classes=0)
        state = torch.load(args.dino_weights, map_location="cpu", weights_only=True)
        require(isinstance(state, dict) and state and all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()),
                "Expected official plain DINO vits16 state dict, not arbitrary pickle or guessed prefixes")
        self.dino.load_state_dict(state, strict=True)
        self.dino.eval().to(args.device)
        self.dino_transform = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.versions = {"torch": torch.__version__, "torchvision": torchvision.__version__, "pillow": PILLOW_VERSION,
                         "numpy": np.__version__, "device": args.device}

    def image(self, path, encoder):
        with Image.open(path) as image:
            tensor = (self.clip_transform if encoder == "clip" else self.dino_transform)(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            vector = self.clip.encode_image(tensor) if encoder == "clip" else self.dino(tensor)
        return vector.detach().cpu().float().numpy().reshape(-1)

    def text(self, value):
        require(isinstance(value, str) and value.strip(), "Missing caption")
        # Official MagicBrush clip.tokenize does not truncate: overlong captions
        # are pending, never silently alter the text or assign zero.
        tokens = self.clip_module.tokenize(value, truncate=False).to(self.device)
        with self.torch.no_grad():
            vector = self.clip.encode_text(tokens)
        return vector.detach().cpu().float().numpy().reshape(-1)


def score_case(benchmark, generated, reference, encoders):
    scores = pixel_metrics(generated, reference["reference_image"])
    if benchmark == "emu_edit":
        scores.pop("l2_mse")
    errors, flags = {}, []
    features = {}
    for encoder in ("clip", "dino"):
        try:
            g = encoders.image(generated, encoder)
            r = encoders.image(reference["reference_image"], encoder)
            metric = "dino" if encoder == "dino" else ("clip_i" if benchmark == "magicbrush" else "clip_im")
            value = cosine(g, r)
            if value is None:
                errors[metric] = "zero_encoder_feature"
            else:
                scores[metric] = value
            features[encoder] = (g, r)
        except (ValueError, RuntimeError) as error:
            errors[encoder] = str(error)
    text_key = "clip_t" if benchmark == "magicbrush" else "clip_out"
    try:
        out = encoders.text(reference.get("output_caption"))
        g, r = features["clip"]
        value = cosine(g, out)
        if value is not None:
            scores[text_key] = value
        else:
            errors[text_key] = "zero_encoder_feature"
        if benchmark == "magicbrush":
            oracle = cosine(r, out)
            if oracle is not None:
                scores["clip_t_oracle"] = oracle
        else:
            source = encoders.text(reference.get("input_caption"))
            direction, flag = direction_score(r, g, source, out)
            if direction is not None:
                scores["clip_dir"] = direction
            if flag:
                flags.append(flag)
    except (ValueError, RuntimeError, KeyError) as error:
        errors[text_key + "_or_direction"] = str(error)
    for metric in METRICS[benchmark]:
        scores.setdefault(metric, None)
    return {"metrics": scores, "metric_errors": errors, "flags": flags}


def summarize(percase, benchmark, expected_ids, arms=ARMS):
    ids = list(expected_ids)
    require(len(ids) == len(set(ids)), "Duplicate expected IDs")
    keyed = {(r["id"], r["arm"]): r for r in percase}
    require(len(keyed) == len(percase), "Duplicate case/arm results")
    require(all(i in ids and a in arms for i, a in keyed), "Foreign score record")
    result = {"expected_cases": len(ids), "arms": {}, "paired_three_arm": {}, "no_combined_total": True}
    for arm in arms:
        metrics = {}
        for metric in METRICS[benchmark]:
            values = [keyed.get((i, arm), {}).get("metrics", {}).get(metric) for i in ids]
            valid = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
            metrics[metric] = {"valid": len(valid), "pending": len(ids) - len(valid),
                "available_subset_mean": sum(valid) / len(valid) if valid else None,
                "full_split_mean": sum(valid) / len(valid) if valid and len(valid) == len(ids) else None}
        result["arms"][arm] = metrics
    for metric in METRICS[benchmark]:
        common = [i for i in ids if all(isinstance(keyed.get((i, a), {}).get("metrics", {}).get(metric), (int, float))
            and math.isfinite(keyed[(i, a)]["metrics"][metric]) for a in arms)]
        result["paired_three_arm"][metric] = {"count": len(common), "ids_sha256": digest(common),
            "means": {a: sum(keyed[(i, a)]["metrics"][metric] for i in common) / len(common) if common else None for a in arms},
            "scope": "same case IDs across all arms; never inferred from differently missing subsets"}
    return result


def load_prepared(directory):
    directory = Path(directory).resolve()
    ready = read(directory / "READY.json")
    benchmark = ready["benchmark"]
    require(benchmark in COUNTS and ready.get("status") == "READY" and ready.get("complete") is True and
            ready.get("count") == COUNTS[benchmark], "Not a complete supported evaluation split")
    manifest = directory / "MANIFEST.jsonl"
    require(ready["manifest"] == str(manifest) and sha(manifest) == ready["manifest_sha256"], "Prepared manifest SHA drift")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    require(len(rows) == COUNTS[benchmark] and len({r["id"] for r in rows}) == len(rows), "Wrong full count/duplicate IDs")
    require(all(r["benchmark"] == benchmark and (benchmark != "magicbrush" or
                r.get("input_mode") == "independent_turn_gt_source_mask_free") for r in rows), "Wrong benchmark/input mode")
    return ready, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--arm-root", action="append", required=True, help="native=PATH, pe_edit=PATH, unimoe_decode=PATH")
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--caption-kind", choices=("local", "global"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--execute", action="store_true", help="Otherwise read-only/no model imports")
    parser.add_argument("--allow-emu-internal", action="store_true")
    parser.add_argument("--official-code", type=Path, help="Local pinned MagicBrush evaluation/image_eval.py")
    for encoder in ("clip", "dino"):
        parser.add_argument("--" + encoder + "-code", type=Path)
        parser.add_argument("--" + encoder + "-code-sha256")
        parser.add_argument("--" + encoder + "-weights", type=Path)
        parser.add_argument("--" + encoder + "-weights-sha256")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    ready, rows = load_prepared(args.prepared)
    bench = ready["benchmark"]
    roots = {}
    for item in args.arm_root:
        arm, root = item.split("=", 1)
        require(arm in ARMS and arm not in roots and Path(root).is_absolute(), "Invalid/duplicate arm root")
        roots[arm] = root
    require(set(roots) == set(ARMS), "All three arms must be declared, even if currently pending")
    captions = None
    if bench == "magicbrush":
        require(args.captions and args.caption_kind and args.captions.name == args.caption_kind + "_descriptions.json",
                "Explicit official local/global caption file required")
        matching = [s for s in ready["source_files"] if Path(s["path"]) == args.captions.resolve()]
        require(len(matching) == 1 and sha(args.captions) == matching[0]["sha256"], "Caption file not bound to prepared source")
        captions = read(args.captions)
    references, predictions = {}, {}
    for row in rows:
        references[row["id"]] = reference_for(row, captions)
        for arm in ARMS:
            predictions[(row["id"], arm)] = prediction_for(row, arm, roots[arm])
    coverage = {arm: {status: sum(p["status"] == status for (i, a), p in predictions.items() if a == arm)
                     for status in ("ok", "pending", "held_integrity")} for arm in ARMS}
    if not args.execute:
        print(json.dumps({"mode": "dryrun", "benchmark": bench, "expected": len(rows), "coverage": coverage,
            "reference_role": "GT_target" if bench == "magicbrush" else "input_source",
            "model_loaded": False, "writes": 0, "emu_execution_requires_explicit_internal_opt_in": True}, ensure_ascii=False))
        return
    require(args.out and args.out.is_absolute() and not args.out.exists(), "Execute requires a fresh absolute --out namespace")
    require(bench != "emu_edit" or args.allow_emu_internal, "Emu exact official backbone not verified; explicit internal opt-in required")
    require(args.official_code and sha(args.official_code) == MAGIC_CODE_SHA, "Missing/wrong pinned MagicBrush metric source")
    for encoder in ("clip", "dino"):
        require(all(getattr(args, encoder + "_" + suffix) for suffix in ("code", "code_sha256", "weights", "weights_sha256")),
                "Explicit local encoder paths and hashes required")
    require(any(v["status"] == "ok" for v in predictions.values()), "No verified generated outputs; do not load models")
    models = LocalEncoders(args)
    method = {"schema": SCHEMA, "protocol_id": "magicbrush_single_native_tools_v1" if bench == "magicbrush" else "emu_tools_INTERNAL_v1",
        "not_official_exact_emu": bench == "emu_edit", "benchmark": bench, "expected": len(rows),
        "manifest_sha256": ready["manifest_sha256"], "prepared_ready_sha256": sha(args.prepared / "READY.json"),
        "metric_source_revision": MAGIC_REV, "metric_source_sha256": MAGIC_CODE_SHA,
        "reference_role": "GT_target" if bench == "magicbrush" else "input_source_NOT_edited_GT",
        "caption_kind": args.caption_kind if bench == "magicbrush" else "published input_caption/output_caption",
        "caption_sha256": sha(args.captions) if args.captions else None,
        "pixel_resize": "RGB BICUBIC to reference size; float32 [0,1]; L2 is MSE not RMSE",
        "clip": {"architecture": "OpenAI ViT-B/32", "code": str(args.clip_code), "code_tree_sha256": args.clip_code_sha256,
                 "weights": str(args.clip_weights), "weights_sha256": args.clip_weights_sha256, "tokenize_truncate": False},
        "dino": {"architecture": "dino_vits16", "code": str(args.dino_code), "code_tree_sha256": args.dino_code_sha256,
                 "weights": str(args.dino_weights), "weights_sha256": args.dino_weights_sha256,
                 "transform": "Resize256 BICUBIC,CenterCrop224,ToTensor,ImageNet mean/std"},
        "emu_direction_internal_choice": "cos(unit(g)-unit(source),unit(outcaption)-unit(incaption)); zero delta=0+flag",
        "missing_policy": "pending/null, never zero; full mean withheld until full metric denominator",
        "aggregation": "per-metric case micro mean; common IDs across all three arms separately; no combined total",
        "versions": models.versions, "runner_sha256": sha(__file__), "paid_API_calls": 0}
    freeze(args.out / "METHOD.json", method)
    percase = []
    for row in rows:
        reference = references[row["id"]]
        for arm in ARMS:
            prediction = predictions[(row["id"], arm)]
            result = {"id": row["id"], "arm": arm, "benchmark": bench, "task": row.get("task_type"),
                      "prediction": prediction, "reference": reference, "method_sha256": digest(method), "metrics": {}}
            if prediction["status"] == "ok":
                require(sha(prediction["image"]) == prediction["sha256"] and
                        sha(reference["reference_image"]) == reference["reference_sha256"], "Scoring input changed since preflight")
                result.update(score_case(bench, prediction["image"], reference, models))
                require(sha(prediction["image"]) == prediction["sha256"] and
                        sha(reference["reference_image"]) == reference["reference_sha256"], "Scoring input changed during metric computation")
            freeze(args.out / "cases" / arm / (task_key(row) + ".json"), result)
            percase.append(result)
    summary = summarize(percase, bench, [r["id"] for r in rows])
    summary.update(benchmark=bench, protocol_id=method["protocol_id"], coverage=coverage, method_sha256=digest(method))
    freeze(args.out / "SUMMARY.json", summary)
    print(json.dumps({"completed_scoring_pass": True, "benchmark": bench, "summary": str(args.out / "SUMMARY.json"),
                      "full_generation_coverage": all(v["ok"] == len(rows) for v in coverage.values())}))


if __name__ == "__main__":
    main()
