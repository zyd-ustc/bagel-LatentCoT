"""Produce inverted-pair and native-parent caches without any seed search."""

import hashlib
import json
import math
import re
import time
from pathlib import Path

import torch

from . import CACHE_KIND, INVERTED_PAIR, NATIVE_PARENT
from .data import LinearManifest, read_rows
from .diagnosis import review_preview
from .field import (
    NativeT2I, forward_native, inversion_grid, phase_grid, preview_image, protocol,
    reverse_heun_states, time_key,
)
from .images import load_pair, save_image
from .loss import build_perceptual, reconstruction_metrics
from .model import StateEditor


def _safe(uid):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(uid)).strip("._")
    digest = hashlib.sha1(str(uid).encode()).hexdigest()[:8]
    return (value or "uid") + "_" + digest


def _write_jsonl(path, row):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _state_path(output, uid, time):
    folder = Path(output) / "states" / _safe(uid)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{time_key(time)}.pt"
    return path


def _relative(output, path):
    return str(Path(path).resolve().relative_to(Path(output).resolve()))


def _save_payload(output, uid, time, payload):
    path = _state_path(output, uid, time)
    torch.save({key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in payload.items()}, path)
    return _relative(output, path)


def _recon_image(output, uid, time, side, image):
    if image is None or not bool(torch.isfinite(image).all()):
        return None
    folder = Path(output) / "reconstructions"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{_safe(uid)}_{time_key(time)}_{side}.png"
    save_image(path, image)
    return _relative(output, path)


def _perceptual(args, device):
    if args.no_lpips:
        return None
    return build_perceptual(device)


def _accepted(metrics, cfg, semantic_ok):
    inversion = cfg["inversion"]
    if not semantic_ok:
        return False
    l1 = float(metrics["l1"])
    if not math.isfinite(l1) or l1 > float(inversion["target_l1_max"]):
        return False
    if "lpips" in metrics:
        lpips = float(metrics["lpips"])
        if not math.isfinite(lpips) or lpips > float(inversion["target_lpips_max"]):
            return False
    return True


def _common_row(row, source_kind):
    return dict(
        uid=str(row["uid"]),
        source_kind=source_kind,
        split=row["split"],
        source_family=row.get("source_family"),
        prompt_family=row.get("prompt_family"),
        source_image=row.get("source_image"),
        target_image=row.get("target_image"),
        edit_instruction=row["edit_instruction"],
        parent_condition=row["parent_condition"],
        tail_condition=row["tail_condition"],
        preserve=row.get("preserve", ""),
        condition_protocol=row["condition_protocol"],
        tail_source=row.get("tail_source", "declared"),
        semantic_quality_source=row.get("semantic_quality_source"),
        target_quality_source=row.get("target_quality_source"),
    )


def _cache_header(rt, cfg, command, args):
    return dict(
        kind=CACHE_KIND,
        protocol=protocol(rt, cfg),
        command=command,
        native_model_id=str(rt.identity.get("model")),
        noise_scale=rt.noise_scale(),
        cfg=float(cfg["cfg_scale"]),
        cfg_norm=cfg["cfg_norm"],
        t_eps=float(cfg["t_eps"]),
        forward_grid=[float(t) for t in phase_grid(rt, cfg)],
        forward_solver="euler",
        inversion_grid=[float(t) for t in inversion_grid(rt, cfg, int(cfg["inversion"]["substeps"]))],
        inversion_solver="reverse_heun",
        preview_protocol=cfg.get("preview_protocol", "heun_then_euler_from_state_v1"),
        source_seed_or_linear_noise_seed=args.seed if getattr(args, "seed", None) is not None else "none",
        semantic_quality_source="existing_pair_approval",
        target_quality_source="target_approved",
    )


def _write_manifest(output, header, rows, shard):
    header = dict(header)
    header["records"] = rows
    header["data_version"] = hashlib.sha256(
        json.dumps(
            [(row["uid"], sorted(row["states"]), {k: v["accepted"] for k, v in row["states"].items()}) for row in rows],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    name = "manifest.json" if shard is None else f"manifest.shard_{shard:03d}.json"
    path = Path(output) / name
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(header, ensure_ascii=False) + "\n")
    partial.replace(path)
    return path


def produce_inverted(rt, cfg, args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = LinearManifest(args.linear, rt, cfg)
    grid = phase_grid(rt, cfg)
    state_times = [float(t) for t in cfg["state_times"]]
    substeps = int(cfg["inversion"]["substeps"])
    perceptual = _perceptual(args, rt.device)
    editor = StateEditor(rt, cfg).eval() if args.cache_base_prediction else None
    header = _cache_header(rt, cfg, "prepare-state-pairs/inverted", args)
    rows = []
    start, stop, stride = _shard_slice(len(manifest.rows), args)
    progress = output / "progress"
    progress.mkdir(exist_ok=True)
    log = output / "production.jsonl"
    for index in range(start, stop, stride):
        row = manifest.rows[index]
        marker = progress / f"{_safe(row['uid'])}.json"
        if args.skip_existing and marker.exists():
            rows.append(json.loads(marker.read_text()))
            continue
        began = time.perf_counter()
        source, target, valid, rect = load_pair(row, rt.width, rt.height, rt.device)
        parent = NativeT2I(rt, row["parent_condition"], cfg)
        tail = NativeT2I(rt, row["tail_condition"], cfg)
        source_states = reverse_heun_states(
            source, 1.0, cfg["state_times"], grid, parent, substeps
        )
        target_states = reverse_heun_states(
            target, 1.0, cfg["state_times"], grid, tail, substeps
        )
        states = {}
        source_metrics, target_metrics = {}, {}
        for time in state_times:
            key = time_key(time)
            anchor = source_states[key]
            destination = target_states[key]
            preview = preview_image(anchor, time, grid, parent)
            source_recon = forward_native(anchor, time, grid, parent)
            target_recon = forward_native(destination, time, grid, tail)
            source_metrics[key] = reconstruction_metrics(source_recon, source, valid, perceptual)
            target_metrics[key] = reconstruction_metrics(target_recon, target, valid, perceptual)
            accepted = (
                _accepted(target_metrics[key], cfg, bool(row.get("target_quality_source")))
                and bool(torch.isfinite(anchor).all())
                and bool(torch.isfinite(destination).all())
                and bool(torch.isfinite(preview).all())
            )
            base = None
            if editor is not None:
                base = editor.image_clean(anchor, time, editor.image_prefix(preview)["image"])
            payload = dict(
                time=float(time),
                state_a=anchor.float().cpu(),
                state_b=destination.float().cpu(),
                preview=preview.float().cpu(),
                base_prediction=None if base is None else base.float().cpu(),
                valid=valid.float().cpu(),
                source_reconstruction=source_metrics[key],
                target_reconstruction=target_metrics[key],
                accepted=accepted,
            )
            state_file = _save_payload(output, row["uid"], time, payload)
            states[key] = dict(
                path=state_file,
                accepted=accepted,
                source_reconstruction=source_metrics[key],
                target_reconstruction=target_metrics[key],
                source_reconstruction_image=_recon_image(output, row["uid"], time, "source", source_recon),
                target_reconstruction_image=_recon_image(output, row["uid"], time, "target", target_recon),
            )
        record = _common_row(row, INVERTED_PAIR)
        record.update(
            valid_region=list(rect),
            states=states,
            source_seed_or_linear_noise_seed="none",
            source_reconstruction_metrics=source_metrics,
            target_reconstruction_metrics=target_metrics,
        )
        marker.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        rows.append(record)
        _write_jsonl(
            log,
            dict(
                uid=row["uid"],
                accepted={k: v["accepted"] for k, v in states.items()},
                seconds=time.perf_counter() - began,
            ),
        )
    path = _write_manifest(output, header, rows, args.shard_index if args.shard_count > 1 else None)
    return path


def _load_targets(path):
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return {str(row["uid"]): row for row in data}
    if isinstance(data, dict) and "records" in data:
        return {str(row["uid"]): row for row in data["records"]}
    if isinstance(data, dict):
        return {str(key): value for key, value in data.items()}
    raise ValueError("Target manifest must be a mapping, list, or records object")


def produce_native_parent(rt, cfg, args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prompts = read_rows(args.prompts)
    targets = _load_targets(args.targets)
    grid = phase_grid(rt, cfg)
    state_times = [float(t) for t in cfg["state_times"]]
    supported = {time_key(t) for t in state_times}
    read = float(cfg["read_time"])
    substeps = int(cfg["inversion"]["substeps"])
    perceptual = _perceptual(args, rt.device)
    editor = StateEditor(rt, cfg).eval() if args.cache_base_prediction else None
    header = _cache_header(rt, cfg, "prepare-state-pairs/native-parent", args)
    rows = []
    start, stop, stride = _shard_slice(len(prompts), args)
    progress = output / "progress"
    progress.mkdir(exist_ok=True)
    log = output / "production.jsonl"
    for index in range(start, stop, stride):
        source = prompts[index]
        raw = source.get("raw_prompt") or source.get("prompt")
        if not raw:
            raise ValueError("A native-parent prompt is missing")
        uid = str(source.get("uid") or f"{index:05d}")
        marker = progress / f"{_safe(uid)}.json"
        if args.skip_existing and marker.exists():
            rows.append(json.loads(marker.read_text()))
            continue
        began = time.perf_counter()
        split = source.get("split", "train")
        seed = int(source.get("seed", int(cfg["seed"]) + index))
        generator = torch.Generator(device=rt.device).manual_seed(seed)
        state = torch.randn((1, 3, rt.height, rt.width), device=rt.device, generator=generator) * rt.noise_scale()
        native = NativeT2I(rt, raw, cfg)
        states, previews = {}, {}
        for left, right in zip(grid[:-1], grid[1:]):
            if abs(float(left) - 1.0) < 1e-7:
                continue
            key = time_key(left)
            if key in supported:
                states[key] = state.detach().float().clone()
                previews[key] = preview_image(state, left, grid, native)
            step = float(right) - float(left)
            state = state + step * native(state, state.new_tensor(float(left)))
        if not bool(torch.isfinite(state).all()):
            _write_jsonl(log, dict(uid=uid, skipped="nonfinite_parent"))
            continue
        final_path = output / "targets" / f"{_safe(uid)}_source_final.png"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(final_path, state)

        review = review_preview(rt, previews[time_key(read)], raw, cfg)
        edit_request, preserve, failed = (
            review["edit_request"], review["preserve"], review["failed"],
        )
        target_item = targets.get(uid)
        if isinstance(target_item, str):
            target_item = {"target_image": target_item}
        use_keep = not edit_request
        if not use_keep:
            if target_item is None:
                _write_jsonl(log, dict(uid=uid, skipped="target_missing", failed=failed))
                continue
            if not bool(target_item.get("semantic_approved", True)):
                _write_jsonl(log, dict(uid=uid, skipped="target_rejected", failed=failed))
                continue

        if use_keep:
            edit = cfg["keep_instruction"]
            target_pixels = target_mask = target_rect = None
            tail_states = dict(states)
            tail_native = None
            target_source = "native_keep"
        else:
            edit = edit_request
            target_path = target_item.get("target_image")
            if not target_path:
                _write_jsonl(log, dict(uid=uid, skipped="target_path_missing", failed=failed))
                continue
            resolved_target = (
                Path(target_path) if Path(target_path).is_absolute()
                else Path(args.targets).parent / target_path
            )
            _, target_pixels, target_mask, target_rect = load_pair(
                {"source_image": str(final_path), "target_image": str(resolved_target)},
                rt.width,
                rt.height,
                rt.device,
            )
            tail_native = NativeT2I(rt, raw, cfg)
            tail_states = reverse_heun_states(
                target_pixels, 1.0, cfg["state_times"], grid, tail_native, substeps
            )
            target_source = target_item.get("target_quality_source", "external_editor")

        record = dict(
            uid=uid,
            source_kind=NATIVE_PARENT,
            split=split,
            source_family=source.get("source_family", uid),
            prompt_family=source.get("prompt_family", uid),
            source_image=_relative(output, final_path),
            target_image=target_item.get("target_image") if (target_item and not use_keep) else None,
            edit_instruction=edit,
            parent_condition=raw,
            tail_condition=raw,
            preserve=preserve if not use_keep else "",
            condition_protocol=cfg["condition_protocol"],
            valid_region=list(target_rect) if target_rect else None,
            source_seed_or_linear_noise_seed=seed,
            states={},
            source_reconstruction_metrics={},
            target_reconstruction_metrics={},
            semantic_quality_source="native_forward",
            target_quality_source=target_source,
        )
        for time in state_times:
            key = time_key(time)
            anchor = states[key]
            destination = anchor if use_keep else tail_states[key]
            valid = (
                torch.ones(1, 1, rt.height, rt.width, device=rt.device)
                if target_mask is None
                else target_mask
            )
            target_metrics = {}
            accepted = (
                use_keep
                and bool(torch.isfinite(anchor).all())
                and bool(torch.isfinite(previews[key]).all())
            )
            if not use_keep:
                recon = forward_native(destination, time, grid, tail_native)
                target_metrics = reconstruction_metrics(recon, target_pixels, valid, perceptual)
                accepted = (
                    _accepted(target_metrics, cfg, True)
                    and bool(torch.isfinite(anchor).all())
                    and bool(torch.isfinite(destination).all())
                    and bool(torch.isfinite(previews[key]).all())
                )
                target_recon = recon
            else:
                target_recon = None
            base = None
            if editor is not None:
                base = editor.image_clean(anchor, time, editor.image_prefix(previews[key])["image"])
            payload = dict(
                time=float(time),
                state_a=anchor.float().cpu(),
                state_b=destination.float().cpu(),
                preview=previews[key].float().cpu(),
                base_prediction=None if base is None else base.float().cpu(),
                valid=valid.float().cpu(),
                source_reconstruction={},
                target_reconstruction=target_metrics,
                accepted=bool(accepted),
            )
            state_file = _save_payload(output, uid, time, payload)
            record["states"][key] = dict(
                path=state_file,
                accepted=bool(accepted),
                source_reconstruction={},
                target_reconstruction=target_metrics,
                source_reconstruction_image=None,
                target_reconstruction_image=_recon_image(output, uid, time, "target", target_recon),
            )
            record["target_reconstruction_metrics"][key] = target_metrics
        marker.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        rows.append(record)
        _write_jsonl(
            log,
            dict(
                uid=uid,
                keep=use_keep,
                accepted={k: v["accepted"] for k, v in record["states"].items()},
                seconds=time.perf_counter() - began,
            ),
        )
    path = _write_manifest(output, header, rows, args.shard_index if args.shard_count > 1 else None)
    return path


def _shard_slice(total, args):
    count = int(getattr(args, "shard_count", 1) or 1)
    index = int(getattr(args, "shard_index", 0) or 0)
    if count < 1 or not 0 <= index < count:
        raise ValueError("Invalid shard index/count")
    limit = getattr(args, "limit", None)
    if limit is not None:
        total = min(int(total), int(limit))
    return index, total, count
