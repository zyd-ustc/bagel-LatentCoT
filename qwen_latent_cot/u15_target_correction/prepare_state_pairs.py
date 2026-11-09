"""Produce inverted-pair and native-parent caches without any seed search."""

import hashlib
import json
import os
import math
import re
import time
from pathlib import Path

import torch

from . import (
    CACHE_KIND, INVERTED_PAIR, NATIVE_PARENT,
    ROLE_I_EDIT, ROLE_R_EDIT, ROLE_R_KEEP,
)
from .conditions import compact
from .data import LinearManifest, read_rows
from .diagnosis import review_preview
from .labels import (
    effective_acceptance as _effective_acceptance,
    high_frequency as _high_frequency,
)
from .field import (
    NativeT2I, discrete_inverse_states, forward_native, inversion_endpoint,
    inversion_grid, inversion_solver, native_forward_states, phase_grid,
    preview_image, protocol, reverse_heun_states, time_key,
)
from .images import load_pair, save_image
from .loss import build_perceptual, reconstruction_metrics
from .model import StateEditor


def _safe(uid):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(uid)).strip("._")
    digest = hashlib.sha1(str(uid).encode()).hexdigest()[:8]
    return (value or "uid") + "_" + digest


def _write_jsonl(path, row):
    """Append one line with a single unbuffered write.

    Several shards share these files, and buffered ``open("a")`` splits writes
    on a network filesystem, which interleaves lines from different shards.
    """
    data = (json.dumps(row, ensure_ascii=False) + "\n").encode()
    handle = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(handle, data)
    finally:
        os.close(handle)


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


def _invert(state, grid, velocity, cfg, substeps, endpoint):
    """Label states for one side with the configured backward scheme."""
    times = cfg["state_times"]
    if inversion_solver(cfg) == "discrete_fixpoint":
        snapshots, diagnostics = discrete_inverse_states(
            state, grid, times, velocity,
            iterations=int(cfg["inversion"]["iterations"]),
            damping=float(cfg["inversion"]["damping"]),
            initial=reverse_heun_states(state, 1.0, times, grid, velocity, substeps, endpoint=endpoint),
        )
        return snapshots, diagnostics
    snapshots = reverse_heun_states(state, 1.0, times, grid, velocity, substeps, endpoint=endpoint)
    diagnostics = {
        key: dict(residual=None, segment_residual=None, solver="reverse_heun")
        for key in snapshots
    }
    return snapshots, diagnostics


def _rejection(metrics, cfg, semantic_ok, *, source=False, residual=None, margin_ratio=None):
    """Why a reconstruction fails its cache gate; ``None`` means it passes.

    The first gate is convergence: the label is only meaningful if the fixed
    point satisfied the deployment Euler relation. Reconstruction caps come
    after that, and inverted pairs are gated on both sides.
    """
    inversion = cfg["inversion"]
    side = "source" if source else "target"
    if residual is not None and float(residual) > float(inversion.get("max_residual", 1.0)):
        return side + "_inversion"
    if not semantic_ok:
        return side + "_semantics"
    # The folded pair criterion is available but switched off by default
    # (``margin_fraction = 0``): while it is off, the absolute caps below apply.
    needed = float(inversion.get("margin_fraction", 0.0))
    if needed > 0:
        if margin_ratio is None:
            return side + "_margin_missing"
        return None if float(margin_ratio) >= needed else side + "_margin"
    l1 = float(metrics.get("l1", math.inf))
    if not math.isfinite(l1) or l1 > float(inversion[side + "_l1_max"]):
        return side + "_l1"
    if not source and "lpips" in metrics:
        lpips = float(metrics["lpips"])
        if not math.isfinite(lpips) or lpips > float(inversion["target_lpips_max"]):
            return side + "_lpips"
    return None


def _accepted(metrics, cfg, semantic_ok):
    return _rejection(metrics, cfg, semantic_ok) is None


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
        geometry=row.get("geometry"),
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
        inversion_solver="discrete_euler_fixpoint",
        inversion_iterations=int(cfg["inversion"]["iterations"]),
        inversion_damping=float(cfg["inversion"]["damping"]),
        inversion_residual_max=float(cfg["inversion"]["max_residual"]),
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
    endpoint = inversion_endpoint(cfg)
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
        source_states, source_inversion = _invert(
            source, grid, parent, cfg, substeps, endpoint
        )
        target_states, target_inversion = _invert(
            target, grid, tail, cfg, substeps, endpoint
        )
        states = {}
        source_metrics, target_metrics, gates = {}, {}, {}
        for state_time in state_times:
            key = time_key(state_time)
            anchor = source_states[key]
            destination = target_states[key]
            preview = preview_image(anchor, state_time, grid, parent)
            source_recon = forward_native(anchor, state_time, grid, parent)
            target_recon = forward_native(destination, state_time, grid, tail)
            source_metrics[key] = reconstruction_metrics(source_recon, source, valid, perceptual)
            target_metrics[key] = reconstruction_metrics(target_recon, target, valid, perceptual)
            first = source_inversion[key].get("segment_residual")
            second = target_inversion[key].get("segment_residual")
            residual = None if first is None and second is None else max(
                value for value in (first, second) if value is not None
            )
            # Does the continued target state belong to the target or the source?
            against_source = reconstruction_metrics(target_recon, source, valid, None)
            pair_l1 = reconstruction_metrics(target, source, valid, None)["l1"]
            separation = float(against_source["l1"]) - float(target_metrics[key]["l1"])
            margin_ratio = separation / max(float(pair_l1), 1e-6)
            # The label is (a, b) with b defined by "continues to the target", so
            # the pair criterion applies to the target side; the source side is
            # recorded and its state plausibility comes from R data instead.
            reason = _rejection(
                target_metrics[key], cfg, bool(row.get("target_quality_source")),
                residual=target_inversion[key].get("segment_residual"),
                margin_ratio=margin_ratio,
            )
            finite = all(
                bool(torch.isfinite(value).all())
                for value in (anchor, destination, preview)
            )
            if reason is None and not finite:
                reason = "nonfinite_state"
            accepted = reason is None
            gates[key] = dict(
                accepted=accepted,
                reason=reason,
                inversion_residual=residual,
                target_vs_source_l1=float(against_source["l1"]),
                pair_l1=float(pair_l1),
                margin_ratio=float(margin_ratio),
                inversion_node_residual=max(
                    [value for value in (
                        source_inversion[key].get("residual"),
                        target_inversion[key].get("residual"),
                    ) if value is not None] or [None],
                ),
                source_l1=source_metrics[key].get("l1"),
                target_l1=target_metrics[key].get("l1"),
                target_lpips=target_metrics[key].get("lpips"),
            )
            base = None
            if editor is not None:
                base = editor.image_clean(anchor, state_time, editor.image_prefix(preview)["image"])
            payload = dict(
                time=float(state_time),
                state_a=anchor.float().cpu(),
                state_b=destination.float().cpu(),
                preview=preview.float().cpu(),
                base_prediction=None if base is None else base.float().cpu(),
                valid=valid.float().cpu(),
                source_reconstruction=source_metrics[key],
                target_reconstruction=target_metrics[key],
                accepted=accepted,
            )
            state_file = _save_payload(output, row["uid"], state_time, payload)
            states[key] = dict(
                path=state_file,
                accepted=accepted,
                reject_reason=reason,
                inversion=dict(source=source_inversion[key], target=target_inversion[key]),
                source_reconstruction=source_metrics[key],
                target_reconstruction=target_metrics[key],
                source_reconstruction_image=_recon_image(output, row["uid"], state_time, "source", source_recon),
                target_reconstruction_image=_recon_image(output, row["uid"], state_time, "target", target_recon),
            )
        record = _common_row(row, INVERTED_PAIR)
        record["role"] = ROLE_I_EDIT
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
                task=row.get("task"),
                tail_source=row.get("tail_source"),
                accepted={k: value["accepted"] for k, value in gates.items()},
                gates=gates,
                seconds=time.perf_counter() - began,
            ),
        )
    path = _write_manifest(output, header, rows, args.shard_index if args.shard_count > 1 else None)
    return path


def _load_targets(path):
    """Targets from JSONL, a JSON list, a ``records`` object, or a uid mapping."""
    if not path:
        return {}
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return {str(row["uid"]): row for row in rows if isinstance(row, dict) and "uid" in row}
    if isinstance(data, list):
        return {str(row["uid"]): row for row in data}
    if isinstance(data, dict) and "records" in data:
        return {str(row["uid"]): row for row in data["records"]}
    if isinstance(data, dict):
        return {str(key): value for key, value in data.items()}
    raise ValueError("Target manifest must be a mapping, list, or records object")


def _target_binding_error(target_item, seed, parent_branch_id, edit_request):
    """A target must be approved and bound to this exact parent and request."""
    if not isinstance(target_item, dict):
        return "target_missing"
    if target_item.get("semantic_approved") is not True:
        return "target_unapproved"
    if compact(target_item.get("edit_instruction")) != edit_request:
        return "target_instruction_mismatch"
    if str(target_item.get("parent_branch_id") or "") != parent_branch_id:
        return "target_parent_mismatch"
    if "seed" in target_item:
        try:
            if int(target_item["seed"]) != int(seed):
                return "target_seed_mismatch"
        except Exception:
            return "target_seed_mismatch"
    if not target_item.get("target_image"):
        return "target_path_missing"
    return None


def _keep_designation(uid, share):
    """Stable R-keep assignment, independent of the diagnosis outcome.

    Keep must cover scenes that are already correct and scenes that are not;
    deriving it from "diagnosis passed" alone would bias the keep pool.
    """
    share = float(share or 0.0)
    if share <= 0:
        return False
    if share >= 1:
        return True
    digest = hashlib.sha1(("keep:" + str(uid)).encode()).hexdigest()
    return int(digest[:8], 16) / float(16**8) < share


def _refresh_candidate(marker, targets):
    """A skipped marker is worth recomputing once its edit target arrives.

    The parent trajectory is deterministic for the recorded seed, so refreshing
    only recomputes the records a later target pass can turn into R-edit.
    """
    if not isinstance(marker, dict) or marker.get("diagnosis_status") != "needs_edit":
        return False
    branch = str(marker.get("parent_branch_id") or "")
    try:
        seed = int(branch.split(":")[0])
    except ValueError:
        return False
    item = targets.get(str(marker.get("uid")))
    if isinstance(item, str):
        item = {"target_image": item}
    return (
        _target_binding_error(item, seed, branch, str(marker.get("edit_instruction") or ""))
        is None
    )


def produce_native_parent(rt, cfg, args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prompts = read_rows(args.prompts)
    targets = _load_targets(args.targets)
    explicit_requests = _load_targets(getattr(args, "explicit_requests", None))
    grid = phase_grid(rt, cfg)
    state_times = [float(t) for t in cfg["state_times"]]
    read = float(cfg["read_time"])
    substeps = int(cfg["inversion"]["substeps"])
    endpoint = inversion_endpoint(cfg)
    iterations = int(cfg["inversion"]["iterations"])
    damping = float(cfg["inversion"]["damping"])
    perceptual = _perceptual(args, rt.device)
    editor = StateEditor(rt, cfg).eval() if args.cache_base_prediction else None
    header = _cache_header(rt, cfg, "prepare-state-pairs/native-parent", args)
    rows = []
    start, stop, stride = _shard_slice(len(prompts), args)
    progress = output / "progress"
    progress.mkdir(exist_ok=True)
    log = output / "production.jsonl"
    shards = int(getattr(args, "shard_count", 1) or 1)
    target_requests = output / (
        "target_requests.jsonl" if shards <= 1
        else f"target_requests.shard_{int(getattr(args, 'shard_index', 0)):03d}.jsonl"
    )
    for index in range(start, stop, stride):
        source = prompts[index]
        raw = source.get("raw_prompt") or source.get("prompt")
        if not raw:
            raise ValueError("A native-parent prompt is missing")
        uid = str(source.get("uid") or f"{index:05d}")
        explicit = explicit_requests.get(uid) if explicit_requests else None
        if explicit_requests and explicit is None:
            continue
        marker = progress / f"{_safe(uid)}.json"
        if args.skip_existing and marker.exists():
            previous = json.loads(marker.read_text())
            refresh = bool(getattr(args, "refresh_needs_edit", False)) and _refresh_candidate(
                previous, targets
            )
            if not refresh:
                rows.append(previous)
                continue
        began = time.perf_counter()
        split = source.get("split", "train")
        seed = int(source.get("seed", int(cfg["seed"]) + index))
        generator = torch.Generator(device=rt.device).manual_seed(seed)
        state = torch.randn((1, 3, rt.height, rt.width), device=rt.device, generator=generator) * rt.noise_scale()
        native = NativeT2I(rt, raw, cfg)
        forward = native_forward_states(state, grid, state_times, native)
        states, previews = forward["states"], forward["previews"]
        state = forward["final"]
        if not bool(torch.isfinite(state).all()):
            _write_jsonl(log, dict(uid=uid, skipped="nonfinite_parent"))
            continue
        final_path = output / "targets" / f"{_safe(uid)}_source_final.png"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(final_path, state)

        parent_round_trip = {}
        if getattr(args, "reconstruct_parent", False):
            # The final image is an on-model sample, so this round trip measures
            # the inversion floor itself rather than an off-model corpus.
            back, back_diagnostics = _invert(state, grid, native, cfg, substeps, endpoint)
            whole = torch.ones(1, 1, rt.height, rt.width, device=rt.device)
            for state_time in state_times:
                key = time_key(state_time)
                recon = forward_native(back[key], state_time, grid, native)
                parent_round_trip[key] = dict(
                    reconstruction_metrics(recon, state, whole, perceptual),
                    inversion_residual=back_diagnostics[key].get("segment_residual"),
                )

        parent_branch_id = f"{seed}:t2i"
        if explicit is not None:
            review, edit_request, preserve, failed = None, explicit["edit_instruction"], "", []
            diagnosis_status = "explicit_request"
        else:
            try:
                review = review_preview(rt, previews[time_key(read)], raw, cfg)
            except ValueError as error:
                # An unparseable decomposition is a data problem, not a shard failure.
                _write_jsonl(log, dict(uid=uid, skipped="diagnosis_failed", error=str(error)[:200]))
                continue
            edit_request, preserve, failed = (
                review["edit_request"], review["preserve"], review["failed"],
            )
        if not failed and explicit is None:
            diagnosis_status = "passed"
        elif edit_request and explicit is None:
            diagnosis_status = "needs_edit"
        elif explicit is None:
            diagnosis_status = "unactionable"
        designated_keep = False if explicit is not None else _keep_designation(
            uid, getattr(args, "keep_share", 0.0)
        )
        if diagnosis_status == "unactionable" and not designated_keep:
            _write_jsonl(
                log,
                dict(uid=uid, skipped="unactionable_diagnosis", parent_branch_id=parent_branch_id, failed=failed),
            )
            continue

        target_item = explicit if explicit is not None else targets.get(uid)
        if isinstance(target_item, str):
            target_item = {"target_image": target_item}
        if explicit is not None and not explicit.get("target_image"):
            _write_jsonl(log, dict(uid=uid, skipped="explicit_target_missing"))
            continue
        if (diagnosis_status == "needs_edit" and not designated_keep) or explicit is not None:
            binding_error = _target_binding_error(
                target_item, seed, parent_branch_id, edit_request
            )
            if binding_error:
                # Ask only for targets that do not exist yet, and never twice.
                _write_jsonl(
                    target_requests,
                    dict(
                        uid=uid,
                        parent_branch_id=parent_branch_id,
                        seed=seed,
                        source_image=_relative(output, final_path),
                        parent_condition=raw,
                        edit_instruction=edit_request,
                        preserve=preserve,
                        questions=(review or {}).get("questions", []),
                        failed_questions=failed,
                    ),
                )
                _write_jsonl(
                    log,
                    dict(
                        uid=uid,
                        skipped=binding_error,
                        parent_branch_id=parent_branch_id,
                        failed=failed,
                    ),
                )
                continue
            target_path = target_item.get("target_image")
            anchors = Path(getattr(args, "explicit_requests", None) or args.targets).parent
            resolved_target = (
                Path(target_path) if Path(target_path).is_absolute() else anchors / target_path
            )
            try:
                _, target_pixels, target_mask, target_rect = load_pair(
                    {"source_image": str(final_path), "target_image": str(resolved_target)},
                    rt.width,
                    rt.height,
                    rt.device,
                )
            except ValueError as error:
                _write_jsonl(
                    log,
                    dict(
                        uid=uid,
                        skipped="target_geometry",
                        parent_branch_id=parent_branch_id,
                        error=str(error),
                    ),
                )
                continue
            tail_condition = (
                (target_item.get("tail_condition") or raw) if explicit is not None else raw
            )
            tail_native = NativeT2I(rt, tail_condition, cfg)
            tail_states, tail_inversion = _invert(
                target_pixels, grid, tail_native, cfg, substeps, endpoint
            )
            edit = edit_request
            target_source = target_item.get("target_quality_source", "external_editor")
            use_keep = False
            role = ROLE_R_EDIT
            task_type = "explicit_edit" if explicit is not None else "self_correct"
        else:
            target_pixels = target_mask = target_rect = None
            tail_states = dict(states)
            tail_inversion = None
            tail_native = None
            tail_condition = raw
            edit = cfg["keep_instruction"]
            target_source = "native_keep"
            use_keep = True
            role = ROLE_R_KEEP
            task_type = "self_correct"

        record = dict(
            uid=uid,
            source_kind=NATIVE_PARENT,
            split=split,
            parent_branch_id=parent_branch_id,
            diagnosis_status=diagnosis_status,
            role=role,
            task_type=task_type,
            diagnosed_edit_request=edit_request or None,
            source_family=source.get("source_family", uid),
            prompt_family=source.get("prompt_family", uid),
            source_image=_relative(output, final_path),
            target_image=target_item.get("target_image") if (target_item and not use_keep) else None,
            edit_instruction=edit,
            parent_condition=raw,
            tail_condition=tail_condition,
            preserve=preserve if not use_keep else "",
            condition_protocol=cfg["condition_protocol"],
            valid_region=list(target_rect) if target_rect else None,
            source_seed_or_linear_noise_seed=seed,
            states={},
            source_reconstruction_metrics={},
            target_reconstruction_metrics={},
            semantic_quality_source="native_forward",
            target_quality_source=target_source,
            parent_round_trip=parent_round_trip,
        )
        for state_time in state_times:
            key = time_key(state_time)
            anchor = states[key]
            destination = anchor if use_keep else tail_states[key]
            valid = (
                torch.ones(1, 1, rt.height, rt.width, device=rt.device)
                if target_mask is None
                else target_mask
            )
            target_metrics = {}
            reason = None
            inversion_residual = None
            verdict = None
            if not use_keep:
                # x_eff: what the frozen tail actually renders from the label state.
                recon = forward_native(destination, state_time, grid, tail_native)
                target_metrics = reconstruction_metrics(recon, target_pixels, valid, perceptual)
                inversion_residual = tail_inversion[key].get("segment_residual")
                verdict = _effective_acceptance(rt, state, recon, target_item, cfg)
                reason = None if verdict["approved"] else "effective_" + str(verdict["reason"])
                target_recon = recon
            else:
                target_recon = None
            if reason is None and not all(
                bool(torch.isfinite(value).all())
                for value in (anchor, destination, previews[key])
            ):
                reason = "nonfinite_state"
            accepted = reason is None
            base = None
            if editor is not None:
                base = editor.image_clean(anchor, state_time, editor.image_prefix(previews[key])["image"])
            payload = dict(
                time=float(state_time),
                state_a=anchor.float().cpu(),
                state_b=destination.float().cpu(),
                preview=previews[key].float().cpu(),
                base_prediction=None if base is None else base.float().cpu(),
                valid=valid.float().cpu(),
                source_reconstruction={},
                target_reconstruction=target_metrics,
                accepted=accepted,
            )
            state_file = _save_payload(output, uid, state_time, payload)
            record["states"][key] = dict(
                path=state_file,
                accepted=accepted,
                reject_reason=reason,
                effective=verdict,
                effective_quality=dict(
                    pending_review=True,
                    l1_vs_candidate=target_metrics.get("l1"),
                    lpips_vs_candidate=target_metrics.get("lpips"),
                    hf_effective=_high_frequency(target_recon) if verdict is not None else None,
                    hf_parent=_high_frequency(state),
                ),
                inversion=None if tail_inversion is None else tail_inversion[key],
                source_reconstruction={},
                target_reconstruction=target_metrics,
                source_reconstruction_image=None,
                target_reconstruction_image=_recon_image(output, uid, state_time, "target", target_recon),
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
