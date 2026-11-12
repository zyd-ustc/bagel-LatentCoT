"""Isochronic state flow: prepare caches, train, sample, native baselines, judge."""

import argparse
import json
import time
from pathlib import Path

from .conditions import compact, derived_tail_condition
from .config import load_config


def _model_args(parser):
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--distributed", action="store_true")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    judge = commands.add_parser("judge")
    for key in ("requests", "output", "base-url", "model", "key-env"):
        judge.add_argument("--" + key, required=True)

    train = commands.add_parser("train")
    _model_args(train)
    train.add_argument("--linear")
    train.add_argument("--inverted")
    train.add_argument("--native")
    train.add_argument("--output", required=True)
    train.add_argument("--resume")
    train.add_argument("--init-from")
    train.add_argument("--steps", type=int)

    sample = commands.add_parser("sample")
    _model_args(sample)
    sample.add_argument("--prompts", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--checkpoint")
    sample.add_argument("--paired", action="store_true", help="Save direct continuation under the same target")
    sample.add_argument("--post-edit", action="store_true", help="Include matched native IT2I after generation")
    sample.add_argument("--limit", type=int, default=32)

    native = commands.add_parser("native")
    _model_args(native)
    native.add_argument("--mode", choices=("t2i", "it2i"), required=True)
    native.add_argument("--prompts", required=True)
    native.add_argument("--output", required=True)
    native.add_argument("--limit", type=int, default=32)

    evaluate = commands.add_parser("eval-linear")
    _model_args(evaluate)
    evaluate.add_argument("--linear", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--limit", type=int, default=32)
    evaluate.add_argument("--seed", type=int, default=20261107)
    evaluate.add_argument("--time", type=float)
    evaluate.add_argument("--steps", type=int, default=8)
    evaluate.add_argument("--no-lpips", action="store_true")

    prepare = commands.add_parser("prepare-state-pairs")
    _model_args(prepare)
    prepare.add_argument("--mode", choices=("inverted", "native-parent"), required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--linear")
    prepare.add_argument("--prompts")
    prepare.add_argument("--targets")
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--seed", type=int)
    prepare.add_argument("--shard-index", type=int, default=0)
    prepare.add_argument("--shard-count", type=int, default=1)
    prepare.add_argument("--skip-existing", action="store_true")
    prepare.add_argument(
        "--refresh-needs-edit",
        action="store_true",
        help="With --skip-existing, recompute markers whose target has since arrived",
    )
    prepare.add_argument("--no-lpips", action="store_true")
    prepare.add_argument("--cache-base-prediction", action="store_true")
    prepare.add_argument(
        "--explicit-requests",
        help="Produce explicit-edit records only, keyed by uid, from approved targets",
    )
    prepare.add_argument(
        "--reconstruct-parent",
        action="store_true",
        help="Also invert each parent final image and record its round-trip error",
    )
    prepare.add_argument(
        "--keep-share",
        type=float,
        default=0.0,
        help="Fraction of native-parent prompts designated R-keep by uid hash",
    )
    return parser


def _explicit_edit(row, raw):
    """An explicit edit instruction replaces closed-loop diagnosis."""
    instruction = str(row.get("edit_instruction", "")).strip()
    if not instruction:
        return None
    edit = compact(instruction)
    tail = row.get("tail_condition")
    if not str(tail or "").strip():
        if row.get("allow_derived_tail"):
            tail = derived_tail_condition(raw, edit)
    return dict(
        edit_request=edit,
        tail_condition=compact(tail),
        preserve=compact(row.get("preserve")),
    )


def _train(args, cfg, device):
    from .train import train

    if args.resume and args.init_from:
        raise ValueError("Use either --resume or --init-from, not both")
    train(args, cfg, device)


def _prepare(args, cfg, device):
    from .loading import load_native
    from .prepare_state_pairs import produce_inverted, produce_native_parent

    rt = load_native(args.model, args.source, cfg, device, distributed=False)
    if args.mode == "inverted":
        if not args.linear:
            raise ValueError("Inverted cache production needs --linear")
        path = produce_inverted(rt, cfg, args)
    else:
        if not args.prompts:
            raise ValueError("Native-parent production needs --prompts")
        path = produce_native_parent(rt, cfg, args)
    print(json.dumps(dict(manifest=str(path))))


def _native(args, cfg, device):
    from .loading import load_native
    from .native import run_native

    rt = load_native(args.model, args.source, cfg, device, distributed=False)
    run_native(rt, cfg, args.mode, args.prompts, args.output, args.limit, device)


def _load_editor(rt, cfg, checkpoint):
    """Build the editor and load only its trainable weights."""
    from .checkpoint import editor_identity, load_stage_weights
    from .model import StateEditor

    editor = StateEditor(rt, cfg).eval()
    return editor, load_stage_weights(checkpoint, editor, editor_identity(rt, cfg))


def _sample(args, cfg, device):
    from . import distributed as dist
    from .data import read_rows
    from .images import load_canvas, save_image
    from .loading import load_native
    from .sampling import run

    rt = load_native(args.model, args.source, cfg, device, args.distributed)
    editor = None
    if args.checkpoint:
        editor, meta = _load_editor(rt, cfg, args.checkpoint)
        print(json.dumps(dict(checkpoint=str(args.checkpoint), stage=meta.get("stage"), step=meta.get("step"))))
    if args.distributed and editor is not None:
        dist.broadcast_parameters(editor)

    out = Path(args.output).resolve()
    if dist.rank() == 0:
        out.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    rows = read_rows(args.prompts)[: args.limit]
    for index in range(dist.rank(), len(rows), dist.world_size()):
        row = rows[index]
        raw = row.get("raw_prompt") or row.get("prompt")
        if not raw:
            raise ValueError(f"Row {index} is missing a prompt")
        seed = int(row.get("seed", cfg["seed"] + index))
        uid = row.get("uid") or row.get("prompt_id") or f"{index:05d}"
        explicit = _explicit_edit(row, raw)
        began = time.perf_counter()
        alternate_preview = row.get("conditioning_preview")
        if alternate_preview and not explicit:
            raise ValueError("A conditioning preview intervention requires an explicit edit")
        if alternate_preview:
            preview_path = Path(args.prompts).resolve().parent / alternate_preview
            alternate_preview, _, _ = load_canvas(preview_path, rt.width, rt.height, rt.device)
        images, record = run(
            rt, raw, seed, cfg, editor=editor, explicit=explicit,
            paired=args.paired, post_edit=args.post_edit, conditioning_preview=alternate_preview,
        )
        record["seconds"] = time.perf_counter() - began
        record["uid"] = uid
        record["vqa_list"] = row.get("vqa_list", [])
        record["skills"] = row.get("skills", [])
        record["conditioning_preview"] = row.get("conditioning_preview")
        record["checkpoint"] = args.checkpoint
        record["mode"] = "explicit" if explicit else "closed_loop"
        for key, image in images.items():
            save_image(out / f"{index:05d}_{key}.png", image)
        (out / f"{index:05d}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    dist.barrier()
    if dist.rank() == 0:
        (out / "complete.json").write_text(json.dumps(dict(samples=len(rows))) + "\n")


def _eval_linear(args, cfg, device):
    from . import distributed as dist
    from .data import LinearManifest
    from .eval_linear import evaluate_linear
    from .loading import load_native

    rt = load_native(args.model, args.source, cfg, device, args.distributed)
    editor, _ = _load_editor(rt, cfg, args.checkpoint)
    editor.checkpoint = args.checkpoint
    if args.distributed:
        dist.broadcast_parameters(editor)
    manifest = LinearManifest(args.linear, rt, cfg)
    evaluate_linear(
        rt,
        cfg,
        editor,
        manifest,
        args.output,
        limit=args.limit,
        seed=args.seed,
        time=args.time if args.time is not None else float(cfg["read_time"]),
        steps=args.steps,
        no_lpips=args.no_lpips,
    )


def main():
    args = _parser().parse_args()
    if args.command == "judge":
        from .judge import judge as judge_rows

        judge_rows(args.requests, args.output, args.base_url, args.model, args.key_env)
        return

    from . import distributed as dist
    from .accelerator import set_device

    cfg = load_config(args.config)
    if args.command in ("prepare-state-pairs", "native") and args.distributed:
        raise ValueError("Cache production and native baselines use prompt shards, not torchrun")
    device = dist.initialize(args.device) if args.distributed else set_device(args.device)

    runners = {
        "train": _train,
        "prepare-state-pairs": _prepare,
        "native": _native,
        "sample": _sample,
        "eval-linear": _eval_linear,
    }
    runners[args.command](args, cfg, device)
    if args.distributed:
        dist.shutdown()


if __name__ == "__main__":
    main()
