"""Entry point for the anchored dual-state line."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import LoopConfig
from .legacy import refuse_legacy_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description="Anchored Dual-State Loop")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-check", help="run the native-bypass and dependency contract")
    show = commands.add_parser("show-config")
    show.add_argument("--config", default="configs/anchored_dual_state_loop.json")
    reject = commands.add_parser("reject-legacy")
    reject.add_argument("checkpoint", help="JSON or torch checkpoint to refuse if it is not ADSL")
    train = commands.add_parser("train-gen-only", help="train the GEN-only residual and gate on 64 devices")
    train.add_argument("--model", required=True)
    train.add_argument(
        "--train", required=True,
        help="caption-image JSONL, or an annotation-index directory whose generation train prompt is the text",
    )
    train.add_argument("--output", required=True)
    train.add_argument("--loop-config", default="configs/anchored_gen_only.json")
    train.add_argument("--train-config", default="configs/anchored_gen_only_train.json")
    train.add_argument("--device", default="npu")
    train.add_argument("--distributed", action="store_true")
    train.add_argument("--resume")
    train.add_argument("--steps", type=int)
    train.add_argument("--global-batch", type=int)
    curriculum = commands.add_parser(
        "train-anchored",
        help="train the two-stage GEN-only then dual curriculum on 64 devices",
    )
    curriculum.add_argument("--model", required=True)
    curriculum.add_argument(
        "--train", required=True,
        help="caption-image JSONL, or an annotation-index directory whose generation train prompt is the text",
    )
    curriculum.add_argument("--output", required=True)
    curriculum.add_argument("--loop-config", default="configs/anchored_gen_only.json")
    curriculum.add_argument("--train-config", default="configs/anchored_curriculum_train.json")
    curriculum.add_argument("--observation-checkpoint", required=True)
    curriculum.add_argument("--val", default=None, help="val JSONL, or an annotation index whose generation val split is the probe")
    curriculum.add_argument("--device", default="npu")
    curriculum.add_argument("--distributed", action="store_true")
    curriculum.add_argument("--resume")
    curriculum.add_argument("--global-batch", type=int)
    args = parser.parse_args(argv)
    if args.command == "train-gen-only":
        _train_gen_only(args)
        return
    if args.command == "train-anchored":
        _train_anchored(args)
        return
    if args.command == "self-check":
        from .self_check import run_self_check
        names = run_self_check()
        print(json.dumps({"ok": True, "checks": names}))
        return
    if args.command == "show-config":
        config = LoopConfig.from_json(args.config)
        print(json.dumps({"experiment": "anchored_dual_state_loop", **config.to_dict()}, indent=2))
        return
    path = Path(args.checkpoint)
    if path.suffix == ".json":
        blob = json.loads(path.read_text())
    else:
        import torch
        blob = torch.load(path, map_location="cpu", weights_only=False)
    refuse_legacy_checkpoint(blob)
    print(json.dumps({"ok": True, "experiment": blob.get("experiment")}))


def _train_gen_only(args) -> None:
    if not args.distributed:
        raise SystemExit("train-gen-only runs through scripts/anchored_gen_only_64.sh")
    from ..bagel.backbone import load_native
    from ..diagnostics import distributed as parallel
    from .gen_only_data import open_gen_only_source
    from .gen_only_train import (
        GenOnlyTrainConfig,
        require_gen_only_loop,
        require_sixty_four,
        train_gen_only,
    )
    from .loop import AnchoredDualStateLoop
    train = GenOnlyTrainConfig.from_json(args.train_config)
    if args.steps is not None:
        train.steps = args.steps
    if args.global_batch is not None:
        train.global_batch = args.global_batch
    loop_config = LoopConfig.from_json(args.loop_config)
    require_gen_only_loop(loop_config)
    train.validate()
    device = parallel.initialize(args.device)
    require_sixty_four(parallel.world_size(), train.global_batch)
    pairs = open_gen_only_source(args.train, train.seed)
    output = Path(args.output)
    if parallel.rank() == 0:
        if args.resume and not Path(args.resume).is_file():
            raise ValueError("resume checkpoint is missing")
        if not args.resume and output.exists() and any(output.iterdir()):
            raise ValueError("use a fresh output directory")
        output.mkdir(parents=True, exist_ok=True)
    parallel.barrier()
    bundle = load_native(args.model, device, train.timestep_shift)
    loop = AnchoredDualStateLoop(bundle.model.language_model, loop_config).to(device)
    train_gen_only(loop, bundle, pairs, train, output, resume=args.resume)


def _train_anchored(args) -> None:
    if not args.distributed:
        raise SystemExit("train-anchored runs through scripts/anchored_train_64.sh")
    from ..bagel.backbone import load_native
    from ..diagnostics import distributed as parallel
    from .anchored_train import (
        AnchoredTrainConfig,
        require_stage2_reader,
        train_anchored,
    )
    from .gen_only_data import open_curriculum_val, open_gen_only_source
    from .gen_only_train import require_gen_only_loop, require_sixty_four
    from .loop import AnchoredDualStateLoop
    train = AnchoredTrainConfig.from_json(args.train_config)
    if args.global_batch is not None:
        train.global_batch = args.global_batch
    loop_config = LoopConfig.from_json(args.loop_config)
    require_gen_only_loop(loop_config)
    train.validate()
    require_stage2_reader(args.observation_checkpoint, loop_config.observation_source_mode)
    val_pairs = open_curriculum_val(args.train, args.val, train.seed + 1)
    device = parallel.initialize(args.device)
    require_sixty_four(parallel.world_size(), train.global_batch)
    pairs = open_gen_only_source(args.train, train.seed)
    output = Path(args.output)
    if parallel.rank() == 0:
        if args.resume and not Path(args.resume).is_file():
            raise ValueError("resume checkpoint is missing")
        if not args.resume and output.exists() and any(output.iterdir()):
            raise ValueError("use a fresh output directory")
        output.mkdir(parents=True, exist_ok=True)
    parallel.barrier()
    bundle = load_native(args.model, device, train.timestep_shift)
    loop = AnchoredDualStateLoop(bundle.model.language_model, loop_config).to(device)
    train_anchored(
        loop, bundle, pairs, train, output,
        observation_checkpoint=args.observation_checkpoint, val_pairs=val_pairs, resume=args.resume,
    )


if __name__ == "__main__":
    main()
