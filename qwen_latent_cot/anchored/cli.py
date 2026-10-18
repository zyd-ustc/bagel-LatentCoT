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
    args = parser.parse_args(argv)
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


if __name__ == "__main__":
    main()
