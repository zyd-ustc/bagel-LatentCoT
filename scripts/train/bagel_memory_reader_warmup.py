"""Phase 1A.0 reader warm-up; supports torchrun synchronous adapter training."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import yaml

from qwen_latent_cot.bagel.reader_warmup import (
    load_warmup_records, train_warmup, validate_warmup_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/memory_reader_warmup.yaml")
    for key in ("model-path", "prompt-data", "heldout-prompt-data", "output-dir", "device"):
        parser.add_argument(f"--{key}")
    for key in ("max-steps", "eval-max-prompts"):
        parser.add_argument(f"--{key}", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text()) or {}
    for key in ("model_path", "prompt_data", "heldout_prompt_data", "output_dir",
                "device", "max_steps", "eval_max_prompts"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    config = validate_warmup_config(config)
    train_records, heldout_records = load_warmup_records(config)
    if args.validate_only:
        print(json.dumps(dict(stage="phase1a_0_reader_warmup",
            train_records=len(train_records), heldout_records=len(heldout_records),
            config=config), indent=2))
        return
    import torch
    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed warm-up currently requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        config["device"] = f"cuda:{local_rank}"
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    try:
        train_warmup(config, train_records, heldout_records)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
