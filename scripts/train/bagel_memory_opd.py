"""Single-device BAGEL Phase 1A T0 OPD training entrypoint."""

import argparse
import json
from pathlib import Path

import yaml

from qwen_latent_cot.bagel.opd_training import validate_opd_config, load_training_records, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training/memory_opd_t0.yaml")
    for key in ("model-path", "prompt-data", "teacher-cot-data", "teacher-baseline-json",
                "reader-warmup-checkpoint", "reader-warmup-eval-json",
                "output-dir", "device"):
        p.add_argument(f"--{key}")
    for key in ("max-steps", "max-prompts"):
        p.add_argument(f"--{key}", type=int)
    p.add_argument("--allow-field-only-debug", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()
    config = yaml.safe_load(Path(args.config).read_text()) or {}
    for key in ("model_path", "prompt_data", "teacher_cot_data", "teacher_baseline_json",
                "reader_warmup_checkpoint", "reader_warmup_eval_json",
                "output_dir", "device", "max_steps", "max_prompts"):
        value = getattr(args,key)
        if value is not None:
            config[key]=value
    if args.allow_field_only_debug:
        config["allow_field_only_debug"] = True
    config = validate_opd_config(config)
    records = load_training_records(config)
    if args.validate_only:
        print(json.dumps(dict(stage="phase1a_1a_gate_only", records=len(records), config=config),indent=2))
        return
    train(config, records)


if __name__ == "__main__":
    main()
