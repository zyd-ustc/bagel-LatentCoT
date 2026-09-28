"""Pretraining teacher baseline or held-out Phase 1A field/image controls."""

import argparse
import json
from pathlib import Path

import yaml

from qwen_latent_cot.bagel.cot_teacher import sha256_file
from qwen_latent_cot.bagel.opd_evaluation import (
    fixed_state_metrics, generate_five_arm_images, load_reader_checkpoint,
    teacher_baseline)
from qwen_latent_cot.bagel.opd_runtime import OPDRuntime
from qwen_latent_cot.bagel.opd_training import SCHEMA, load_training_records, validate_opd_config


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training/memory_opd_t0.yaml")
    p.add_argument("--model-path")
    p.add_argument("--prompt-data")
    p.add_argument("--teacher-cot-data")
    p.add_argument("--output-dir")
    p.add_argument("--adapter-path")
    p.add_argument("--allow-train-split-debug", action="store_true")
    p.add_argument("--max-prompts", type=int)
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--generate-images", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text()) or {}
    for name in ("model_path","prompt_data","teacher_cot_data","output_dir"):
        value=getattr(args,name)
        if value is not None:
            config[name]=value
    config["max_prompts"]=args.max_prompts if args.max_prompts is not None else (8 if args.baseline_only else 2)
    config=validate_opd_config(config)
    records=load_training_records(config)
    if args.validate_only:
        print(json.dumps(dict(records=len(records), config=config),indent=2))
        return
    output=Path(config["output_dir"])
    if output.exists():
        raise FileExistsError(output)
    runtime=OPDRuntime.load_model(config)
    load_training_records(config, tokenizer=runtime.inferencer.tokenizer)
    if args.adapter_path:
        meta=load_reader_checkpoint(runtime,args.adapter_path)
        training_digest=meta.get("prompt_data_sha256")
        if not args.allow_train_split_debug:
            if training_digest==sha256_file(config["prompt_data"]):
                raise ValueError("evaluation source equals the training split; supply held-out prompts")
            training_source=Path(meta.get("config",{}).get("prompt_data", ""))
            if not training_source.is_file():
                raise ValueError("cannot audit train/held-out overlap without original training prompt source")
            train_prompts={json.loads(line)["prompt"] for line in training_source.read_text().splitlines()
                           if line.strip()}
            if any(row["prompt"] in train_prompts for row in records):
                raise ValueError("held-out evaluation overlaps training prompts")
    if args.baseline_only and args.adapter_path:
        raise ValueError("teacher baseline must use frozen base, no reader checkpoint")
    output.mkdir(parents=True,exist_ok=False)
    if args.baseline_only:
        seeds=[int(config["seed"]),int(config["seed"])+1]
        diffs=teacher_baseline(runtime,records,seeds=seeds)
        result=dict(schema=SCHEMA,kind="teacher_baseline", model_path=str(Path(config["model_path"]).resolve()),
            teacher_cache_sha256=sha256_file(config["teacher_cot_data"]),
            prompt_ids=[row["prompt_id"] for row in records], num_steps=config["num_steps"],cfg=1.0,
            field_rms_by_seed=diffs, semantic_teacher_minus_native=None,
            evidence_scope="field_difference_only_not_semantic_improvement")
        (output/"teacher_baseline.json").write_text(json.dumps(result,indent=2)+"\n")
    else:
        result=fixed_state_metrics(runtime,records,seed=int(config["seed"]))
        (output/"field_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    if args.generate_images:
        generate_five_arm_images(runtime,records,output,seed=int(config["seed"]))
    print(json.dumps(dict(output=str(output),result=result),ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
