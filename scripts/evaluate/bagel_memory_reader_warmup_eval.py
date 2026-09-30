"""Heldout bank-readout evaluation and a checkpoint-bound OPD readiness report."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from qwen_latent_cot.bagel.cot_teacher import sha256_file
from qwen_latent_cot.bagel.reader_warmup import (
    SCHEMA, ReaderWarmupRuntime, evaluate_warmup, inspect_warmup_checkpoint, load_warmup_checkpoint,
    load_warmup_records, validate_warmup_config, warmup_evaluation_report)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/training/memory_reader_warmup.yaml")
    parser.add_argument("--checkpoint",required=True)
    for key in ("model-path","prompt-data","heldout-prompt-data","output-dir","device"):
        parser.add_argument(f"--{key}")
    parser.add_argument("--max-prompts",type=int)
    parser.add_argument("--validate-only",action="store_true")
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text()) or {}
    for key in ("model_path","prompt_data","heldout_prompt_data","output_dir","device"):
        value=getattr(args,key)
        if value is not None:
            config[key]=value
    if args.max_prompts is not None:
        config["eval_max_prompts"]=args.max_prompts
    config=validate_warmup_config(config)
    train,heldout=load_warmup_records(config)
    heldout=heldout[:config["eval_max_prompts"]]
    checkpoint=Path(args.checkpoint)
    meta=inspect_warmup_checkpoint(checkpoint)
    if (not checkpoint.is_file() or meta.get("schema")!=SCHEMA
            or meta.get("prompt_data_sha256")!=sha256_file(config["prompt_data"])
            or meta.get("heldout_prompt_data_sha256")!=sha256_file(config["heldout_prompt_data"])):
        raise ValueError("warm-up checkpoint does not match the original prompt splits")
    if args.validate_only:
        print(json.dumps(dict(train_records=len(train),heldout_records=len(heldout),
            checkpoint=str(checkpoint),config=config),indent=2))
        return
    output=Path(config["output_dir"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite warm-up evaluation output: {output}")
    runtime=ReaderWarmupRuntime.load_model(config)
    load_warmup_checkpoint(runtime,checkpoint)
    evaluation=evaluate_warmup(runtime,heldout,seed=config["seed"])
    # Reconstruct the initial frozen GEN-O readout exactly: B=0 removes the
    # adapter for any A. Reuse the same native states, prompts, query and seed.
    saved={name:p.detach().clone() for name,p in runtime.model.named_parameters()
           if "memory_reader.output_adapter.B.weight" in name}
    named=dict(runtime.model.named_parameters())
    try:
        with torch.no_grad():
            for name in saved:
                named[name].zero_()
        initial=evaluate_warmup(runtime,heldout,seed=config["seed"])
    finally:
        with torch.no_grad():
            for name,value in saved.items():
                named[name].copy_(value)
    report=warmup_evaluation_report(runtime,config,checkpoint,evaluation,initial,heldout)
    output.mkdir(parents=True,exist_ok=False)
    (output/"warmup_gate.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(output=str(output),checks=report["checks"],
        ready_for_opd=report["ready_for_opd"],heldout_error_mse=report["heldout_error_mse"])),flush=True)


if __name__=="__main__":
    main()
