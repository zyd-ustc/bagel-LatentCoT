"""Generate offline frozen BAGEL Text-CoT records; invalid rows fail closed."""

import argparse
import json
from pathlib import Path

from qwen_latent_cot.bagel.cot_teacher import (
    TEMPLATE_VERSION, SEMANTIC_CATEGORIES, teacher_instruction, validate_teacher_record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.max_retries < 0 or (args.max_prompts is not None and args.max_prompts < 1):
        raise ValueError("max_retries must be >=0 and max_prompts must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"inspect the previous incomplete cache: {partial}")
    source = [json.loads(line) for line in Path(args.prompt_data).read_text().splitlines() if line.strip()]
    if args.max_prompts is not None:
        source = source[:args.max_prompts]
    if not source:
        raise ValueError("prompt source is empty")
    ids, prompts = set(), set()
    for index, row in enumerate(source):
        key = str(row.get("prompt_id", row.get("id", index)))
        prompt = row.get("prompt")
        if (not isinstance(prompt, str) or not prompt.strip()
                or row.get("category") not in SEMANTIC_CATEGORIES
                or key in ids or prompt in prompts):
            raise ValueError(f"row {index} needs unique semantic T2I id/prompt and allowed category")
        ids.add(key)
        prompts.add(prompt)
    if args.validate_only:
        print(json.dumps(dict(records=len(source),template=TEMPLATE_VERSION)))
        return
    from qwen_latent_cot.bagel.opd_runtime import OPDRuntime
    runtime = OPDRuntime.load_model(dict(model_path=args.model_path, device=args.device,
                                         height=512, width=512, num_loop_tokens=8,
                                         memory_loop_start_layer=12,
                                         memory_loop_end_layer=20), stage="base")
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("x", encoding="utf-8") as handle:
        for index, row in enumerate(source):
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip() or not row.get("category"):
                raise ValueError(f"row {index} needs a semantic T2I prompt and category")
            previous_error=None
            for attempt in range(args.max_retries + 1):
                context = runtime.inferencer.init_gen_context()
                instruction=teacher_instruction(prompt)
                if previous_error is not None:
                    instruction += f"\nCorrect this validation failure: {previous_error}. Return the seven-section plan only."
                context = runtime.inferencer.update_context_text(
                    instruction, context)
                reasoning = runtime.inferencer.gen_text(context, max_length=160,
                                                        do_sample=False,
                                                        temperature=1.)
                cached = dict(prompt_id=str(row.get("prompt_id", row.get("id", index))),
                              prompt=prompt, category=row["category"],
                              reasoning_text=reasoning.strip(),
                              teacher_template_version=TEMPLATE_VERSION,
                              teacher_model_revision=Path(args.model_path).name,
                              reasoning_generation=dict(max_tokens=160, do_sample=False,
                                                        temperature=1.0))
                try:
                    validate_teacher_record(cached, tokenizer=runtime.inferencer.tokenizer)
                    break
                except ValueError as exc:
                    previous_error=str(exc)
                    if attempt == args.max_retries:
                        raise ValueError(f"teacher quality gate failed at source row {index}")
            handle.write(json.dumps(cached, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"cached {index + 1}/{len(source)} {cached['prompt_id']}", flush=True)
    partial.replace(output)


if __name__ == "__main__":
    main()
