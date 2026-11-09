"""Caption source images with Qwen3-VL for the L parent condition."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image


def _read_jsonl(path):
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _done(path):
    if not os.path.exists(path):
        return set()
    out = set()
    for row in _read_jsonl(path):
        uid = row.get("uid")
        if uid and row.get("parent_condition"):
            out.add(str(uid))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="/data/zyd_workspace/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--image-max-edge", type=int, default=768)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = _done(output)
    rows = []
    for index, row in enumerate(_read_jsonl(args.input)):
        if index % args.shards != args.shard:
            continue
        if str(row.get("uid")) in done:
            continue
        rows.append(row)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(json.dumps(dict(shard=args.shard, rows=0, skipped=len(done))))
        return

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.npu.set_device(args.device)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to(args.device).eval()

    prompt = (
        "Describe this image in one detailed sentence. Cover the main subjects, "
        "their visible attributes, important spatial relations, and the setting. "
        "Do not speculate and do not mention any editing instruction."
    )
    began = time.perf_counter()
    with output.open("a") as handle:
        for index, row in enumerate(rows):
            uid = str(row["uid"])
            path = row.get("source_image") or row.get("image")
            if not path or not os.path.exists(path):
                record = dict(uid=uid, error="missing_image", image=path)
            else:
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                    if max(image.size) > args.image_max_edge:
                        image.thumbnail((args.image_max_edge, args.image_max_edge))
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = processor(text=[text], images=[image], return_tensors="pt").to(
                        args.device
                    )
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                        )
                    caption = processor.batch_decode(
                        generated[:, inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )[0].strip()
                    record = dict(uid=uid, parent_condition=caption, image=path)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 20 == 0:
                print(
                    json.dumps(
                        dict(
                            shard=args.shard,
                            done=index + 1,
                            total=len(rows),
                            seconds=time.perf_counter() - began,
                        )
                    ),
                    flush=True,
                )
    print(json.dumps(dict(shard=args.shard, rows=len(rows), complete=True)))


if __name__ == "__main__":
    main()
