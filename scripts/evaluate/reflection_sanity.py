#!/usr/bin/env python3
"""Sanity check: is BAGEL's understanding/reflection path actually working?

Runs on a REAL image and compares
  A) our ``InterleaveInferencer.generate_image_reflection``
  B) the official ``inferencer(..., understanding_output=True)`` call shape

If A and B agree and both describe the image correctly, the understanding path
is fine and any bad reflection on a draft is a *draft quality* problem (t too
high), not an inference bug.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from bagel_common import add_native_model_args, load_native_bagel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_native_model_args(parser)
    parser.add_argument("--image", required=True)
    parser.add_argument("--user-text", default="Describe this image in detail.")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--skip-official", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[model] loading frozen native BAGEL checkpoint", flush=True)
    backbone, inferencer = load_native_bagel(args)

    image = Image.open(args.image).convert("RGB")
    image.save(output_dir / "input.png")

    ours = inferencer.generate_image_reflection(
        image,
        user_text=str(args.user_text),
        system_prompt=str(args.system_prompt) or None,
        max_length=int(args.max_tokens),
        do_sample=False,
    )
    print("=" * 30, "OURS (generate_image_reflection)", "=" * 30)
    print(ours, flush=True)

    official = None
    if not args.skip_official:
        result = inferencer(
            image=image,
            text=str(args.user_text),
            understanding_output=True,
            max_think_token_n=int(args.max_tokens),
            do_sample=False,
        )
        official = result.get("text")
        print("=" * 30, "OFFICIAL (understanding_output=True)", "=" * 30)
        print(official, flush=True)

    (output_dir / "reflection_sanity.json").write_text(
        json.dumps(
            {
                "image": str(Path(args.image).expanduser().resolve()),
                "user_text": str(args.user_text),
                "system_prompt": str(args.system_prompt),
                "max_tokens": int(args.max_tokens),
                "ours": ours,
                "official": official,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] {output_dir / 'reflection_sanity.json'}", flush=True)


if __name__ == "__main__":
    main()
