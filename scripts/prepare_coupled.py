"""Produce coupled forward pairs (same noise, two conditions) as an R-edit cache.

Usage::

    prepare_coupled.py --model ... --source ... --config ... \
        --requests explicit/requests_all.jsonl --output r_coupled \
        --limit 300 --shard-index 0 --shard-count 16

Each request supplies ``parent_condition``, ``tail_condition``, ``edit_instruction``
and the two question sets; the script runs the frozen native flow twice from one
noise draw, so the label needs neither an external editor nor an inversion.
"""

import argparse
import json

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.coupled import produce
from qwen_latent_cot.u15_target_correction.loading import load_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--source-cache", required=True,
                        help="Cache whose records hold the real a, preview and final image")
    parser.add_argument("--teacher", choices=("it2i", "text"), default="it2i")
    parser.add_argument("--teacher-image", choices=("preview", "source"), default="preview")
    parser.add_argument("--teacher-prompt", choices=("joint", "instruction", "target"),
                        default="joint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    device = set_device(args.device)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, distributed=False)
    path, written = produce(rt, cfg, args)
    print(json.dumps(dict(manifest=str(path), written=written)))


if __name__ == "__main__":
    main()
