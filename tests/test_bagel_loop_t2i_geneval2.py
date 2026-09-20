from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "evaluate"))

from score_bagel_loop_t2i_geneval2 import load_generation_inputs  # noqa: E402


def test_geneval2_loader_aligns_benchmark_and_arm_maps(tmp_path):
    output_dir = tmp_path / "run"
    map_dir = output_dir / "geneval2"
    map_dir.mkdir(parents=True)
    prompts = ["prompt two", "prompt one"]
    arms = [
        {"id": "Z0", "slug": "z0_vanilla"},
        {"id": "K1", "slug": "k1_read_write"},
    ]
    images = {}
    for arm in arms:
        image_map = {}
        for index, prompt in enumerate(prompts):
            path = output_dir / f"{arm['id']}_{index}.png"
            path.write_bytes(b"png")
            image_map[prompt] = str(path)
        relative = f"geneval2/{arm['slug']}_image_paths.json"
        (output_dir / relative).write_text(json.dumps(image_map), encoding="utf-8")
        images[arm["id"]] = relative
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "prompts": prompts,
                "arms": arms,
                "geneval2_image_maps": images,
            }
        ),
        encoding="utf-8",
    )
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        "\n".join(
            json.dumps(
                {
                    "prompt": prompt,
                    "atom_count": 1,
                    "skills": ["object"],
                    "vqa_list": [["Is it present?", "Yes"]],
                }
            )
            for prompt in reversed(prompts)
        ),
        encoding="utf-8",
    )

    rows, loaded_arms, arm_maps = load_generation_inputs(output_dir, benchmark)
    assert [row["prompt"] for row in rows] == prompts
    assert [arm["id"] for arm in loaded_arms] == ["Z0", "K1"]
    assert set(arm_maps["Z0"]) == set(prompts)
