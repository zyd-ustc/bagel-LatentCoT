from pathlib import Path

import torch
from PIL import Image

from qwen_latent_cot.anchored.sample_compare import (
    comparison_strip,
    euler_nodes,
    initial_noise,
    load_prompts,
    steps_inside,
)


def test_trained_window_covers_twenty_five_contiguous_updates():
    nodes, dts = euler_nodes(50, 3.0)
    assert nodes.shape[0] == 49
    assert torch.all(dts > 0)
    progress = 1.0 - nodes
    inside = (progress >= 0.15) & (progress <= 0.65)
    assert int(inside.sum()) == 25
    indexes = torch.where(inside)[0]
    assert int(indexes[-1] - indexes[0] + 1) == 25
    assert steps_inside(nodes) == 25


def test_one_seed_repeats_the_same_noise():
    left = initial_noise(16, 8, 17)
    right = initial_noise(16, 8, 17)
    assert torch.equal(left, right)
    assert not torch.equal(left, initial_noise(16, 8, 18))


def test_prompt_file_has_one_line_per_card():
    path = Path("configs/anchored_sample_prompts_64.jsonl")
    prompts = load_prompts(path)
    assert len(prompts) == 64
    assert len(set(prompts)) == 64


def test_strip_is_prompt_plus_three_images():
    image = Image.new("RGB", (8, 6), (10, 20, 30))
    panels = [
        ("native", image),
        ("gen_only_400", image.copy()),
        ("dual_600_1round", image.copy()),
    ]
    strip = comparison_strip("a red bowl", panels)
    assert strip.size[0] == 300 + 24
    assert strip.size[1] >= 32 + 6
