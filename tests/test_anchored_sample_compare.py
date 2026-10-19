from pathlib import Path

import torch
from PIL import Image

from qwen_latent_cot.anchored.native_path import BranchContext
from qwen_latent_cot.anchored.sample_compare import (
    attach_predict_flow,
    comparison_strip,
    difference_heatmap,
    euler_nodes,
    forward_flow_branches,
    initial_noise,
    load_prompts,
    mean_absolute_difference,
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


def _flow_call(cfg_text_scale, cfg_img_scale, text_cache, image_cache):
    length = torch.tensor([4], dtype=torch.int32)
    index = torch.arange(4)
    return {
        "timestep": torch.tensor([0.5, 0.5]),
        "packed_vae_token_indexes": index,
        "packed_text_indexes": index[:1],
        "packed_indexes": index,
        "packed_position_ids": index,
        "packed_seqlens": length,
        "key_values_lens": torch.tensor([0], dtype=torch.int32),
        "past_key_values": object(),
        "packed_key_value_indexes": torch.zeros(0, dtype=torch.long),
        "cfg_text_scale": cfg_text_scale,
        "cfg_text_packed_position_ids": index,
        "cfg_text_packed_query_indexes": index,
        "cfg_text_key_values_lens": torch.tensor([0], dtype=torch.int32),
        "cfg_text_past_key_values": text_cache,
        "cfg_text_packed_key_value_indexes": torch.zeros(0, dtype=torch.long),
        "cfg_img_scale": cfg_img_scale,
        "cfg_img_packed_position_ids": index,
        "cfg_img_packed_query_indexes": index,
        "cfg_img_key_values_lens": torch.tensor([0], dtype=torch.int32),
        "cfg_img_past_key_values": image_cache,
        "cfg_img_packed_key_value_indexes": torch.zeros(0, dtype=torch.long),
    }


def test_text_cfg_is_a_separate_branch_and_image_cfg_stays_off():
    text_cache, image_cache = object(), object()
    branches = forward_flow_branches(_flow_call(4.0, 1.0, text_cache, image_cache))
    assert set(branches) == {"cond", "cfg_text"}
    assert isinstance(branches["cfg_text"], BranchContext)
    assert branches["cfg_text"].past_key_values is text_cache
    assert branches["cfg_text"].past_key_values is not branches["cond"].past_key_values
    assert branches["cond"].flow_time == 0.5


def test_patch_calls_predict_flow_and_restores_the_native_method():
    seen = []

    class Model:
        def _forward_flow(self, **_kwargs):
            seen.append("native")
            return "native"

    class Loop:
        def predict_flow(self, model, x_t, timestep, *_args, **_kwargs):
            seen.append(("loop", model is holder[0], float(timestep.reshape(-1)[0])))
            return "loop"

    holder = []
    model = Model()
    holder.append(model)
    call = _flow_call(4.0, 1.0, object(), object())
    with attach_predict_flow(model, Loop()):
        result = model._forward_flow(
            x_t=torch.zeros(2, 2),
            timestep=call["timestep"],
            packed_vae_token_indexes=call["packed_vae_token_indexes"],
            packed_vae_position_ids=torch.arange(2),
            packed_text_ids=torch.tensor([1]),
            packed_text_indexes=call["packed_text_indexes"],
            packed_indexes=call["packed_indexes"],
            packed_position_ids=call["packed_position_ids"],
            packed_seqlens=call["packed_seqlens"],
            key_values_lens=call["key_values_lens"],
            past_key_values=call["past_key_values"],
            packed_key_value_indexes=call["packed_key_value_indexes"],
            cfg_text_scale=4.0,
            cfg_text_packed_position_ids=call["cfg_text_packed_position_ids"],
            cfg_text_packed_query_indexes=call["cfg_text_packed_query_indexes"],
            cfg_text_key_values_lens=call["cfg_text_key_values_lens"],
            cfg_text_past_key_values=call["cfg_text_past_key_values"],
            cfg_text_packed_key_value_indexes=call["cfg_text_packed_key_value_indexes"],
            cfg_img_scale=1.0,
        )
    assert result == "loop"
    assert seen == [("loop", True, 0.5)]
    assert model._forward_flow() == "native"
    assert seen[-1] == "native"


def test_heatmap_amplifies_a_one_level_difference_by_ten():
    left = Image.new("RGB", (2, 2), (0, 0, 0))
    right = Image.new("RGB", (2, 2), (1, 1, 1))
    assert mean_absolute_difference(left, right) == 1.0
    heat = difference_heatmap(left, right, gain=10.0)
    assert heat.getpixel((0, 0))[0] >= 20
    assert difference_heatmap(left, left).getpixel((0, 0)) == (0, 0, 0)


def test_strip_holds_the_prompt_and_every_panel():
    image = Image.new("RGB", (8, 6), (10, 20, 30))
    names = (
        "native", "gen_only_400", "dual_600", "dual_shuffled",
        "|dual-native|x10", "|dual-gen|x10",
    )
    panels = [(name, image.copy()) for name in names]
    strip = comparison_strip("a red bowl", panels)
    assert strip.size[0] == 300 + 8 * len(panels)
    assert strip.size[1] >= 32 + 6
