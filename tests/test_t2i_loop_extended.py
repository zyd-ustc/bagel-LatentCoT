import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from test_anchored_t2i_loop import flow_inputs, tiny_model
from test_t2i_loop_entrypoints import (
    TinyTokenizer,
    TinyVAE,
    install_tiny_backbone,
    script_module,
)

from qwen_latent_cot.bagel.anchored_loop import memory_slot_stats
from qwen_latent_cot.bagel.inferencer import InterleaveInferencer
from qwen_latent_cot.bagel.loop_checkpoint import (
    load_loop_checkpoint,
    save_loop_checkpoint,
)
from qwen_latent_cot.data.t2i import BucketBatchSampler, T2IDataset, collate_t2i

TOKENS = {"bos_token_id": 0, "eos_token_id": 1, "start_of_image": 2, "end_of_image": 3}
ROOT = Path(__file__).resolve().parents[1]


def test_shared_alpha_checkpoint_supports_trained_r3_and_unseen_r4(tmp_path):
    model, config = tiny_model(2, alpha=0.3)
    assert model.t2i_loop.output_alpha.shape == (1,)
    assert config.runtime_loop_depth == 3 and config.allocated_max_loop_depth == 4
    save_loop_checkpoint(
        model,
        tmp_path,
        step=9,
        model_path="native",
        training_depth_counts={"3": 9},
        round_training_steps=[9, 9, 9, 0],
        depth_curriculum=[],
    )
    model.t2i_loop.config = replace(config, runtime_loop_depth=4)
    load_loop_checkpoint(model, tmp_path)
    assert model.t2i_loop.config.runtime_loop_depth == 4
    inputs = flow_inputs(model)
    with torch.no_grad():
        r4 = model.forward_t2i_loop(
            **inputs, loop_config=replace(config, runtime_loop_depth=4)
        )
        r0 = model.forward_t2i_loop(
            **inputs, loop_config=replace(config, runtime_loop_depth=0)
        )
    assert len(r4.velocities) == 4
    torch.testing.assert_close(r0.velocity, r4.base_velocity, rtol=0, atol=0)
    with pytest.raises(ValueError, match="allocated"):
        replace(config, runtime_loop_depth=5)


@pytest.mark.parametrize(
    "grad,ds,diagnostics,expected",
    [
        (False, True, False, 1),
        (False, True, True, 5),
        (True, True, False, 5),
        (True, False, False, 1),
    ],
)
def test_suffix_runs_only_when_needed(grad, ds, diagnostics, expected, monkeypatch):
    model, config = tiny_model(2, alpha=0.3)
    inputs = flow_inputs(model)
    calls = []
    layer = model.language_model.model.layers[3]
    original = layer.forward_inference

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(layer, "forward_inference", record)
    runtime = replace(
        config,
        runtime_loop_depth=4,
        log_loop_stats=diagnostics,
        loop_deep_supervision=ds,
    )
    with torch.set_grad_enabled(grad):
        result = model.forward_t2i_loop(**inputs, loop_config=runtime)
    assert len(calls) == expected
    assert len(result.velocities) == (4 if expected == 5 else 1)
    assert (result.base_velocity is None) == (expected == 1)
    with torch.no_grad():
        all_readouts = model.forward_t2i_loop(
            **inputs, loop_config=replace(runtime, log_loop_stats=True)
        )
    torch.testing.assert_close(result.velocity, all_readouts.velocity, rtol=0, atol=0)


def test_packed_variable_resolution_matches_independent_samples_and_backward():
    model, config = tiny_model(2, alpha=0.3)
    inferencer = InterleaveInferencer(model, TinyVAE(), TinyTokenizer(), TOKENS)
    prompts, shapes = (
        ["two cubes", "a sphere", "red cube"],
        [(16, 16), (8, 16), (12, 16)],
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        condition = inferencer.prepare_condition(prompts, shapes)
        noise = torch.randn(36, 8)
        result = model.forward_t2i_loop(
            x_t=noise, timestep=torch.full((36,), 0.6), **condition.inputs
        )
        for prompt, shape, image_noise, velocity in zip(
            prompts,
            shapes,
            noise.split([16, 8, 12]),
            result.velocity.split([16, 8, 12]),
        ):
            single = inferencer.prepare_condition([prompt], shape)
            reference = model.forward_t2i_loop(
                x_t=image_noise,
                timestep=torch.full((len(image_noise),), 0.6),
                **single.inputs,
            )
            torch.testing.assert_close(velocity, reference.velocity, rtol=0, atol=0)
        result.velocity.float().square().mean().backward()
    assert model.t2i_loop.output_alpha.grad is not None
    assert condition.shapes == shapes


def test_memory_projection_maps_each_sample_to_its_packed_gen_tokens():
    model, _ = tiny_model()
    adapter = model.t2i_loop.reentry
    with torch.no_grad():
        adapter.memory_projection.weight.copy_(torch.eye(32))
    delta = torch.zeros(5, 32, dtype=torch.bfloat16)
    memory = torch.stack([torch.ones(2, 32), torch.full((2, 32), 7.0)]).to(
        torch.bfloat16
    )
    projected = adapter(delta, memory, torch.tensor([2, 3]))
    torch.testing.assert_close(
        projected[:, 0], torch.tensor([1, 1, 7, 7, 7], dtype=torch.bfloat16)
    )


def test_full_text_dropout_matches_the_inference_text_removed_state():
    model, _ = tiny_model(2, alpha=0.3)
    inferencer = InterleaveInferencer(model, TinyVAE(), TinyTokenizer(), TOKENS)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        dropped = inferencer.prepare_condition(
            ["two cubes"], (16, 16), text_drop_mask=[True]
        )
        regular = inferencer.prepare_condition(["two cubes"], (16, 16))
        inputs = {
            key: value
            for key, value in regular.inputs.items()
            if not key.startswith("cfg_")
        }
        for target, source in [
            ("packed_position_ids", "packed_position_ids"),
            ("packed_indexes", "packed_query_indexes"),
            ("key_values_lens", "key_values_lens"),
            ("past_key_values", "past_key_values"),
            ("packed_key_value_indexes", "packed_key_value_indexes"),
        ]:
            inputs[target] = regular.inputs[f"cfg_text_{source}"]
        noise = torch.randn(16, 8)
        time = torch.full((16,), 0.6)
        actual = model.forward_t2i_loop(x_t=noise, timestep=time, **dropped.inputs)
        reference = model.forward_t2i_loop(x_t=noise, timestep=time, **inputs)
    assert dropped.inputs["key_values_lens"].tolist() == [0]
    assert all(
        value is None for value in dropped.inputs["past_key_values"].key_cache.values()
    )
    torch.testing.assert_close(actual.velocity, reference.velocity, rtol=0, atol=0)


def test_mixed_text_dropout_and_variable_shapes_match_independent_branches():
    model, _ = tiny_model(2, alpha=0.3)
    inferencer = InterleaveInferencer(model, TinyVAE(), TinyTokenizer(), TOKENS)
    prompts, shapes, drops = (
        ["two cubes", "red cube", "a sphere"],
        [(16, 16), (8, 16), (12, 16)],
        [False, True, False],
    )
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        batch = inferencer.prepare_condition(prompts, shapes, text_drop_mask=drops)
        assert batch.inputs["key_values_lens"].tolist()[1] == 0
        noise = torch.randn(36, 8)
        result = model.forward_t2i_loop(
            x_t=noise, timestep=torch.full((36,), 0.6), **batch.inputs
        )
        for prompt, shape, drop, part, actual in zip(
            prompts,
            shapes,
            drops,
            noise.split([16, 8, 12]),
            result.velocity.split([16, 8, 12]),
        ):
            single = inferencer.prepare_condition(
                [prompt], shape, text_drop_mask=[drop]
            )
            expected = model.forward_t2i_loop(
                x_t=part, timestep=torch.full((len(part),), 0.6), **single.inputs
            )
            torch.testing.assert_close(actual, expected.velocity, rtol=0, atol=0)


def test_native_resize_preserves_structural_frame_and_variable_collation(tmp_path):
    image = Image.new("RGB", (16, 8), "white")
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((15, 7), (0, 0, 255))
    image.save(tmp_path / "wide.png")
    Image.new("RGB", (8, 16), "red").save(tmp_path / "tall.png")
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"prompt": "two cubes", "image": name, "bucket": "structural"})
            for name in ["wide.png", "tall.png"]
        )
    )
    dataset = T2IDataset(path, 16, stride=4)
    wide, tall = dataset[0], dataset[1]
    assert wide["pixels"].shape == (3, 8, 16)
    assert tall["pixels"].shape == (3, 16, 8)
    assert wide["pixels"][:, 0, 0].tolist() == [1, -1, -1]
    assert wide["pixels"][:, -1, -1].tolist() == [-1, -1, 1]
    batch = collate_t2i([wide, tall])
    assert batch["bucket"] == ["structural", "structural"]
    assert batch["image_shape"] == [(8, 16), (16, 8)]


def test_bucket_sampler_respects_weights_and_seed(tmp_path):
    rows = [
        {"prompt": "a cube", "image": "image.png", "bucket": bucket}
        for bucket in ["ordinary", "structural", "easy", "noop"]
    ]
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    dataset = T2IDataset(path, 16, stride=4)
    first = list(BucketBatchSampler(dataset, 2, 6000, seed=91))
    second = list(BucketBatchSampler(dataset, 2, 6000, seed=91))
    assert first == second
    assert all(batch[0] == batch[1] for batch in first)
    for index, expected in enumerate([0.4, 0.3, 0.2, 0.1]):
        assert sum(batch[0] == index for batch in first) / 6000 == pytest.approx(
            expected, abs=0.025
        )
    dataset.rows.pop()
    with pytest.raises(ValueError, match="no training records"):
        BucketBatchSampler(dataset, 1, 1, weights={"noop": 1.0})


def test_slot_diagnostics_include_signed_cosine_std_and_updates():
    stats = memory_slot_stats(torch.tensor([[[1.0, 1.0], [-1.0, -1.0]]]))
    assert stats["mean_pairwise_cosine"] == pytest.approx(-1)
    assert stats["slot_std"] == 1
    assert stats["sigma1_ratio"] == pytest.approx(1)
    model, config = tiny_model()
    with torch.no_grad():
        result = model.forward_t2i_loop(**flow_inputs(model))
    assert all(row["memory_update_ratio"] > 0 for row in result.stats)


def test_training_with_text_removed_and_variable_shapes_logs_bucket_metrics(
    tmp_path, monkeypatch
):
    install_tiny_backbone(monkeypatch)
    Image.new("RGB", (16, 8), "red").save(tmp_path / "wide.png")
    Image.new("RGB", (8, 16), "blue").save(tmp_path / "tall.png")
    data = tmp_path / "data.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"prompt": "two cubes", "image": image, "bucket": "structural"})
            for image in ["wide.png", "tall.png"]
        )
    )
    model, config = tiny_model()
    cfg = dict(
        model_path="native",
        data_path=str(data),
        output_dir=str(tmp_path / "train"),
        image_size=16,
        batch_size=2,
        steps=2,
        save_every=2,
        runtime_loop_depth=3,
        device="cpu",
        text_cond_dropout_prob=1.0,
        bucket_weights={"structural": 1.0},
        depth_curriculum=[1, 2, 3],
        loop=config.to_dict(),
    )
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(cfg))
    module = script_module(ROOT / "scripts/train/train_t2i_loop.py")
    monkeypatch.setattr("sys.argv", ["train_t2i_loop.py", "--config", str(path)])
    module.main()
    records = [
        json.loads(line)
        for line in (tmp_path / "train/metrics.jsonl").read_text().splitlines()
    ]
    assert all(row["text_drop_mask"] == [True, True] for row in records)
    assert all(row["condition_losses"]["conditional"] is None for row in records)
    assert all(row["condition_losses"]["text_removed"] is not None for row in records)
    assert all(
        row["bucket_parameter_metrics"]["bucket"] == "structural" for row in records
    )
    assert all(row["grad_norm"] > 0 for row in records)
    assert all(row["loop_stats"] for row in records)
    assert all(
        "memory_update_ratio" in stats and "slot_std" in stats["memory_slot_stats"]
        for row in records
        for stats in row["loop_stats"]
    )
