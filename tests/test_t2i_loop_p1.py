import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from safetensors.torch import load_file
from test_anchored_t2i_loop import flow_inputs, tiny_model
from test_t2i_loop_entrypoints import install_tiny_backbone, script_module

import qwen_latent_cot.bagel.anchored_loop as runner
from qwen_latent_cot.bagel.anchored_loop import (
    AnchorState,
    LoopConfig,
    LoopModules,
    run_anchored_loop,
)
from qwen_latent_cot.bagel.depth_curriculum import (
    resolve_depth_curriculum,
    sample_curriculum_depth,
)
from qwen_latent_cot.bagel.loop_checkpoint import save_loop_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_shuffled_donor_is_fixed_and_reader_writes_never_recycle():
    config = LoopConfig(
        enable_t2i_loop=True,
        loop_start_layer=0,
        loop_end_layer=1,
        loop_depth=3,
        memory_slots=1,
        loop_mode="memory_only",
        memory_control="shuffled",
        loop_output_alpha_init=1,
        log_loop_stats=True,
    )
    modules = LoopModules(1, config)
    with torch.no_grad():
        modules.memory_init.zero_()
    anchor = AnchorState(torch.tensor([[[1.0]], [[7.0]]]), torch.zeros(2, 1, 1))
    writer_inputs, reader_inputs = [], []

    def body(
        gen,
        memory,
        _,
        capture_memory_reads=None,
        memory_read_overrides=None,
        memory_reference_reads=None,
    ):
        if capture_memory_reads is not None:
            writer_inputs.append(memory.clone())
            capture_memory_reads.append(memory.clone())
            return gen + memory, memory + gen, []
        reader_inputs.append(memory_read_overrides[0].clone())
        # A poisoned reader write must never reach a later canonical writer.
        return gen + memory_read_overrides[0], torch.full_like(memory, 999), []

    result = run_anchored_loop(anchor, modules, config, body, lambda x: x)
    for index in range(3):
        torch.testing.assert_close(
            writer_inputs[index].flatten(), torch.tensor([1.0, 7.0]) * index
        )
        torch.testing.assert_close(
            reader_inputs[index].flatten(), torch.tensor([7.0, 1.0]) * index
        )
    assert all(log["memory_permutation"] == [1, 0] for log in result.stats)
    assert all(log["memory_writer"] == "canonical_correct" for log in result.stats)
    # Identical initial M offers no R1 causal signal in a one-layer body.
    torch.testing.assert_close(result.velocities[0], anchor.gen_entry)


def test_real_shuffled_reads_differ_but_canonical_writer_matches_correct(monkeypatch):
    model, config = tiny_model(2, alpha=1)
    with torch.no_grad():
        model.t2i_loop.gate_logits.zero_()  # Visible intervention under BF16.
    inputs = flow_inputs(model)
    recorded = []
    original = runner.memory_slot_stats

    def capture(value):
        recorded.append(value.clone())
        return original(value)

    monkeypatch.setattr(runner, "memory_slot_stats", capture)
    with torch.no_grad():
        correct = model.forward_t2i_loop(**inputs)
        correct_states = recorded[:]
        recorded.clear()
        shuffled = model.forward_t2i_loop(
            **inputs, loop_config=replace(config, memory_control="shuffled")
        )
    assert len(recorded) == len(correct_states) == 6
    for actual, expected in zip(recorded, correct_states):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert shuffled.stats[0]["layers"][0]["memory_read_delta_ratio"] == 0
    assert shuffled.stats[0]["layers"][1]["memory_read_delta_ratio"] > 0
    assert not torch.equal(shuffled.velocities[0], correct.velocities[0])
    assert all(row["memory_permutation"] == [1, 0] for row in shuffled.stats)


@pytest.mark.parametrize(
    "mode,slots", [("direct_native_gen_only", 0), ("direct_native_memory", 2)]
)
def test_direct_native_is_full_recycle_without_new_gate_or_alpha(
    mode, slots, monkeypatch
):
    model, config = tiny_model(slots, mode, alpha=0)

    def forbidden(*args):
        raise AssertionError("direct native must not use the anchored gate")

    monkeypatch.setattr(model.t2i_loop, "gate", forbidden)
    inputs = flow_inputs(model)
    with torch.no_grad():
        model.t2i_loop.gate_logits.fill_(-torch.inf)
        result = model.forward_t2i_loop(**inputs)
        model.t2i_loop.config = replace(config, log_loop_stats=False)
        via_sampler = model._forward_flow(**inputs)
        model.t2i_loop.output_alpha.fill_(99)
        model.t2i_loop.gate_logits.fill_(9)
        again = model.forward_t2i_loop(**inputs)
    torch.testing.assert_close(via_sampler, result.velocity, rtol=0, atol=0)
    torch.testing.assert_close(again.velocity, result.velocity, rtol=0, atol=0)
    assert not torch.equal(result.velocity, result.base_velocity)
    assert all(log["alpha"] is None for log in result.stats)
    assert all(layer["gate"] is None for log in result.stats for layer in log["layers"])
    assert config.memory_slots == slots


def test_direct_native_recycles_exit_without_base_blend():
    config = LoopConfig(
        enable_t2i_loop=True,
        loop_start_layer=0,
        loop_end_layer=1,
        loop_mode="direct_native_gen_only",
        memory_slots=0,
        loop_depth=2,
        loop_gate_init=0,
        loop_output_alpha_init=0,
    )
    modules = LoopModules(1, config)
    anchor = AnchorState(torch.ones(1, 1, 1), torch.full((1, 1, 1), 10.0))
    entries = []

    def body(gen, memory, _):
        entries.append(gen.clone())
        return gen + 2, memory, []

    result = run_anchored_loop(anchor, modules, config, body, lambda x: x)
    assert [entry.item() for entry in entries] == [10, 12]
    assert [velocity.item() for velocity in result.velocities] == [12, 14]


def test_default_curriculum_boundaries_and_every_new_depth_is_exercised():
    cfg = yaml.safe_load((ROOT / "configs/training/t2i_loop_stage1.yaml").read_text())
    phases = resolve_depth_curriculum(
        cfg["depth_curriculum"], cfg["loop"]["loop_depth"], 1000
    )
    assert [(p["start_step"], p["end_step"], p["depths"]) for p in phases] == [
        (0, 300, [1]),
        (300, 700, [1, 2]),
        (700, 1000, [1, 2, 3]),
    ]
    assert sample_curriculum_depth(phases, 0) == (1, 0)
    assert sample_curriculum_depth(phases, 300) == (2, 1)
    assert sample_curriculum_depth(phases, 700) == (3, 2)
    for step in [299, 699, 999]:
        depth, phase = sample_curriculum_depth(phases, step)
        assert depth in phases[phase]["depths"]


@pytest.mark.parametrize(
    "curriculum,max_depth,steps",
    [
        ([1, 2], 3, 1000),
        (
            [
                {"until": 0.3, "depths": [1]},
                {"until": 0.7, "depths": [1, 2]},
                {"until": 1.0, "depths": [1, 2, 3]},
            ],
            3,
            2,
        ),
        ([{"until": 0.7, "depths": [1]}], 1, 10),
    ],
)
def test_invalid_or_untrained_curriculum_fails_before_loading(
    curriculum, max_depth, steps
):
    with pytest.raises(ValueError):
        resolve_depth_curriculum(curriculum, max_depth, steps)


def test_stage1_scheduled_training_updates_third_alpha_and_records_coverage(
    tmp_path, monkeypatch
):
    install_tiny_backbone(monkeypatch)
    Image.new("RGB", (16, 16), "red").save(tmp_path / "image.png")
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"prompt": "two cubes", "image": "image.png"}) + "\n")
    config = LoopConfig(
        enable_t2i_loop=True,
        loop_start_layer=1,
        loop_end_layer=3,
        loop_depth=3,
        memory_slots=2,
    )
    defaults = yaml.safe_load(
        (ROOT / "configs/training/t2i_loop_stage1.yaml").read_text()
    )
    cfg = dict(
        model_path=str(tmp_path / "base"),
        data_path=str(data),
        output_dir=str(tmp_path / "train"),
        device="cpu",
        image_size=16,
        steps=10,
        save_every=3,
        learning_rate=0.01,
        depth_curriculum=defaults["depth_curriculum"],
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
    assert [records[index]["depth"] for index in [0, 3, 7]] == [1, 2, 3]
    early = json.loads((tmp_path / "train/step_000003/loop.json").read_text())
    final = json.loads((tmp_path / "train/step_000010/loop.json").read_text())
    assert early["round_training_steps"] == [3, 0, 0]
    assert final["round_training_steps"][2] > 0
    assert final["training_depth_counts"]["3"] > 0
    assert (
        load_file(str(tmp_path / "train/step_000010/loop.safetensors"))["output_alpha"][
            2
        ]
        != 0
    )


@pytest.mark.parametrize("alpha,gate", [(0.1, 0.02), (0.9, 0.8)])
def test_matrix_legacy_and_pure_direct_ignore_anchored_cli_parameters(
    alpha, gate, tmp_path, monkeypatch
):
    install_tiny_backbone(monkeypatch)
    seen = []
    # Intercept live forwards to capture evidence from the actual matrix entry.
    from qwen_latent_cot.bagel.modeling import Bagel

    original = Bagel.forward_t2i_loop

    def capture(model, **kwargs):
        result = original(model, **kwargs)
        seen.append((kwargs["loop_config"], result.velocity.clone()))
        return result

    monkeypatch.setattr(Bagel, "forward_t2i_loop", capture)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt": "two cubes"}) + "\n")
    module = script_module(ROOT / "scripts/evaluate/t2i_loop_matrix.py")
    for index, (a, g) in enumerate([(alpha, gate), (0.0, 0.0)]):
        monkeypatch.setattr(
            "sys.argv",
            [
                "t2i_loop_matrix.py",
                "--model-path",
                "native",
                "--prompts",
                str(prompts),
                "--output-dir",
                str(tmp_path / str(index)),
                "--device",
                "cpu",
                "--image-size",
                "16",
                "--num-timesteps",
                "2",
                "--start-layer",
                "1",
                "--end-layer",
                "3",
                "--depths",
                "1",
                "--memory-slots",
                "2",
                "--modes",
                "legacy_memory_only,direct_native_gen_only",
                "--alpha",
                str(a),
                "--gate",
                str(g),
            ],
        )
        module.main()
    assert len(seen) == 6  # base + two controls, one diffusion step, two runs
    for offset in [1, 2]:
        torch.testing.assert_close(seen[offset][1], seen[offset + 3][1], rtol=0, atol=0)
    assert seen[2][0].memory_slots == 0
    manifest = json.loads((tmp_path / "0/manifest.json").read_text())
    assert (
        manifest["arms"]["legacy_memory_only_R1_K2_correct"]["readout"]
        == "parent_suffix_no_alpha"
    )
    assert manifest["arms"]["direct_native_gen_only_R1_K0_correct"]["gate"] == "ungated"


def test_matrix_rejects_checkpoint_rounds_not_yet_trained(tmp_path, monkeypatch):
    model, _ = tiny_model()
    save_loop_checkpoint(
        model,
        tmp_path / "checkpoint",
        step=3,
        model_path="native",
        training_depth_counts={"1": 3, "2": 0, "3": 0},
        round_training_steps=[3, 0, 0],
        depth_curriculum=[],
    )
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt": "two cubes"}) + "\n")
    module = script_module(ROOT / "scripts/evaluate/t2i_loop_matrix.py")
    monkeypatch.setattr(
        "sys.argv",
        [
            "t2i_loop_matrix.py",
            "--model-path",
            "native",
            "--prompts",
            str(prompts),
            "--output-dir",
            str(tmp_path / "eval"),
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--modes",
            "gen_memory_anchored",
            "--depths",
            "3",
            "--memory-slots",
            "2",
        ],
    )
    with pytest.raises(ValueError, match="untrained round"):
        module.main()
