"""Same-seed Euler samples: native, GEN-only step 400, and dual step 600.

One process draws one prompt. The three arms share the initial noise and the
prompt KV. Inside the trained window the loop replaces ``_forward_flow`` with
``predict_flow`` at one correction round. Steps outside that window stay on
the native velocity. This writes images. It does not score them.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .config import LoopConfig
from .loop import AnchoredDualStateLoop
from .native_path import BranchContext


ARMS = ("native", "gen_only_400", "dual_600_1round")


def euler_nodes(steps: int, shift: float) -> tuple[torch.Tensor, torch.Tensor]:
    """BAGEL's decreasing grid. ``steps`` includes the final 0, so updates are ``steps - 1``."""
    if steps < 2 or shift <= 0:
        raise ValueError("sampler needs at least two nodes and a positive shift")
    times = torch.linspace(1, 0, steps)
    times = shift * times / (1 + (shift - 1) * times)
    return times[:-1], times[:-1] - times[1:]


def steps_inside(nodes: torch.Tensor, start: float = 0.15, end: float = 0.65) -> int:
    """How many Euler updates have progress ``u = 1 - t`` inside the trained interval."""
    progress = 1.0 - nodes
    return int(((progress >= start) & (progress <= end)).sum())


def initial_noise(rows: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(rows, width, generator=generator)


def load_prompts(path: Path) -> list[str]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        text = str(row["prompt"]).strip()
        if not text:
            raise ValueError("prompt line is empty")
        prompts.append(text)
    if not prompts:
        raise ValueError("prompt file is empty")
    return prompts


def _prompt_cache(model, tokenizer, token_ids, prompt: str, device):
    from ..diagnostics.native import device_inputs, empty_cache
    from ..diagnostics.numerics import autocast_for

    decoder = model.language_model.model
    if prompt:
        inputs, lengths, positions = model.prepare_prompts(
            [0], [0], [prompt], tokenizer, token_ids,
        )
        cache = empty_cache(decoder)
        moved = device_inputs(inputs, device)
        with autocast_for(device):
            cache = model.forward_cache_update_text(cache, **moved)
        return cache, lengths, positions
    return empty_cache(decoder), [0], [0]


def flow_for_prompt(bundle, prompt: str, resolution: int, device) -> dict:
    """Conditional prompt cache plus an empty-text CFG branch. No image condition."""
    from ..diagnostics.native import device_inputs

    model = bundle.model
    cache, lengths, positions = _prompt_cache(
        model, bundle.tokenizer, bundle.token_ids, prompt, device,
    )
    flow = model.prepare_vae_latent(lengths, positions, [(resolution, resolution)], bundle.token_ids)
    flow.pop("packed_init_noises")
    flow = device_inputs(flow, device)
    flow["past_key_values"] = cache
    text_cache, text_lengths, text_positions = _prompt_cache(
        model, bundle.tokenizer, bundle.token_ids, "", device,
    )
    cfg = model.prepare_vae_latent_cfg(text_lengths, text_positions, [(resolution, resolution)])
    flow.update({
        key.replace("cfg_", "cfg_text_", 1): value
        for key, value in device_inputs(cfg, device).items()
    })
    flow["cfg_text_past_key_values"] = text_cache
    return flow


def _branches(flow: dict, flow_time: float) -> dict[str, BranchContext]:
    """``cond`` and ``cfg_text``. Image CFG is off, so there is no third branch."""
    shared = dict(
        query_lens=flow["packed_seqlens"],
        packed_text_indexes=flow["packed_text_indexes"],
        packed_vae_token_indexes=flow["packed_vae_token_indexes"],
        flow_time=flow_time,
    )
    cond = BranchContext(
        name="cond",
        packed_query_position_ids=flow["packed_position_ids"],
        packed_query_indexes=flow["packed_indexes"],
        past_key_values=flow["past_key_values"],
        key_values_lens=flow["key_values_lens"],
        packed_key_value_indexes=flow["packed_key_value_indexes"],
        **shared,
    )
    cfg_text = BranchContext(
        name="cfg_text",
        packed_query_position_ids=flow["cfg_text_packed_position_ids"],
        packed_query_indexes=flow["cfg_text_packed_query_indexes"],
        past_key_values=flow["cfg_text_past_key_values"],
        key_values_lens=flow["cfg_text_key_values_lens"],
        packed_key_value_indexes=flow["cfg_text_packed_key_value_indexes"],
        **shared,
    )
    return {"cond": cond, "cfg_text": cfg_text}


def _native_velocity(model, flow, x, flow_time, cfg_text_scale, cfg_renorm_type, cfg_renorm_min):
    timestep = torch.full((x.shape[0],), flow_time, device=x.device)
    return model._forward_flow(
        x_t=x,
        timestep=timestep,
        packed_vae_token_indexes=flow["packed_vae_token_indexes"],
        packed_vae_position_ids=flow["packed_vae_position_ids"],
        packed_text_ids=flow["packed_text_ids"],
        packed_text_indexes=flow["packed_text_indexes"],
        packed_indexes=flow["packed_indexes"],
        packed_position_ids=flow["packed_position_ids"],
        packed_seqlens=flow["packed_seqlens"],
        key_values_lens=flow["key_values_lens"],
        past_key_values=flow["past_key_values"],
        packed_key_value_indexes=flow["packed_key_value_indexes"],
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
        cfg_text_scale=cfg_text_scale,
        cfg_text_packed_position_ids=flow["cfg_text_packed_position_ids"],
        cfg_text_packed_query_indexes=flow["cfg_text_packed_query_indexes"],
        cfg_text_key_values_lens=flow["cfg_text_key_values_lens"],
        cfg_text_past_key_values=flow["cfg_text_past_key_values"],
        cfg_text_packed_key_value_indexes=flow["cfg_text_packed_key_value_indexes"],
        cfg_img_scale=1.0,
    )


def _loop_velocity(loop, model, flow, x, flow_time, cfg_text_scale, cfg_renorm_type, cfg_renorm_min):
    timestep = torch.full((x.shape[0],), flow_time, device=x.device)
    return loop.predict_flow(
        model,
        x,
        timestep,
        flow["packed_text_ids"],
        flow["packed_text_indexes"],
        flow["packed_vae_token_indexes"],
        flow["packed_vae_position_ids"],
        flow["packed_seqlens"],
        _branches(flow, flow_time),
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=1.0,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
        train=False,
    )


def sample_arm(model, flow, noise, nodes, dts, velocity_fn, device) -> torch.Tensor:
    from ..diagnostics.numerics import autocast_for

    state = noise.to(device)
    for flow_time, dt in zip(nodes.tolist(), dts.tolist()):
        with torch.inference_mode(), autocast_for(device):
            velocity = velocity_fn(state, float(flow_time))
        state = state - velocity.to(state.dtype) * dt
    return state


def decode_latent(bundle, latent: torch.Tensor, resolution: int, device) -> Image.Image:
    from ..diagnostics.numerics import autocast_for

    model = bundle.model
    patch = model.latent_patch_size
    side = resolution // model.latent_downsample
    channels = model.latent_channel
    grid = torch.einsum(
        "hwpqc->chpwq",
        latent.reshape(side, side, patch, patch, channels),
    ).reshape(1, channels, side * patch, side * patch)
    with torch.inference_mode(), autocast_for(device):
        image = bundle.vae.decode(grid)
    array = ((image[0].float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().byte().numpy()
    return Image.fromarray(array)


def _wrap_prompt(draw: ImageDraw.ImageDraw, prompt: str, width: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in prompt.split():
        trial = word if not line else f"{line} {word}"
        if draw.textlength(trial) > width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def comparison_strip(prompt: str, panels: list[tuple[str, Image.Image]]) -> Image.Image:
    """Four columns: the prompt, then native, gen_only_400, and dual_600 at one round."""
    if len(panels) != 3:
        raise ValueError("the strip is the prompt plus the three named arms")
    text_w = 300
    label_h = 32
    image_h = panels[0][1].height
    scratch = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines = _wrap_prompt(scratch, prompt, text_w - 24)
    text_h = 12 + 18 * len(lines) + 12
    width = text_w + sum(image.width for _, image in panels)
    canvas = Image.new("RGB", (width, max(label_h + image_h, text_h)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        draw.text((12, 12 + 18 * index), line, fill=(0, 0, 0))
    x = text_w
    for name, image in panels:
        draw.text((x + 12, 8), name, fill=(0, 0, 0))
        canvas.paste(image, (x, label_h))
        x += image.width
    return canvas


def _apply_arm(loop, blob, variant: str) -> None:
    loop.load_state(blob)
    loop.config.variant = variant
    loop.config.correction_rounds = 1
    loop.config.feedback_mode = "correct"
    loop.config.force_native_bypass = False
    loop.config.force_full_zero_path = False
    loop.config.validate(len(loop.decoder.layers))
    loop.eval()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gen-only-checkpoint", type=Path, required=True)
    parser.add_argument("--dual-checkpoint", type=Path, required=True)
    parser.add_argument("--observation-checkpoint", type=Path, required=True)
    parser.add_argument("--loop-config", type=Path, default=Path("configs/anchored_gen_only.json"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--sampler-steps", type=int, default=50)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-renorm-type", default="global")
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--device", default="npu")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    prompts = load_prompts(args.prompts)
    if rank >= len(prompts):
        raise SystemExit(f"rank {rank} has no prompt; file has {len(prompts)}")
    prompt = prompts[rank]
    from ..bagel.accelerator import set_device
    from ..bagel.backbone import load_native

    device = set_device(f"{args.device}:{local}" if args.device != "cpu" else "cpu")
    bundle = load_native(args.model, device, args.timestep_shift)
    decoder = bundle.model.language_model.model
    decoder.enable_taylorseer = False
    for layer in decoder.layers:
        layer.enable_taylorseer = False
    loop_config = LoopConfig.from_json(args.loop_config)
    loop_config.observation_checkpoint = str(args.observation_checkpoint)
    loop_config.observation_source_mode = "gen_full_query"
    loop_config.variant = "gen_only"
    loop_config.correction_rounds = 1
    loop_config.enabled = True
    loop = AnchoredDualStateLoop(bundle.model.language_model, loop_config).to(device).eval()
    gen_blob = torch.load(args.gen_only_checkpoint, map_location="cpu", weights_only=False)
    dual_blob = torch.load(args.dual_checkpoint, map_location="cpu", weights_only=False)
    expected_step = {"gen_only": 400, "dual": 600}
    for blob, stage in ((gen_blob, "gen_only"), (dual_blob, "dual")):
        if blob.get("trainer") != "anchored_curriculum":
            raise SystemExit(f"{stage} checkpoint trainer is {blob.get('trainer')}")
        if blob.get("stage") != stage or blob.get("variant") != stage or int(blob.get("step", -1)) != expected_step[stage]:
            raise SystemExit(
                f"{stage} checkpoint is variant {blob.get('variant')} "
                f"stage {blob.get('stage')} step {blob.get('step')}"
            )
    flow = flow_for_prompt(bundle, prompt, args.resolution, device)
    nodes, dts = euler_nodes(args.sampler_steps, args.timestep_shift)
    noise = initial_noise(
        int(flow["packed_vae_token_indexes"].shape[0]),
        bundle.model.latent_channel * bundle.model.latent_patch_size ** 2,
        args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    images = []

    def native_fn(state, flow_time):
        return _native_velocity(
            bundle.model, flow, state, flow_time,
            args.cfg_text_scale, args.cfg_renorm_type, args.cfg_renorm_min,
        )

    latent = sample_arm(bundle.model, flow, noise, nodes, dts, native_fn, device)
    images.append(("native", decode_latent(bundle, latent, args.resolution, device)))
    for blob, variant, title in (
        (gen_blob, "gen_only", "gen_only_400"),
        (dual_blob, "dual", "dual_600_1round"),
    ):
        _apply_arm(loop, blob, variant)

        def loop_fn(state, flow_time, _loop=loop):
            return _loop_velocity(
                _loop, bundle.model, flow, state, flow_time,
                args.cfg_text_scale, args.cfg_renorm_type, args.cfg_renorm_min,
            )

        latent = sample_arm(bundle.model, flow, noise, nodes, dts, loop_fn, device)
        images.append((title, decode_latent(bundle, latent, args.resolution, device)))
    strip = comparison_strip(f"[{rank:02d}] {prompt}", images)
    image_path = args.output / f"rank_{rank:02d}.png"
    strip.save(image_path)
    record = {
        "rank": rank,
        "world": world,
        "prompt": prompt,
        "seed": args.seed,
        "arms": [name for name, _image in images],
        "sampler_steps": args.sampler_steps,
        "euler_updates": int(nodes.shape[0]),
        "updates_inside_u_0.15_0.65": steps_inside(nodes),
        "correction_rounds": 1,
        "cfg_text_scale": args.cfg_text_scale,
        "cfg_img_scale": 1.0,
        "image": str(image_path),
    }
    (args.output / f"rank_{rank:02d}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(
        f"rank {rank} wrote {image_path.name} inside={record['updates_inside_u_0.15_0.65']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
