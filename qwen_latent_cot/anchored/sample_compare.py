"""Same-seed samples through ``Bagel.generate_image``.

One process draws one prompt. Native keeps ``_forward_flow``. GEN-only, dual,
and dual-shuffled temporarily replace that method with
``AnchoredDualStateLoop.predict_flow``. The Euler update stays inside
``generate_image``. Shuffled packs the prompt with one fixed donor so the
existing batch roll has two samples, then keeps only the prompt image.
This writes images and amplified differences. It does not score quality.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .config import LoopConfig
from .loop import AnchoredDualStateLoop
from .native_path import BranchContext


ARMS = ("native", "gen_only_400", "dual_600", "dual_shuffled")
# Fixed second sample for the batch roll. Its picture is discarded.
SHUFFLE_DONOR_PROMPT = "A plain white ceramic plate on a gray table."


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


def _text_cache(model, tokenizer, token_ids, prompts: list[str], device):
    from ..diagnostics.native import device_inputs, empty_cache
    from ..diagnostics.numerics import autocast_for

    decoder = model.language_model.model
    count = len(prompts)
    if not any(prompts):
        return empty_cache(decoder), [0] * count, [0] * count
    if not all(prompts):
        raise ValueError("a packed text cache is either all prompts or all empty")
    inputs, lengths, positions = model.prepare_prompts(
        [0] * count, [0] * count, prompts, tokenizer, token_ids,
    )
    cache = empty_cache(decoder)
    with autocast_for(device):
        cache = model.forward_cache_update_text(cache, **device_inputs(inputs, device))
    return cache, lengths, positions


def flow_for_prompts(bundle, prompts: list[str], resolution: int, device) -> dict:
    """Conditional prompt caches plus an empty-text CFG branch. No image condition."""
    from ..diagnostics.native import device_inputs

    model = bundle.model
    sizes = [(resolution, resolution)] * len(prompts)
    cache, lengths, positions = _text_cache(
        model, bundle.tokenizer, bundle.token_ids, prompts, device,
    )
    flow = model.prepare_vae_latent(lengths, positions, sizes, bundle.token_ids)
    flow.pop("packed_init_noises")
    flow = device_inputs(flow, device)
    flow["past_key_values"] = cache
    text_cache, text_lengths, text_positions = _text_cache(
        model, bundle.tokenizer, bundle.token_ids, [""] * len(prompts), device,
    )
    cfg = model.prepare_vae_latent_cfg(text_lengths, text_positions, sizes)
    flow.update({
        key.replace("cfg_", "cfg_text_", 1): value
        for key, value in device_inputs(cfg, device).items()
    })
    flow["cfg_text_past_key_values"] = text_cache
    return flow


def flow_for_prompt(bundle, prompt: str, resolution: int, device) -> dict:
    return flow_for_prompts(bundle, [prompt], resolution, device)


def forward_flow_branches(call: dict) -> dict[str, BranchContext]:
    """Map one ``Bagel._forward_flow`` call onto ``predict_flow`` branches.

    Text and image rows stay the conditional layout, matching ``_forward_flow``.
    A CFG branch is built only when its scale is above 1, which is also when
    the native method runs that extra forward. Image CFG is absent for T2I.
    """
    flow_time = float(call["timestep"].reshape(-1)[0])
    shared = dict(
        query_lens=call["packed_seqlens"],
        packed_text_indexes=call["packed_text_indexes"],
        packed_vae_token_indexes=call["packed_vae_token_indexes"],
        flow_time=flow_time,
    )
    branches = {
        "cond": BranchContext(
            name="cond",
            packed_query_position_ids=call["packed_position_ids"],
            packed_query_indexes=call["packed_indexes"],
            past_key_values=call["past_key_values"],
            key_values_lens=call["key_values_lens"],
            packed_key_value_indexes=call["packed_key_value_indexes"],
            **shared,
        ),
    }
    if call.get("cfg_text_scale", 1.0) > 1.0:
        branches["cfg_text"] = BranchContext(
            name="cfg_text",
            packed_query_position_ids=call["cfg_text_packed_position_ids"],
            packed_query_indexes=call["cfg_text_packed_query_indexes"],
            past_key_values=call["cfg_text_past_key_values"],
            key_values_lens=call["cfg_text_key_values_lens"],
            packed_key_value_indexes=call["cfg_text_packed_key_value_indexes"],
            **shared,
        )
    if call.get("cfg_img_scale", 1.0) > 1.0:
        branches["cfg_img"] = BranchContext(
            name="cfg_img",
            packed_query_position_ids=call["cfg_img_packed_position_ids"],
            packed_query_indexes=call["cfg_img_packed_query_indexes"],
            past_key_values=call["cfg_img_past_key_values"],
            key_values_lens=call["cfg_img_key_values_lens"],
            packed_key_value_indexes=call["cfg_img_packed_key_value_indexes"],
            **shared,
        )
    return branches


@contextmanager
def attach_predict_flow(model, loop):
    """Point ``generate_image``'s velocity call at ``loop.predict_flow``.

    ``generate_image`` keeps the 50-step Euler update. The replacement is
    removed on the way out, including when sampling raises.
    """
    original = model._forward_flow

    def _anchored_forward_flow(
        x_t,
        timestep,
        packed_vae_token_indexes,
        packed_vae_position_ids,
        packed_text_ids,
        packed_text_indexes,
        packed_indexes,
        packed_position_ids,
        packed_seqlens,
        key_values_lens,
        past_key_values,
        packed_key_value_indexes,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        cfg_text_scale=1.0,
        cfg_text_packed_position_ids=None,
        cfg_text_packed_query_indexes=None,
        cfg_text_key_values_lens=None,
        cfg_text_past_key_values=None,
        cfg_text_packed_key_value_indexes=None,
        cfg_img_scale=1.0,
        cfg_img_packed_position_ids=None,
        cfg_img_packed_query_indexes=None,
        cfg_img_key_values_lens=None,
        cfg_img_past_key_values=None,
        cfg_img_packed_key_value_indexes=None,
        cfg_type="parallel",
        **_cache,
    ):
        del cfg_type, _cache
        branches = forward_flow_branches({
            "timestep": timestep,
            "packed_vae_token_indexes": packed_vae_token_indexes,
            "packed_text_indexes": packed_text_indexes,
            "packed_indexes": packed_indexes,
            "packed_position_ids": packed_position_ids,
            "packed_seqlens": packed_seqlens,
            "key_values_lens": key_values_lens,
            "past_key_values": past_key_values,
            "packed_key_value_indexes": packed_key_value_indexes,
            "cfg_text_scale": cfg_text_scale,
            "cfg_text_packed_position_ids": cfg_text_packed_position_ids,
            "cfg_text_packed_query_indexes": cfg_text_packed_query_indexes,
            "cfg_text_key_values_lens": cfg_text_key_values_lens,
            "cfg_text_past_key_values": cfg_text_past_key_values,
            "cfg_text_packed_key_value_indexes": cfg_text_packed_key_value_indexes,
            "cfg_img_scale": cfg_img_scale,
            "cfg_img_packed_position_ids": cfg_img_packed_position_ids,
            "cfg_img_packed_query_indexes": cfg_img_packed_query_indexes,
            "cfg_img_key_values_lens": cfg_img_key_values_lens,
            "cfg_img_past_key_values": cfg_img_past_key_values,
            "cfg_img_packed_key_value_indexes": cfg_img_packed_key_value_indexes,
        })
        return loop.predict_flow(
            model,
            x_t,
            timestep,
            packed_text_ids,
            packed_text_indexes,
            packed_vae_token_indexes,
            packed_vae_position_ids,
            packed_seqlens,
            branches,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            train=False,
        )

    model._forward_flow = _anchored_forward_flow
    try:
        yield
    finally:
        model._forward_flow = original


def generate_latents(model, flow: dict, noise: torch.Tensor, args, device):
    """One ``generate_image`` call. The Euler step is ``x - v * dt`` inside Bagel."""
    from ..diagnostics.numerics import autocast_for

    with torch.inference_mode(), autocast_for(device):
        unpacked = model.generate_image(
            packed_text_ids=flow["packed_text_ids"],
            packed_text_indexes=flow["packed_text_indexes"],
            packed_init_noises=noise.to(device),
            packed_vae_position_ids=flow["packed_vae_position_ids"],
            packed_vae_token_indexes=flow["packed_vae_token_indexes"],
            packed_seqlens=flow["packed_seqlens"],
            packed_position_ids=flow["packed_position_ids"],
            packed_indexes=flow["packed_indexes"],
            past_key_values=flow["past_key_values"],
            key_values_lens=flow["key_values_lens"],
            packed_key_value_indexes=flow["packed_key_value_indexes"],
            num_timesteps=args.sampler_steps,
            timestep_shift=args.timestep_shift,
            cfg_renorm_min=args.cfg_renorm_min,
            cfg_renorm_type=args.cfg_renorm_type,
            cfg_interval=(0.0, 1.0),
            cfg_text_scale=args.cfg_text_scale,
            cfg_text_packed_query_indexes=flow["cfg_text_packed_query_indexes"],
            cfg_text_packed_position_ids=flow["cfg_text_packed_position_ids"],
            cfg_text_past_key_values=flow["cfg_text_past_key_values"],
            cfg_text_key_values_lens=flow["cfg_text_key_values_lens"],
            cfg_text_packed_key_value_indexes=flow["cfg_text_packed_key_value_indexes"],
            cfg_img_scale=1.0,
            enable_taylorseer=False,
        )
    return unpacked


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


def _rgb(image: Image.Image) -> torch.Tensor:
    import numpy as np

    return torch.from_numpy(np.asarray(image, dtype=np.float32))


def mean_absolute_difference(left: Image.Image, right: Image.Image) -> float:
    if left.size != right.size:
        raise ValueError("difference images differ in size")
    return float((_rgb(left) - _rgb(right)).abs().mean())


def difference_heatmap(left: Image.Image, right: Image.Image, gain: float = 10.0) -> Image.Image:
    """Per-pixel mean |left - right|, multiplied by ``gain``, drawn black-red-yellow-white."""
    if left.size != right.size:
        raise ValueError("heatmap images differ in size")
    level = ((_rgb(left) - _rgb(right)).abs().mean(dim=-1) * gain / 255.0).clamp(0, 1)
    red = (level * 3).clamp(0, 1)
    green = (level * 3 - 1).clamp(0, 1)
    blue = (level * 3 - 2).clamp(0, 1)
    rgb = torch.stack((red, green, blue), dim=-1).mul(255).byte().numpy()
    return Image.fromarray(rgb)


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
    """Prompt column, then one column per panel. Panels carry their own labels."""
    if not panels:
        raise ValueError("the strip needs at least one image")
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


def _apply_arm(loop, blob, variant: str, feedback_mode: str) -> None:
    loop.load_state(blob)
    loop.config.variant = variant
    loop.config.correction_rounds = 1
    loop.config.feedback_mode = feedback_mode
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
    nodes, _dts = euler_nodes(args.sampler_steps, args.timestep_shift)
    noise = initial_noise(
        int(flow["packed_vae_token_indexes"].shape[0]),
        bundle.model.latent_channel * bundle.model.latent_patch_size ** 2,
        args.seed,
    )
    donor_flow = flow_for_prompts(
        bundle, [prompt, SHUFFLE_DONOR_PROMPT], args.resolution, device,
    )
    donor_noise = torch.cat((noise, noise), dim=0)
    if donor_flow["packed_vae_token_indexes"].shape[0] != donor_noise.shape[0]:
        raise RuntimeError("shuffled pack does not have two copies of the prompt latent")
    args.output.mkdir(parents=True, exist_ok=True)

    def one_image(flow_inputs, packed_noise, use_loop: bool) -> Image.Image:
        context = attach_predict_flow(bundle.model, loop) if use_loop else nullcontext()
        with context:
            unpacked = generate_latents(bundle.model, flow_inputs, packed_noise, args, device)
        return decode_latent(bundle, unpacked[0], args.resolution, device)

    native = one_image(flow, noise, False)
    _apply_arm(loop, gen_blob, "gen_only", "correct")
    gen_only = one_image(flow, noise, True)
    _apply_arm(loop, dual_blob, "dual", "correct")
    dual = one_image(flow, noise, True)
    _apply_arm(loop, dual_blob, "dual", "shuffled")
    shuffled = one_image(donor_flow, donor_noise, True)
    images = [
        ("native", native),
        ("gen_only_400", gen_only),
        ("dual_600", dual),
        ("dual_shuffled", shuffled),
        ("|dual-native|x10", difference_heatmap(dual, native)),
        ("|dual-gen|x10", difference_heatmap(dual, gen_only)),
    ]
    strip = comparison_strip(f"[{rank:02d}] {prompt}", images)
    image_path = args.output / f"rank_{rank:02d}.png"
    strip.save(image_path)
    record = {
        "rank": rank,
        "world": world,
        "prompt": prompt,
        "seed": args.seed,
        "arms": [name for name, _image in images],
        "sampler": "Bagel.generate_image",
        "loop_velocity": "AnchoredDualStateLoop.predict_flow",
        "sampler_steps": args.sampler_steps,
        "euler_updates": int(nodes.shape[0]),
        "updates_inside_u_0.15_0.65": steps_inside(nodes),
        "correction_rounds": 1,
        "cfg_text_scale": args.cfg_text_scale,
        "cfg_img_scale": 1.0,
        "shuffle_donor_prompt": SHUFFLE_DONOR_PROMPT,
        "mean_abs_dual_native": mean_absolute_difference(dual, native),
        "mean_abs_dual_gen_only": mean_absolute_difference(dual, gen_only),
        "heatmap_gain": 10.0,
        "image": str(image_path),
    }
    (args.output / f"rank_{rank:02d}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(
        f"rank {rank} wrote {image_path.name} inside={record['updates_inside_u_0.15_0.65']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
