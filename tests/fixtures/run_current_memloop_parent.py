"""Independent oracle: executed in a subprocess importing the frozen Git parent."""

import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, sys.argv[1])
from qwen_latent_cot.bagel.modeling import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
)
from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import NaiveCache

payload = torch.load(sys.argv[2], weights_only=False)
outputs = []
for case in payload:
    torch.manual_seed(case["seed"])
    config = Qwen2Config(**case["llm_config"])
    model = (
        Bagel(
            Qwen2ForCausalLM(config),
            None,
            BagelConfig(
                llm_config=config,
                visual_und=False,
                vae_config=SimpleNamespace(z_channels=2, downsample=2),
                max_latent_size=8,
                num_loop_tokens=case["slots"],
                loop_depth=case["depth"] + 1,
                memory_loop_start_layer=1,
                memory_loop_end_layer=3,
            ),
        )
        .to(torch.bfloat16)
        .eval()
    )
    missing, unexpected = model.load_state_dict(case["weights"], strict=False)
    assert missing == ["loop_memory"] and not unexpected, (missing, unexpected)
    model.init_loop_memory_from_boundary_embeddings([1, 2], seed=0)
    batch = case["batch"]
    sizes = [(16, 16)] * batch
    inputs = model.prepare_vae_latent(
        [3 + i for i in range(batch)],
        [5 + i for i in range(batch)],
        sizes,
        {"start_of_image": 1, "end_of_image": 2},
    )
    inputs.pop("packed_init_noises")
    # Old preparation also exposes sampler bookkeeping; the flow accepts neither.
    inputs.pop("packed_vae_seqlens", None)

    def cache(values):
        result = NaiveCache(4)
        result.key_cache, result.value_cache = (
            values["key_cache"],
            values["value_cache"],
        )
        return result

    inputs.update(
        x_t=case["x_t"],
        timestep=case["timestep"],
        past_key_values=cache(case["cache"]),
        loop_memory=model.loop_memory.repeat(batch, 1),
        memory_loop_repeat=case["depth"] + 1,
        memory_loop_start=1,
        memory_loop_end=3,
        cfg_text_scale=case["text_scale"],
        cfg_img_scale=case["img_scale"],
        cfg_renorm_min=0.5,
        cfg_renorm_type=case["renorm"],
    )
    for name, lens, ropes in [
        ("text", [0] * batch, [0] * batch),
        ("img", [3 + i for i in range(batch)], [5 + i for i in range(batch)]),
    ]:
        cfg = model.prepare_vae_latent_cfg(lens, ropes, sizes)
        inputs.update(
            {
                key.replace("cfg_", f"cfg_{name}_", 1): value
                for key, value in cfg.items()
            }
        )
        inputs[f"cfg_{name}_past_key_values"] = cache(case[f"{name}_cache"])
    with torch.no_grad():
        velocity = model._forward_flow_loop(**inputs)[0]
    outputs.append(velocity)
torch.save(outputs, sys.argv[3])
