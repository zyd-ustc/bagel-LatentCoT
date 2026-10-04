"""Load the original BAGEL weights and new anchored-loop modules."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .anchored_loop import LoopConfig, configure_stage1
from .modeling import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
    load_ae,
)
from .modeling.qwen2 import Qwen2Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class BagelBackbone:
    """BAGEL model, tokenizer, image encoder, and native VAE bundle.

    Attributes:
        cfg: BAGEL checkpoint and loading configuration.
        raw_model: BAGEL `nn.Module` after weight load + vocab resize.
        tokenizer: BAGEL Qwen2 tokenizer with its native special tokens.
        token_ids: Resolved special-token id bundle.
        vit_image_size: Square image side length used by ViT (px).
        vit_patch_size: ViT patch size (14 for SigLIP-NaViT).
        vit_max_num_patch_per_side: Upper bound used for RoPE extrapolation.
    """

    cfg: Dict[str, object]
    raw_model: Optional[torch.nn.Module] = None
    bagel: Optional[torch.nn.Module] = field(default=None, repr=False)
    tokenizer: Optional[object] = None
    token_ids: Optional[Dict[str, int]] = None
    vit_image_size: int = 0
    vit_patch_size: int = 0
    vit_max_num_patch_per_side: int = 70
    vae_model: Optional[torch.nn.Module] = field(default=None, repr=False)
    pretrained_time_embedder_state: Dict[str, torch.Tensor] = field(
        default_factory=dict, repr=False
    )

    @staticmethod
    def _load_bagel_state_dict(
        model_path: Path,
        *,
        ae_path: Optional[Path] = None,
    ) -> Dict[str, torch.Tensor]:
        """Load a BAGEL state dict regardless of layout.

        Supports two on-disk variants:

        1. **Combined**: a single ``model.safetensors`` (or ``ema.safetensors``)
           containing every parameter (native combined layout).
        2. **Sharded**: a directory of ``language_model_model_layers_*.safetensors``,
           ``language_model_lm_head.safetensors``, ``vit_model.safetensors``,
           etc., plus a ``model.safetensors`` that only carries the residual
           connector / vae<->llm bridge weights (the ModelScope mirror layout).

        For variant 2 we union every shard at the root of ``model_path``,
        skipping the VAE weights (already loaded via ``load_ae``).
        """
        from safetensors.torch import load_file

        ae_resolved = ae_path.resolve() if ae_path is not None else None

        combined_candidates = [
            model_path / "ema.safetensors",
        ]
        combined_path = next((p for p in combined_candidates if p.exists()), None)
        if combined_path is not None:
            return load_file(str(combined_path), device="cpu")

        shard_paths: List[Path] = sorted(
            p
            for p in model_path.glob("*.safetensors")
            if ae_resolved is None or p.resolve() != ae_resolved
        )
        if not shard_paths:
            raise FileNotFoundError(
                f"No BAGEL state dict under {model_path}: expected "
                "model.safetensors / ema.safetensors or sharded *.safetensors files."
            )

        merged: Dict[str, torch.Tensor] = {}
        for shard in shard_paths:
            chunk = load_file(str(shard), device="cpu")
            duplicates = merged.keys() & chunk.keys()
            if duplicates:
                logger.warning(
                    "Skipping %d duplicate keys from %s (already loaded earlier).",
                    len(duplicates),
                    shard.name,
                )
                for k in duplicates:
                    chunk.pop(k, None)
            merged.update(chunk)
        return merged

    def load(self) -> "BagelBackbone":
        model_path = Path(str(self.cfg["model_path"]))
        if not model_path.exists():
            raise FileNotFoundError(f"BAGEL model_path does not exist: {model_path}")

        llm_config = Qwen2Config.from_json_file(model_path / "llm_config.json")
        # BAGEL checkpoints were authored with Transformers 4.x, whose
        # PreTrainedConfig materialized an absent pad_token_id as None.
        # Transformers 5.x may leave the attribute absent altogether, while
        # the bundled NaViT Qwen2 implementation still reads it directly.
        if not hasattr(llm_config, "pad_token_id"):
            llm_config.pad_token_id = None
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False

        vit_config = SiglipVisionConfig.from_json_file(model_path / "vit_config.json")
        vit_config.rope = False
        if int(vit_config.num_hidden_layers) > 1:
            vit_config.num_hidden_layers = int(vit_config.num_hidden_layers) - 1

        disable_visual_gen = bool(self.cfg.get("disable_visual_gen", True))
        visual_gen = not disable_visual_gen
        disable_gen_expert = bool(
            self.cfg.get("disable_gen_expert", disable_visual_gen)
        )
        llm_config.layer_module = (
            "Qwen2DecoderLayer" if disable_gen_expert else "Qwen2MoTDecoderLayer"
        )

        ae_candidates = [
            model_path / "ae.safetensors",
            model_path / "vae" / "ae.safetensors",
            model_path / "vae" / "diffusion_pytorch_model.safetensors",
        ]
        ae_path = next((p for p in ae_candidates if p.exists()), None)
        if visual_gen and ae_path is None:
            raise FileNotFoundError(
                f"No VAE weights under {model_path}. Expected one of: "
                + ", ".join(str(c) for c in ae_candidates)
            )
        vae_config = None
        if visual_gen:
            _vae, vae_config = load_ae(str(ae_path))
            del _vae

        bagel_config = BagelConfig(
            visual_gen=visual_gen,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
            timestep_shift=float(self.cfg.get("timestep_shift", 1.0)),
            t2i_loop=LoopConfig(**dict(self.cfg.get("t2i_loop", {}))).to_dict(),
        )

        llm = Qwen2ForCausalLM(llm_config)
        vit = SiglipVisionModel(vit_config)
        model = Bagel(llm, vit, bagel_config)
        if hasattr(model.vit_model.vision_model.embeddings, "convert_conv2d_to_linear"):
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                vit_config,
                meta=False,
            )

        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        state_dict = self._load_bagel_state_dict(model_path, ae_path=ae_path)
        # Retained for checkpoint provenance diagnostics.
        self.pretrained_time_embedder_state = {
            key.removeprefix("time_embedder."): value.detach().clone()
            for key, value in state_dict.items()
            if key.startswith("time_embedder.")
        }

        ckpt_embed = state_dict.get("language_model.model.embed_tokens.weight")
        if ckpt_embed is not None:
            ckpt_vocab = int(ckpt_embed.shape[0])
            cur_vocab = int(model.language_model.model.embed_tokens.weight.shape[0])
            if ckpt_vocab != cur_vocab:
                model.language_model.resize_token_embeddings(ckpt_vocab)

        # Every native tensor is required. Only new loop modules are allowed
        # to be absent from the original checkpoint.
        native = {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith("t2i_loop.")
        }
        missing = sorted(set(native) - set(state_dict))
        mismatched = sorted(
            key
            for key in native.keys() & state_dict.keys()
            if native[key].shape != state_dict[key].shape
        )
        if missing or mismatched:
            raise RuntimeError(
                f"Incomplete native BAGEL checkpoint: missing={missing[:8]} mismatched={mismatched[:8]}"
            )
        model.load_state_dict({key: state_dict[key] for key in native}, strict=False)
        # Preserve the native vocabulary. No CoRT or reasoning schema tokens.
        token_ids = {}
        for name, text in {
            "bos_token_id": "<|im_start|>",
            "eos_token_id": "<|im_end|>",
            "start_of_image": "<|vision_start|>",
            "end_of_image": "<|vision_end|>",
        }.items():
            ids = tokenizer.encode(text, add_special_tokens=False)
            if (
                len(ids) != 1
                or ids[0] >= model.language_model.model.embed_tokens.num_embeddings
            ):
                raise RuntimeError(f"Missing native BAGEL special token {text}")
            token_ids[name] = int(ids[0])
        model.config.use_cache = False

        num_image_tokens = int(self.cfg.get("num_image_tokens", 4900))  # 70x70 default
        per_side = int(math.isqrt(num_image_tokens))
        assert per_side * per_side == num_image_tokens, (
            f"num_image_tokens must be a perfect square, got {num_image_tokens}"
        )
        vit_patch_size = int(model.config.vit_config.patch_size)
        vit_image_size = per_side * vit_patch_size

        bagel = model.to(torch.bfloat16)
        self.bagel = bagel
        self.raw_model = bagel
        self.tokenizer = tokenizer
        self.token_ids = token_ids
        self.vit_image_size = vit_image_size
        self.vit_patch_size = vit_patch_size
        self.vit_max_num_patch_per_side = int(model.config.vit_max_num_patch_per_side)
        bagel.t2i_loop.initialize_memory_from_boundaries(
            bagel.language_model.model.embed_tokens.weight,
            [token_ids["start_of_image"], token_ids["end_of_image"]],
            seed=int(self.cfg.get("memory_init_seed", 0)),
        )
        bagel.t2i_loop.float()

        if disable_visual_gen:
            logger.info(
                "Loaded BAGEL in understanding-only mode (visual_gen disabled)."
            )
        if disable_gen_expert:
            logger.info("Loaded BAGEL language model without MoT generation experts.")

        self.vae_model = None
        if visual_gen and ae_path is not None:
            self.vae_model, _ = load_ae(str(ae_path))
            self.vae_model.eval()
            for parameter in self.vae_model.parameters():
                parameter.requires_grad = False

        return self

    def apply_stage1_policy(self) -> List[str]:
        if self.bagel is None:
            raise RuntimeError("load the backbone first")
        return configure_stage1(self.bagel)
