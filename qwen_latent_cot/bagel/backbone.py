"""Load BAGEL-7B-MoT and apply the internal-loop LoRA policy."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

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
from .special_tokens import (
    BagelSpecialTokenIds,
    add_bagel_special_tokens,
    resolve_bagel_special_token_ids,
)
from .loop import (
    GENERATION_ATTENTION_PROJECTIONS,
    TEXT_ATTENTION_PROJECTIONS,
    inject_loop_lora,
    loop_trainable_names,
)


logger = logging.getLogger(__name__)


@dataclass
class BagelBackbone:
    """BAGEL model, tokenizer, image encoder, and native VAE bundle.

    Attributes:
        cfg: BAGEL checkpoint and loading configuration.
        raw_model: BAGEL `nn.Module` after weight load + vocab resize.
        tokenizer: BAGEL Qwen2 tokenizer with CoRT latent special tokens.
        token_ids: Resolved special-token id bundle.
        vit_image_size: Square image side length used by ViT (px).
        vit_patch_size: ViT patch size (14 for SigLIP-NaViT).
        vit_max_num_patch_per_side: Upper bound used for RoPE extrapolation.
    """

    cfg: Dict[str, object]
    raw_model: Optional[torch.nn.Module] = None
    bagel: Optional[torch.nn.Module] = field(default=None, repr=False)
    tokenizer: Optional[object] = None
    token_ids: Optional[BagelSpecialTokenIds] = None
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
           containing every parameter (CoRT's expected layout).
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
            p for p in model_path.glob("*.safetensors")
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
        disable_gen_expert = bool(self.cfg.get("disable_gen_expert", disable_visual_gen))
        llm_config.layer_module = (
            "Qwen2DecoderLayer"
            if disable_gen_expert
            else "Qwen2MoTDecoderLayer"
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
            num_loop_tokens=int(self.cfg.get("num_loop_tokens", 0) or 0),
            loop_depth=int(self.cfg.get("loop_depth", 1) or 1),
            loop_recycle_mode=str(self.cfg.get("loop_recycle_mode", "same_depth")),
            loop_memory_persist=bool(self.cfg.get("loop_memory_persist", True)),
            memory_loop_start_layer=int(self.cfg.get("memory_loop_start_layer", 20)),
            memory_loop_end_layer=int(self.cfg.get("memory_loop_end_layer", 28)),
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

        model.load_state_dict(state_dict, strict=False)

        add_bagel_special_tokens(tokenizer)
        resized_vocab = int(model.language_model.model.embed_tokens.weight.shape[0])
        desired_vocab = max(int(len(tokenizer)), resized_vocab)
        if desired_vocab != resized_vocab:
            model.language_model.resize_token_embeddings(desired_vocab)
        final_vocab = int(model.language_model.model.embed_tokens.weight.shape[0])
        model.config.llm_config.vocab_size = final_vocab
        model.language_model.config.vocab_size = final_vocab
        model.config.use_cache = False

        token_ids = resolve_bagel_special_token_ids(tokenizer)

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

        if disable_visual_gen:
            logger.info("Loaded BAGEL in understanding-only mode (visual_gen disabled).")
        if disable_gen_expert:
            logger.info("Loaded BAGEL language model without MoT generation experts.")

        self.vae_model = None
        if visual_gen and ae_path is not None:
            self.vae_model, _ = load_ae(str(ae_path))
            self.vae_model.eval()
            for parameter in self.vae_model.parameters():
                parameter.requires_grad = False

        return self

    def apply_loop_trainable_policy(
        self,
        *,
        start_layer: int,
        end_layer: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        include_text_kv: bool = False,
    ) -> List[str]:
        """Freeze BAGEL and open only loop-gated attention LoRA in the body."""

        assert self.bagel is not None
        model = self.bagel
        if not bool(getattr(model.config, "visual_gen", False)):
            raise RuntimeError("internal loop requires BAGEL visual generation")
        if not bool(getattr(model, "use_moe", False)):
            raise RuntimeError("internal loop requires BAGEL MoT generation experts")
        if int(rank) <= 0 or int(alpha) <= 0:
            raise ValueError("LoRA rank and alpha must be positive")

        for parameter in model.parameters():
            parameter.requires_grad = False
        inject_loop_lora(
            model,
            start_layer=int(start_layer),
            end_layer=int(end_layer),
            rank=int(rank),
            alpha=int(alpha),
            dropout=float(dropout),
            include_text_kv=bool(include_text_kv),
        )

        allowed_projections = GENERATION_ATTENTION_PROJECTIONS + (
            TEXT_ATTENTION_PROJECTIONS if include_text_kv else ()
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = bool(
                (".lora_A." in name or ".lora_B." in name)
                and any(
                    f".{projection}." in name
                    for projection in allowed_projections
                )
            )
        trainable = loop_trainable_names(
            model,
            start_layer=int(start_layer),
            end_layer=int(end_layer),
            include_text_kv=bool(include_text_kv),
        )
        count = sum(
            parameter.numel()
            for _, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        logger.info(
            "BAGEL loop policy: layers=[%d,%d) trainable=%.3fM tensors=%d "
            "targets=%s",
            int(start_layer),
            int(end_layer),
            count / 1e6,
            len(trainable),
            ",".join(allowed_projections),
        )
        return trainable
