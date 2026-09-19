# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import List, Tuple, Optional, Dict, Any, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

try:
    from data.data_utils import (
        create_sparse_mask,
        get_flattened_position_ids_extrapolate,
        get_flattened_position_ids_interpolate,
        patchify,
    )
except ImportError:
    try:
        # Fallback: CoRT's inlined copy of these utilities
        from src.infer.misc import (
            create_sparse_mask,
            get_flattened_position_ids_extrapolate,
            get_flattened_position_ids_interpolate,
            patchify,
        )
    except ImportError:
        try:
            # Qwen-LatentCoT inlined copy (self-contained import path)
            from qwen_latent_cot.bagel.modeling._bagel_utils import (
                create_sparse_mask,
                get_flattened_position_ids_extrapolate,
                get_flattened_position_ids_interpolate,
                patchify,
            )
        except ImportError:
            # Last resort: these are only needed at runtime, not import time
            create_sparse_mask = None
            get_flattened_position_ids_extrapolate = None
            get_flattened_position_ids_interpolate = None
            patchify = None
from .qwen2_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from ..cache_utils.taylorseer import cache_init
from ...flow_grpo import sde_step_with_logprob

from tqdm import tqdm


class BagelConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        num_loop_tokens=8,
        loop_depth=2,
        loop_uncond_memory="m0",
        loop_recycle_mode="same_depth",
        loop_memory_persist=False,
        memory_loop_start_layer=16,
        memory_loop_end_layer=24,
        round0_gen_reads_memory=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift
        self.num_loop_tokens = int(num_loop_tokens)
        self.loop_depth = int(loop_depth)
        self.loop_uncond_memory = str(loop_uncond_memory)
        self.loop_recycle_mode = str(loop_recycle_mode)
        self.loop_memory_persist = bool(loop_memory_persist)
        self.memory_loop_start_layer = int(memory_loop_start_layer)
        self.memory_loop_end_layer = int(memory_loop_end_layer)
        self.round0_gen_reads_memory = bool(round0_gen_reads_memory)


class Bagel(PreTrainedModel):
    config_class = BagelConfig
    base_model_prefix = "bagel"

    def __init__(self, language_model, vit_model, config: BagelConfig):
        super().__init__(config)
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = (
                config.vae_config.downsample * config.latent_patch_size
            )
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size**2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(
                self.max_latent_size, self.hidden_size
            )

        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = config.vit_config.hidden_size
            self.connector = MLPconnector(
                self.vit_hidden_size, self.hidden_size, config.connector_act
            )
            self.vit_pos_embed = PositionEmbedding(
                self.vit_max_num_patch_per_side, self.hidden_size
            )

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        num_loop_tokens = int(getattr(config, "num_loop_tokens", 8) or 0)
        loop_depth = int(getattr(config, "loop_depth", 2) or 1)
        if num_loop_tokens < 0:
            raise ValueError("num_loop_tokens must be >= 0")
        if loop_depth < 1:
            raise ValueError("loop_depth must be >= 1")
        self.num_loop_tokens = num_loop_tokens
        self.loop_depth = loop_depth
        self.loop_uncond_memory = str(getattr(config, "loop_uncond_memory", "m0"))
        if self.loop_uncond_memory not in ("m0", "zero"):
            raise ValueError("loop_uncond_memory must be 'm0' or 'zero'")
        self.loop_recycle_mode = str(
            getattr(config, "loop_recycle_mode", "same_depth")
        )
        if self.loop_recycle_mode not in ("same_depth", "full_depth"):
            raise ValueError("loop_recycle_mode must be 'same_depth' or 'full_depth'")
        self.loop_memory_persist = bool(
            getattr(config, "loop_memory_persist", False)
        )
        self.memory_loop_start_layer = int(
            getattr(config, "memory_loop_start_layer", 16)
        )
        self.memory_loop_end_layer = int(getattr(config, "memory_loop_end_layer", 24))
        self.round0_gen_reads_memory = bool(
            getattr(config, "round0_gen_reads_memory", False)
        )
        if num_loop_tokens > 0:
            self.loop_memory = nn.Parameter(
                torch.zeros(num_loop_tokens, self.hidden_size)
            )
            self.loop_memory.requires_grad_(False)
        else:
            self.register_parameter("loop_memory", None)
        self.last_loop_diagnostics: List[Dict[str, Any]] = []
        self._init_weights()

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def init_loop_memory_from_boundary_embeddings(
        self, token_ids: Optional[List[int]] = None
    ) -> None:
        """Copy native token embeddings into m0. Phase-0: no new projector."""

        if self.loop_memory is None:
            return
        embed = self.language_model.model.embed_tokens
        if token_ids:
            ids = torch.tensor(token_ids, device=embed.weight.device, dtype=torch.long)
        else:
            ids = torch.arange(
                min(2, int(embed.weight.shape[0])),
                device=embed.weight.device,
                dtype=torch.long,
            )
        with torch.no_grad():
            base = embed(ids).float().mean(dim=0)
            noise = 1e-4 * torch.randn_like(self.loop_memory)
            self.loop_memory.copy_(
                base.to(device=self.loop_memory.device, dtype=self.loop_memory.dtype)
                .unsqueeze(0)
                .expand_as(self.loop_memory)
                + noise
            )

    @staticmethod
    def mot_und_route_indexes(
        packed_text_indexes: torch.LongTensor,
        packed_loop_token_indexes: Optional[torch.LongTensor],
    ) -> torch.LongTensor:
        if packed_loop_token_indexes is None or int(packed_loop_token_indexes.numel()) == 0:
            return packed_text_indexes
        return torch.cat([packed_text_indexes, packed_loop_token_indexes], dim=0)

    @staticmethod
    def memory_slot_stats(memory: torch.Tensor) -> Dict[str, float]:
        """Pairwise cosine / effective rank of K memory slots. Detects collapse."""

        slots = memory.detach().float()
        if slots.ndim != 2:
            slots = slots.reshape(-1, slots.shape[-1])
        k = int(slots.shape[0])
        if k == 0:
            return {
                "mean_abs_pairwise_cosine": float("nan"),
                "pairwise_cosine": float("nan"),
                "effective_rank": 0.0,
                "sigma1_ratio": float("nan"),
            }
        if k == 1:
            return {
                "mean_abs_pairwise_cosine": 1.0,
                "pairwise_cosine": 1.0,
                "effective_rank": 1.0,
                "sigma1_ratio": 1.0,
            }
        normed = torch.nn.functional.normalize(slots, dim=-1)
        gram = normed @ normed.T
        off = gram.fill_diagonal_(0)
        denom = float(k * (k - 1))
        pairwise = float(off.abs().sum() / max(denom, 1.0))
        centered = slots - slots.mean(dim=0, keepdim=True)
        singular = torch.linalg.svdvals(centered)
        energy = singular.clamp_min(0)
        total = float(energy.sum().clamp_min(1e-12))
        share = energy / total
        effective_rank = float(torch.exp(-(share * (share + 1e-12).log()).sum()))
        return {
            "mean_abs_pairwise_cosine": pairwise,
            "pairwise_cosine": pairwise,
            "effective_rank": effective_rank,
            "sigma1_ratio": float(energy[0] / total),
        }

    @staticmethod
    def relative_l2(current: torch.Tensor, reference: torch.Tensor) -> float:
        ref = torch.linalg.vector_norm(reference.detach().float()).clamp_min(1e-12)
        return float(
            torch.linalg.vector_norm((current - reference).detach().float()) / ref
        )

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.
        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(
            size=(sequence_length, self.hidden_size)
        )
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(
                sample_lens, split_lens, attn_modes, packed_text_embedding.device
            )
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask,
                B=1,
                H=self.num_heads,
                Q_LEN=seqlen,
                KV_LEN=seqlen,
                device=packed_text_embedding.device,
                BLOCK_SIZE=128,
                _compile=True,
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        if self.config.visual_und:
            cu_seqlens = torch.nn.functional.pad(
                torch.cumsum(vit_token_seqlens, dim=0), (1, 0)
            )
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model(
                packed_pixel_values=packed_vit_tokens,
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        if self.config.visual_gen:
            p = self.latent_patch_size
            packed_latent = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, : h * p, : w * p].reshape(
                    self.latent_channel, h, p, w, p
                )
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(
                    -1, p * p * self.latent_channel
                )
                packed_latent.append(latent)
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps)
            packed_timesteps = (
                self.timestep_shift
                * packed_timesteps
                / (1 + (self.timestep_shift - 1) * packed_timesteps)
            )
            packed_latent = (
                1 - packed_timesteps[:, None]
            ) * packed_latent_clean + packed_timesteps[:, None] * noise
            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = (
                self.vae2llm(packed_latent)
                + packed_timestep_embeds
                + latent_token_pos_emb
            )
            packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes = torch.cat(
                    [packed_text_indexes, packed_vit_token_indexes], dim=0
                )
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )

        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        mse = None
        if self.config.visual_gen:
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
            target = (
                noise - packed_latent_clean
            )  # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
            has_mse = packed_timesteps > 0
            mse = (packed_mse_preds - target[has_mse]) ** 2

        ce = None
        if ce_loss_indexes is not None:
            packed_ce_preds = self.language_model.lm_head(
                last_hidden_state[ce_loss_indexes]
            )
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        return dict(mse=mse, ce=ce)

    def prepare_prompts(
        self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids
    ):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(
            prompts, curr_kvlens, curr_rope
        ):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            # BAGEL adds its own image-generation BOS/EOS immediately below.
            # Asking the tokenizer to add model-level special tokens here is
            # both redundant and incompatible with Transformers 5.x when the
            # checkpoint intentionally has ``bos_token=None``: the slow Qwen2
            # tokenizer then propagates a None token id into ``pad()``.
            text_ids = tokenizer.encode(prompt, add_special_tokens=False)
            text_ids = (
                [new_token_ids["bos_token_id"]]
                + text_ids
                + [new_token_ids["eos_token_id"]]
            )
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(
                range(curr_position_id, curr_position_id + len(text_ids))
            )
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(
                packed_text_position_ids, dtype=torch.long
            ),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(
                packed_key_value_indexes, dtype=torch.long
            ),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(
        self, curr_kvlens, curr_rope, images, transforms, new_token_ids
    ):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = (
            list(),
            list(),
            list(),
        )
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1),
                image_tensor.size(2),
                self.vit_patch_size,
                max_num_patches_per_side=self.vit_max_num_patch_per_side,
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(
                packed_vit_token_indexes, dtype=torch.long
            ),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(
                packed_key_value_indexes, dtype=torch.long
            ),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    def prepare_vit_features(
        self,
        curr_kvlens,
        curr_rope,
        vit_features,
        vit_position_ids,
        new_token_ids,
    ):
        """Prepare connector-before ViT features for native KV-cache prefill.

        This is the feature-space counterpart of :meth:`prepare_vit_images`.
        It preserves BAGEL's exact ``BOI + patches + EOI`` sequence, outer
        language/RoPE positions, and cache indexes while skipping only the
        already-computed frozen ViT forward.  Each feature tensor is ``[K,D]``
        before ``self.connector``; it is not an LLM hidden-state cache.
        """

        if not (
            len(curr_kvlens)
            == len(curr_rope)
            == len(vit_features)
            == len(vit_position_ids)
        ):
            raise ValueError(
                "curr_kvlens, curr_rope, vit_features and vit_position_ids "
                "must have the same batch length"
            )

        packed_vit_feature_indexes = []
        packed_vit_features = []
        packed_vit_position_ids = []
        packed_text_ids, packed_text_indexes = [], []
        packed_seqlens, packed_position_ids, packed_indexes = [], [], []
        packed_key_value_indexes = []

        local_cursor = cache_cursor = 0
        newlens, new_rope = [], []
        for feature, position_ids, curr_kvlen, curr_position_id in zip(
            vit_features, vit_position_ids, curr_kvlens, curr_rope
        ):
            if feature.ndim != 2:
                raise ValueError(
                    f"connector-before ViT features must be [K,D], got {tuple(feature.shape)}"
                )
            position_ids = position_ids.reshape(-1)
            num_img_tokens = int(feature.shape[0])
            if int(position_ids.numel()) != num_img_tokens:
                raise ValueError(
                    "ViT feature/position row mismatch: "
                    f"{num_img_tokens} != {int(position_ids.numel())}"
                )

            packed_key_value_indexes.extend(
                range(cache_cursor, cache_cursor + int(curr_kvlen))
            )
            cache_cursor += int(curr_kvlen)

            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(local_cursor)
            packed_indexes.append(cache_cursor)
            cache_cursor += 1
            local_cursor += 1

            packed_vit_features.append(feature)
            packed_vit_position_ids.append(position_ids)
            packed_vit_feature_indexes.extend(
                range(local_cursor, local_cursor + num_img_tokens)
            )
            packed_indexes.extend(range(cache_cursor, cache_cursor + num_img_tokens))
            cache_cursor += num_img_tokens
            local_cursor += num_img_tokens

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(local_cursor)
            packed_indexes.append(cache_cursor)
            cache_cursor += 1
            local_cursor += 1

            packed_position_ids.extend([int(curr_position_id)] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(int(curr_kvlen) + num_img_tokens + 2)
            new_rope.append(int(curr_position_id) + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_vit_features": torch.cat(packed_vit_features, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_feature_indexes": torch.tensor(
                packed_vit_feature_indexes, dtype=torch.long
            ),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(
                packed_key_value_indexes, dtype=torch.long
            ),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }
        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit_features(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_features: torch.Tensor,
        packed_vit_feature_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_vit_position_hidden: Optional[torch.Tensor] = None,
    ):
        """Write cached connector-before ViT rows through native UND KV prefill."""

        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(
            (sum(packed_seqlens), self.hidden_size)
        )
        packed_sequence[packed_text_indexes] = packed_text_embedding

        connector_dtype = next(self.connector.parameters()).dtype
        packed_vit_token_embed = self.connector(
            packed_vit_features.to(dtype=connector_dtype)
        )
        if packed_vit_position_hidden is None:
            position_hidden = self.vit_pos_embed(packed_vit_position_ids)
        else:
            if tuple(packed_vit_position_hidden.shape) != tuple(
                packed_vit_token_embed.shape
            ):
                raise ValueError(
                    "explicit ViT position hidden shape mismatch: "
                    f"{tuple(packed_vit_position_hidden.shape)} != "
                    f"{tuple(packed_vit_token_embed.shape)}"
                )
            position_hidden = packed_vit_position_hidden.to(
                device=packed_vit_token_embed.device,
                dtype=packed_vit_token_embed.dtype,
            )
        packed_vit_token_embed = packed_vit_token_embed + position_hidden
        packed_sequence[packed_vit_feature_indexes] = packed_vit_token_embed.to(
            packed_sequence.dtype
        )

        extra_inputs = {"mode": "und"} if self.use_moe else {}
        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        return output.past_key_values

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(
            (sum(packed_seqlens), self.hidden_size)
        )
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(vit_token_seqlens, dim=0), (1, 0)
        )
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens,
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_images(
        self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0
    ):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1),
                image_tensor.size(2),
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size,
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, : image_tensor.shape[1], : image_tensor.shape[2]] = (
                image_tensor
            )

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(
                packed_vae_token_indexes, dtype=torch.long
            ),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(
                packed_key_value_indexes, dtype=torch.long
            ),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(
            (sum(packed_seqlens), self.hidden_size)
        )
        packed_sequence[packed_text_indexes] = packed_text_embedding

        padded_latent = vae_model.encode(padded_images)

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, : h * p, : w * p].reshape(
                self.latent_channel, h, p, w, p
            )
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(
                -1, p * p * self.latent_channel
            )
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = (
            self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        )
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes,
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_latent_condition(
        self,
        curr_kvlens,
        curr_rope,
        packed_latents,
        image_shapes,
        new_token_ids,
        timestep=0.0,
    ):
        """Layout for a raw patchified VAE latent condition (no image decode/encode).

        ``packed_latents`` is a list of ``[num_image_tokens, latent_channel*p^2]``
        tensors in BAGEL's own patchified latent space (e.g. an ``x0_hat`` draft
        estimate). Mirrors ``prepare_vae_images`` minus the VAE encode.
        """

        packed_vae_position_ids = list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()
        packed_latents_out = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for latent, image_shape, curr_kvlen, curr_position_id in zip(
            packed_latents, image_shapes, curr_kvlens, curr_rope
        ):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            H, W = image_shape
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            num_img_tokens = h * w
            if int(latent.shape[0]) != num_img_tokens:
                raise ValueError(
                    "latent condition token count mismatch: "
                    f"{int(latent.shape[0])} != {num_img_tokens} for {tuple(image_shape)}"
                )
            vae_position_ids = self.get_flattened_position_ids(
                H,
                W,
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size,
            )
            packed_vae_position_ids.append(vae_position_ids)
            packed_latents_out.append(latent.reshape(-1, latent.shape[-1]))
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_latents": torch.cat(packed_latents_out, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([float(timestep)]),
            "packed_vae_token_indexes": torch.tensor(
                packed_vae_token_indexes, dtype=torch.long
            ),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(
                packed_key_value_indexes, dtype=torch.long
            ),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vae_latent(
        self,
        past_key_values: NaiveCache,
        packed_latents: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        """``forward_cache_update_vae`` for a pre-encoded patchified latent."""

        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(
            (sum(packed_seqlens), self.hidden_size)
        )
        packed_sequence[packed_text_indexes] = packed_text_embedding

        packed_latents = packed_latents.to(self.vae2llm.weight.dtype)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = (
            self.vae2llm(packed_latents) + packed_timestep_embeds + packed_pos_embed
        )
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes,
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        return output.past_key_values

    def prepare_vae_latent(
        self,
        curr_kvlens,
        curr_rope,
        image_sizes,
        new_token_ids,
        num_loop_tokens=None,
    ):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = (
            list(),
            list(),
            list(),
        )
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_vae_seqlens = list()
        packed_key_value_indexes = list()
        packed_boundary_token_indexes = list()
        packed_loop_token_indexes: list = []
        if num_loop_tokens is None:
            num_loop_tokens = int(getattr(self.config, "num_loop_tokens", 0) or 0)
        num_loop_tokens = int(num_loop_tokens)
        if num_loop_tokens < 0:
            raise ValueError("num_loop_tokens must be >= 0")

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(
            image_sizes, curr_kvlens, curr_rope
        ):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen
            image_position_id = curr_position_id

            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(query_curr)
            packed_boundary_token_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            if num_loop_tokens:
                packed_loop_token_indexes.extend(
                    range(query_curr, query_curr + num_loop_tokens)
                )
                packed_indexes.extend(range(curr, curr + num_loop_tokens))
                curr += num_loop_tokens
                query_curr += num_loop_tokens

            vae_posiiton_ids = self.get_flattened_position_ids(
                H,
                W,
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size,
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_init_noises.append(
                torch.randn(
                    num_image_tokens, self.latent_channel * self.latent_patch_size**2
                )
            )
            packed_vae_token_indexes.extend(
                range(query_curr, query_curr + num_image_tokens)
            )
            packed_vae_seqlens.append(num_image_tokens)
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(query_curr)
            packed_boundary_token_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend(
                [image_position_id] * (num_image_tokens + 2 + num_loop_tokens)
            )
            packed_seqlens.append(num_image_tokens + 2 + num_loop_tokens)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(
                packed_vae_token_indexes, dtype=torch.long
            ),
            "packed_vae_seqlens": torch.tensor(packed_vae_seqlens, dtype=torch.int),
            "packed_boundary_token_indexes": torch.tensor(
                packed_boundary_token_indexes, dtype=torch.long
            ),
            "packed_loop_token_indexes": torch.tensor(
                packed_loop_token_indexes, dtype=torch.long
            ),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(
                packed_key_value_indexes, dtype=torch.long
            ),
        }

        return generation_input

    def prepare_vae_latent_cfg(
        self,
        curr_kvlens,
        curr_rope,
        image_sizes,
        num_loop_tokens=None,
    ):
        packed_position_ids, packed_indexes, packed_key_value_indexes = (
            list(),
            list(),
            list(),
        )

        if num_loop_tokens is None:
            num_loop_tokens = int(getattr(self.config, "num_loop_tokens", 0) or 0)
        num_loop_tokens = int(num_loop_tokens)
        if num_loop_tokens < 0:
            raise ValueError("num_loop_tokens must be >= 0")

        curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(
            image_sizes, curr_kvlens, curr_rope
        ):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen
            image_position_id = curr_position_id

            packed_indexes.append(curr)
            curr += 1

            if num_loop_tokens:
                packed_indexes.extend(range(curr, curr + num_loop_tokens))
                curr += num_loop_tokens

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1

            packed_position_ids.extend(
                [image_position_id] * (num_image_tokens + 2 + num_loop_tokens)
            )

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(
                packed_position_ids, dtype=torch.long
            ),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(
                packed_key_value_indexes, dtype=torch.long
            ),
        }

        return generation_input

    @staticmethod
    def prepare_image_schedule(
        num_timesteps: int,
        timestep_shift: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return BAGEL's native shifted model times and Euler step widths."""

        if int(num_timesteps) < 2:
            raise ValueError("num_timesteps must be at least 2")
        if float(timestep_shift) <= 0.0:
            raise ValueError("timestep_shift must be positive")
        schedule = torch.linspace(1, 0, int(num_timesteps), device=device)
        schedule = (
            float(timestep_shift)
            * schedule
            / (1 + (float(timestep_shift) - 1) * schedule)
        )
        return schedule[:-1], schedule[:-1] - schedule[1:]

    @staticmethod
    def image_euler_step(
        x_t: torch.Tensor,
        velocity: torch.Tensor,
        dt: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Advance one native reverse-flow step without changing its numerics."""

        return x_t - velocity.to(x_t.device) * dt

    def predict_image_velocity(self, **kwargs) -> torch.Tensor:
        """Public pause/resume boundary around BAGEL's unchanged flow forward."""

        return self._forward_flow(**kwargs)

    @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_seqlens: torch.IntTensor,
        packed_boundary_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        sde_step_indices: Optional[Tuple[int, ...]] = None,
        sde_noise_level: float = 0.0,
        sde_seed: int = 0,
        return_trajectory: bool = False,
        packed_loop_token_indexes: Optional[torch.LongTensor] = None,
        loop_depth: Optional[int] = None,
        loop_uncond_memory: Optional[str] = None,
        loop_recycle_mode: Optional[str] = None,
        loop_memory_persist: Optional[bool] = None,
        memory_loop_start: Optional[int] = None,
        memory_loop_end: Optional[int] = None,
        round0_gen_reads_memory: Optional[bool] = None,
        return_loop_diagnostics: bool = False,
        enable_taylorseer=False,
    ):
        if enable_taylorseer:
            self.language_model.model.enable_taylorseer = True
            model_pred_cache_dic, model_pred_current = cache_init(self, num_timesteps)
            model_pred_text_cache_dic, model_pred_text_current = cache_init(
                self, num_timesteps
            )
            model_pred_img_cache_dic, model_pred_img_current = cache_init(
                self, num_timesteps
            )
        else:
            self.language_model.model.enable_taylorseer = False
            model_pred_cache_dic, model_pred_current = None, None
            model_pred_text_cache_dic, model_pred_text_current = None, None
            model_pred_img_cache_dic, model_pred_img_current = None, None

        x_t = packed_init_noises

        timesteps, dts = self.prepare_image_schedule(
            num_timesteps, timestep_shift, x_t.device
        )
        selected_sde_steps = {int(index) for index in (sde_step_indices or ())}
        invalid_sde_steps = sorted(
            index for index in selected_sde_steps if not 0 <= index < len(timesteps)
        )
        if invalid_sde_steps:
            raise ValueError(
                f"SDE step indexes must be in [0, {len(timesteps) - 1}], "
                f"got {invalid_sde_steps}"
            )
        trajectory = []
        if packed_loop_token_indexes is None:
            packed_loop_token_indexes = packed_text_ids.new_empty(
                (0,), dtype=torch.long
            )
        memory_loop_enabled = int(packed_loop_token_indexes.numel()) > 0
        inner_depth = int(
            loop_depth
            if loop_depth is not None
            else getattr(self.config, "loop_depth", 2) or 1
        )
        if inner_depth < 1:
            raise ValueError("loop_depth must be >= 1")
        uncond_mode = str(
            loop_uncond_memory
            or getattr(self.config, "loop_uncond_memory", "m0")
        )
        if uncond_mode not in ("m0", "zero"):
            raise ValueError("loop_uncond_memory must be 'm0' or 'zero'")
        if memory_loop_enabled and enable_taylorseer:
            raise ValueError("TaylorSeer is disabled for the MoT hidden memory loop")
        if memory_loop_enabled and self.loop_memory is None:
            raise ValueError("packed_loop_token_indexes is set but loop_memory is None")
        self.last_loop_diagnostics = []
        recycle_mode = str(
            loop_recycle_mode
            if loop_recycle_mode is not None
            else getattr(self.config, "loop_recycle_mode", "same_depth")
        )
        if recycle_mode not in ("same_depth", "full_depth"):
            raise ValueError("loop_recycle_mode must be 'same_depth' or 'full_depth'")
        persist_memory = (
            bool(loop_memory_persist)
            if loop_memory_persist is not None
            else bool(getattr(self.config, "loop_memory_persist", False))
        )
        round0_write = (
            bool(round0_gen_reads_memory)
            if round0_gen_reads_memory is not None
            else bool(getattr(self.config, "round0_gen_reads_memory", False))
        )
        body_start = (
            memory_loop_start
            if memory_loop_start is not None
            else int(getattr(self.config, "memory_loop_start_layer", 16))
        )
        body_end = (
            memory_loop_end
            if memory_loop_end is not None
            else int(getattr(self.config, "memory_loop_end_layer", 24))
        )
        embed_memory = None
        memory_full = None
        memory_text = None
        memory_img = None
        if memory_loop_enabled:
            n_samples = int(packed_vae_seqlens.numel())
            k_total = int(packed_loop_token_indexes.numel())
            if n_samples < 1 or k_total % n_samples != 0:
                raise ValueError(
                    "packed_loop_token_indexes must tile equally across samples"
                )
            embed_memory = self.loop_memory.to(device=x_t.device).repeat(n_samples, 1)
            memory_full = None
            memory_text = None
            memory_img = None

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            if memory_loop_enabled:
                m_in = None if memory_full is None else memory_full.detach().clone()
                m_in_text = (
                    None if memory_text is None else memory_text.detach().clone()
                )
                m_in_img = None if memory_img is None else memory_img.detach().clone()
                rounds = 1 if recycle_mode == "same_depth" else inner_depth
                prev_memory = None
                v_t = None
                step_velocities = []
                for inner_r in range(rounds):
                    v_t, memory_full, memory_text, memory_img, diag = (
                        self._forward_flow_loop(
                            x_t=x_t,
                            timestep=timestep,
                            packed_vae_token_indexes=packed_vae_token_indexes,
                            packed_vae_position_ids=packed_vae_position_ids,
                            packed_text_ids=packed_text_ids,
                            packed_text_indexes=packed_text_indexes,
                            packed_position_ids=packed_position_ids,
                            packed_indexes=packed_indexes,
                            packed_seqlens=packed_seqlens,
                            key_values_lens=key_values_lens,
                            past_key_values=past_key_values,
                            packed_key_value_indexes=packed_key_value_indexes,
                            packed_loop_token_indexes=packed_loop_token_indexes,
                            loop_memory=(
                                memory_full
                                if recycle_mode == "full_depth"
                                and memory_full is not None
                                else embed_memory
                            ),
                            loop_memory_text=(
                                memory_text
                                if recycle_mode == "full_depth"
                                and memory_text is not None
                                else (
                                    torch.zeros_like(embed_memory)
                                    if uncond_mode == "zero"
                                    else embed_memory
                                )
                            ),
                            loop_memory_img=(
                                memory_img
                                if recycle_mode == "full_depth"
                                and memory_img is not None
                                else (
                                    torch.zeros_like(embed_memory)
                                    if uncond_mode == "zero"
                                    else embed_memory
                                )
                            ),
                            packed_boundary_token_indexes=packed_boundary_token_indexes,
                            cfg_renorm_min=cfg_renorm_min,
                            cfg_renorm_type=cfg_renorm_type,
                            cfg_text_scale=cfg_text_scale_,
                            cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                            cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                            cfg_text_key_values_lens=cfg_text_key_values_lens,
                            cfg_text_past_key_values=cfg_text_past_key_values,
                            cfg_text_packed_key_value_indexes=(
                                cfg_text_packed_key_value_indexes
                            ),
                            cfg_img_scale=cfg_img_scale_,
                            cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                            cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                            cfg_img_key_values_lens=cfg_img_key_values_lens,
                            cfg_img_past_key_values=cfg_img_past_key_values,
                            cfg_img_packed_key_value_indexes=(
                                cfg_img_packed_key_value_indexes
                            ),
                            cfg_type=cfg_type,
                            recycle_mode=recycle_mode,
                            memory_loop_repeat=inner_depth,
                            memory_loop_start=body_start,
                            memory_loop_end=body_end,
                            memory_body_in=memory_full
                            if recycle_mode == "same_depth"
                            else None,
                            memory_body_in_text=memory_text
                            if recycle_mode == "same_depth"
                            else None,
                            memory_body_in_img=memory_img
                            if recycle_mode == "same_depth"
                            else None,
                            embed_memory=embed_memory,
                            round0_gen_reads_memory=(
                                True
                                if recycle_mode == "full_depth" and inner_r > 0
                                else round0_write
                            ),
                        )
                    )
                    if prev_memory is None:
                        cosine = float("nan")
                    else:
                        cosine = float(
                            F.cosine_similarity(
                                memory_full.flatten().float(),
                                prev_memory.flatten().float(),
                                dim=0,
                            )
                        )
                    prev_memory = memory_full.detach()
                    step_velocities.append(v_t.detach())
                    if recycle_mode == "full_depth" and len(step_velocities) > 1:
                        diag = dict(diag)
                        diag["delta_v"] = [
                            self.relative_l2(step_velocities[-1], step_velocities[0])
                        ]
                    self.last_loop_diagnostics.append(
                        {
                            "step": int(i),
                            "t": float(t),
                            "r": int(inner_r),
                            "recycle_mode": recycle_mode,
                            "persist": persist_memory,
                            "memory_cosine_to_prev": cosine,
                            **diag,
                        }
                    )
                m_out = (
                    None if memory_full is None else memory_full.detach().clone()
                )
                if not persist_memory:
                    memory_full = None
                    memory_text = None
                    memory_img = None
            else:
                m_in = None
                m_in_text = None
                m_in_img = None
                m_out = None
                result = self.predict_image_velocity(
                    x_t=x_t,
                    timestep=timestep,
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_vae_position_ids=packed_vae_position_ids,
                    packed_text_ids=packed_text_ids,
                    packed_text_indexes=packed_text_indexes,
                    packed_position_ids=packed_position_ids,
                    packed_indexes=packed_indexes,
                    packed_seqlens=packed_seqlens,
                    key_values_lens=key_values_lens,
                    past_key_values=past_key_values,
                    packed_key_value_indexes=packed_key_value_indexes,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    # cfg_text
                    cfg_text_scale=cfg_text_scale_,
                    cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                    cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                    cfg_text_key_values_lens=cfg_text_key_values_lens,
                    cfg_text_past_key_values=cfg_text_past_key_values,
                    cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                    # cfg_img
                    cfg_img_scale=cfg_img_scale_,
                    cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                    cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                    cfg_img_key_values_lens=cfg_img_key_values_lens,
                    cfg_img_past_key_values=cfg_img_past_key_values,
                    cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                    cfg_type=cfg_type,
                    packed_boundary_token_indexes=packed_boundary_token_indexes,
                    model_pred_cache_dic=model_pred_cache_dic,
                    model_pred_current=model_pred_current,
                    model_pred_text_cache_dic=model_pred_text_cache_dic,
                    model_pred_text_current=model_pred_text_current,
                    model_pred_img_cache_dic=model_pred_img_cache_dic,
                    model_pred_img_current=model_pred_img_current,
                )
            if not memory_loop_enabled:
                v_t = result

            if i in selected_sde_steps:
                next_t = timesteps[i + 1] if i + 1 < len(timesteps) else t.new_zeros(())
                noise_generator = torch.Generator(device="cpu").manual_seed(
                    int(sde_seed) + int(i) * 1_000_003
                )
                shared_noise = torch.randn(
                    tuple(x_t.shape),
                    generator=noise_generator,
                    dtype=torch.float32,
                ).to(x_t.device)
                transition = sde_step_with_logprob(
                    v_t,
                    timestep=t,
                    next_timestep=next_t,
                    sample=x_t,
                    sigma_max=timesteps[1] if len(timesteps) > 1 else 0.999,
                    noise_level=float(sde_noise_level),
                    noise=shared_noise,
                )
                trajectory.append(
                    {
                        "step_index": int(i),
                        "sample": transition.sample.detach().clone(),
                        "next_sample": transition.next_sample.detach().clone(),
                        "timestep": t.detach().clone(),
                        "next_timestep": next_t.detach().clone(),
                        "old_log_prob": transition.log_prob.detach().clone(),
                        "sigma_max": (
                            timesteps[1].detach().clone()
                            if len(timesteps) > 1
                            else t.new_tensor(0.999)
                        ),
                        "noise_level": float(sde_noise_level),
                        "m_in": None if m_in is None else m_in.detach().clone(),
                        "m_in_text": (
                            None if m_in_text is None else m_in_text.detach().clone()
                        ),
                        "m_in_img": (
                            None if m_in_img is None else m_in_img.detach().clone()
                        ),
                        "m_out": None if m_out is None else m_out.detach().clone(),
                    }
                )
                x_t = transition.next_sample
            else:
                # Velocity points from data to noise; generation integrates in
                # the reverse direction and therefore subtracts v * dt.
                x_t = self.image_euler_step(x_t, v_t, dts[i])

        if enable_taylorseer:
            del model_pred_cache_dic, model_pred_current
            del model_pred_text_cache_dic, model_pred_text_current
            del model_pred_img_cache_dic, model_pred_img_current

        unpacked_latent = x_t.split(packed_vae_seqlens.tolist())
        if return_trajectory:
            return unpacked_latent, tuple(trajectory)
        return unpacked_latent

    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        packed_boundary_token_indexes: Optional[torch.LongTensor] = None,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        within_step_loop_start: Optional[int] = None,
        within_step_loop_end: Optional[int] = None,
        within_step_loop_repeat: int = 1,
        within_step_loop_damping: float = 1.0,
        # cache
        model_pred_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_current: Optional[int] = None,
        model_pred_text_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_text_current: Optional[int] = None,
        model_pred_img_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_img_current: Optional[int] = None,
    ):
        # This primitive is shared by inference and loop-adapter training.
        # Inference callers already own a no-grad context; decorating this
        # method would silently detach the SFT/RL loss from the loop LoRA.
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(
            (sum(packed_seqlens), self.hidden_size)
        )
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes,
            }

        if int(within_step_loop_repeat) > 1:
            extra_inputs.update(
                within_step_loop_start=within_step_loop_start,
                within_step_loop_end=within_step_loop_end,
                within_step_loop_repeat=int(within_step_loop_repeat),
                within_step_loop_damping=float(within_step_loop_damping),
            )

        if getattr(self.language_model.model, "enable_taylorseer", False):
            self.language_model.model.cache_dic = model_pred_cache_dic
            self.language_model.model.current = model_pred_current

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            packed_boundary_token_indexes=packed_boundary_token_indexes,
            **extra_inputs,
        )
        v_t = self.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if getattr(self.language_model.model, "enable_taylorseer", False):
                self.language_model.model.cache_dic = model_pred_text_cache_dic
                self.language_model.model.current = model_pred_text_current
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            if getattr(self.language_model.model, "enable_taylorseer", False):
                self.language_model.model.cache_dic = model_pred_img_cache_dic
                self.language_model.model.current = model_pred_img_current
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(
                    min=cfg_renorm_min, max=1.0
                )
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)

                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(
                    min=cfg_renorm_min, max=1.0
                )
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        return v_t

    def _combine_cfg_velocities(
        self,
        v_t,
        cfg_text_v_t,
        cfg_img_v_t,
        *,
        cfg_text_scale,
        cfg_img_scale,
        cfg_renorm_min,
        cfg_renorm_type,
    ):
        if cfg_text_scale <= 1.0:
            return v_t
        if cfg_renorm_type == "text_channel":
            v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
            norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
            norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
            scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(
                min=cfg_renorm_min, max=1.0
            )
            v_t_text = v_t_text_ * scale
            if cfg_img_scale > 1.0:
                return cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
            return v_t_text
        v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
        if cfg_img_scale > 1.0:
            v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
        else:
            v_t_ = v_t_text_
        if cfg_renorm_type == "global":
            norm_v_t = torch.norm(v_t)
            norm_v_t_ = torch.norm(v_t_)
        elif cfg_renorm_type == "channel":
            norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
            norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
        else:
            raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
        scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
        return v_t_ * scale

    def _forward_flow_loop(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        packed_loop_token_indexes: torch.LongTensor,
        loop_memory: torch.Tensor,
        loop_memory_text: Optional[torch.Tensor] = None,
        loop_memory_img: Optional[torch.Tensor] = None,
        packed_boundary_token_indexes: Optional[torch.LongTensor] = None,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        recycle_mode: str = "same_depth",
        memory_loop_repeat: int = 1,
        memory_loop_start: Optional[int] = None,
        memory_loop_end: Optional[int] = None,
        memory_body_in: Optional[torch.Tensor] = None,
        memory_body_in_text: Optional[torch.Tensor] = None,
        memory_body_in_img: Optional[torch.Tensor] = None,
        embed_memory: Optional[torch.Tensor] = None,
        round0_gen_reads_memory: bool = False,
    ):
        """Memory-loop velocity. Not @torch.no_grad.

        same_depth: prefix once, recycle memory only inside [s, e), suffix once.
        full_depth: one full F_1:L; caller repeats R times at embedding level.
        """

        if int(packed_loop_token_indexes.numel()) == 0:
            raise ValueError("_forward_flow_loop requires packed_loop_token_indexes")
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(
            (sum(packed_seqlens), self.hidden_size)
        )
        packed_sequence[packed_text_indexes] = packed_text_embedding
        embed_m = (embed_memory if embed_memory is not None else loop_memory).to(
            dtype=packed_sequence.dtype, device=packed_sequence.device
        )
        packed_sequence[packed_loop_token_indexes] = embed_m

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        vae_hidden = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if vae_hidden.dtype != packed_sequence.dtype:
            vae_hidden = vae_hidden.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = vae_hidden

        und_indexes = self.mot_und_route_indexes(
            packed_text_indexes, packed_loop_token_indexes
        )
        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": und_indexes,
            }
        same_depth = str(recycle_mode) == "same_depth"
        block_gen = not bool(round0_gen_reads_memory)
        if same_depth:
            extra_inputs.update(
                packed_memory_token_indexes=packed_loop_token_indexes,
                memory_loop_repeat=int(memory_loop_repeat),
                memory_loop_start=memory_loop_start,
                memory_loop_end=memory_loop_end,
                block_gen_reads_memory=block_gen,
            )
        else:
            extra_inputs.update(
                packed_memory_token_indexes=packed_loop_token_indexes,
                block_gen_reads_memory=block_gen,
            )

        def run_branch(sequence, kv, pos_ids, query_indexes, kv_lens, kv_indexes, body_in):
            kwargs = dict(extra_inputs)
            if same_depth:
                kwargs["memory_body_in"] = body_in
            output = self.language_model.forward_inference(
                packed_query_sequence=sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=pos_ids,
                packed_query_indexes=query_indexes,
                past_key_values=kv,
                key_values_lens=kv_lens,
                packed_key_value_indexes=kv_indexes,
                update_past_key_values=False,
                is_causal=False,
                packed_boundary_token_indexes=packed_boundary_token_indexes,
                **kwargs,
            )
            velocity = self.llm2vae(output.packed_query_sequence)[
                packed_vae_token_indexes
            ]
            if same_depth and output.memory_body_out is not None:
                memory_next = output.memory_body_out
            else:
                memory_next = output.packed_query_sequence[packed_loop_token_indexes]
            vae_out = output.packed_query_sequence[packed_vae_token_indexes]
            return velocity, memory_next, vae_out, output

        v_t, m_full, vae_out, cond_out = run_branch(
            packed_sequence,
            past_key_values,
            packed_position_ids,
            packed_indexes,
            key_values_lens,
            packed_key_value_indexes,
            memory_body_in,
        )
        m_text = m_full
        m_img = m_full
        cfg_text_v_t = None
        cfg_img_v_t = None
        if cfg_text_scale > 1.0:
            seq_text = packed_sequence.clone()
            text_embed = (
                loop_memory_text if loop_memory_text is not None else embed_m
            )
            seq_text[packed_loop_token_indexes] = text_embed.to(
                dtype=packed_sequence.dtype, device=packed_sequence.device
            )
            cfg_text_v_t, m_text, _, text_out = run_branch(
                seq_text,
                cfg_text_past_key_values,
                cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes,
                cfg_text_key_values_lens,
                cfg_text_packed_key_value_indexes,
                memory_body_in_text,
            )
        else:
            text_out = None
        if cfg_img_scale > 1.0:
            seq_img = packed_sequence.clone()
            img_embed = loop_memory_img if loop_memory_img is not None else embed_m
            seq_img[packed_loop_token_indexes] = img_embed.to(
                dtype=packed_sequence.dtype, device=packed_sequence.device
            )
            cfg_img_v_t, m_img, _, img_out = run_branch(
                seq_img,
                cfg_img_past_key_values,
                cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes,
                cfg_img_key_values_lens,
                cfg_img_packed_key_value_indexes,
                memory_body_in_img,
            )
        else:
            img_out = None
        v_t = self._combine_cfg_velocities(
            v_t,
            cfg_text_v_t,
            cfg_img_v_t,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
        )
        stats = self.memory_slot_stats(m_full)
        delta_m: List[float] = []
        delta_g: List[float] = []
        delta_v: List[float] = []
        mem_rounds = getattr(cond_out, "memory_round_hiddens", None) or ()
        gen_rounds = getattr(cond_out, "gen_round_hiddens", None) or ()
        for prev, nxt in zip(mem_rounds, mem_rounds[1:]):
            delta_m.append(self.relative_l2(nxt, prev))
        for prev, nxt in zip(gen_rounds, gen_rounds[1:]):
            delta_g.append(self.relative_l2(nxt, prev))

        def _suffix_velocities(output) -> List[torch.Tensor]:
            hidden_rounds = getattr(output, "gen_suffix_round_hiddens", None) or ()
            return [self.llm2vae(hidden) for hidden in hidden_rounds]

        cond_vs = _suffix_velocities(cond_out)
        text_vs = _suffix_velocities(text_out) if text_out is not None else []
        img_vs = _suffix_velocities(img_out) if img_out is not None else []
        combined = []
        for index, cond_v in enumerate(cond_vs):
            text_v = text_vs[index] if index < len(text_vs) else None
            img_v = img_vs[index] if index < len(img_vs) else None
            combined.append(
                self._combine_cfg_velocities(
                    cond_v,
                    text_v,
                    img_v,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )
            )
        if combined:
            v1 = combined[0]
            for nxt in combined[1:]:
                delta_v.append(self.relative_l2(nxt, v1))
        diagnostics = {
            "memory_rms": float(
                torch.linalg.vector_norm(m_full.float())
                / max(m_full.numel() ** 0.5, 1.0)
            ),
            "vae_hidden_rms": float(
                torch.linalg.vector_norm(vae_out.float())
                / max(vae_out.numel() ** 0.5, 1.0)
            ),
            "velocity_norm": float(torch.linalg.vector_norm(v_t.float())),
            "delta_m": delta_m,
            "delta_g": delta_g,
            "delta_v": delta_v,
            **stats,
        }
        return v_t, m_full, m_text, m_img, diagnostics

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids["bos_token_id"])
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(
                packed_query_position_ids, dtype=torch.long
            ),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(
                packed_key_value_indexes, dtype=torch.long
            ),
        }

        return generation_input

    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
        repetition_penalty: float = 1.0,
        return_last_hidden: bool = False,
    ):
        # Keskar et al. 2019 "CTRL" repetition penalty: for any token that
        # has already been emitted in this generation, divide its logit by
        # `repetition_penalty` if positive and multiply if negative. Per-row
        # (batch size = 1 in current callers; we still implement it per-row
        # for correctness if that ever changes).
        apply_rep_penalty = float(repetition_penalty) != 1.0
        prev_tokens_per_row: List[List[int]] = []

        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            if apply_rep_penalty:
                if not prev_tokens_per_row:
                    prev_tokens_per_row = [[] for _ in range(curr_tokens.shape[0])]
                tokens_cpu = curr_tokens.detach().tolist()
                for row_idx, tok in enumerate(tokens_cpu):
                    prev_tokens_per_row[row_idx].append(int(tok))
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0,
                len(key_values_lens),
                device=key_values_lens.device,
                dtype=key_values_lens.dtype,
            )

            uppacked = list(
                packed_key_value_indexes.split(key_values_lens.tolist(), dim=0)
            )
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            if apply_rep_penalty:
                penalty = float(repetition_penalty)
                for row_idx, prev_ids in enumerate(prev_tokens_per_row):
                    if not prev_ids:
                        continue
                    idx = torch.as_tensor(
                        prev_ids, device=pred_logits.device, dtype=torch.long
                    )
                    row = pred_logits[row_idx]
                    row_at_prev = row.index_select(0, idx)
                    row_at_prev = torch.where(
                        row_at_prev > 0, row_at_prev / penalty, row_at_prev * penalty
                    )
                    pred_logits[row_idx] = row.index_copy(0, idx, row_at_prev)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(
                packed_key_value_indexes.split(key_values_lens.tolist(), dim=0)
            )
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [
                        uppacked[i],
                        torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device),
                    ],
                    dim=0,
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if (
                end_token_id is not None and curr_tokens[0] == end_token_id
            ):  # only support batch=1
                break

        output_device = generated_sequence[0].device
        token_ids = torch.stack(
            [i.to(output_device) for i in generated_sequence], dim=0
        )
        if return_last_hidden:
            return token_ids, packed_query_sequence
        return token_ids

    # for evaluation
    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        images,
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # add images
        for image in images:
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope,
                images=[image],
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(
                    past_key_values, **generation_input
                )

        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            prompts=[prompt],
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(
                past_key_values, **generation_input
            )

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids["eos_token_id"],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:, 0])
        output = output.split("<|im_end|>")[0].split("<|im_start|>")[1]

        return output
