"""Reward adapters for BAGEL loop GRPO.

GenEval supplies the semantic objective from decoded images. The trained FLUX
Diffusion-RM consumes BAGEL's compatible 16-channel clean VAE latent directly
and acts only as a quality guardrail.
"""

from __future__ import annotations

import hashlib
import pickle
import socket
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image


FLUX_VAE_CONTRACT = {
    "latent_channels": 16,
    "latent_downsample": 8,
    "scale_factor": 0.3611,
    "shift_factor": 0.1159,
}


def audit_bagel_flux_vae_contract(vae_model: torch.nn.Module) -> dict[str, float | int]:
    """Fail closed unless BAGEL exposes the FLUX latent tensor contract."""

    observed = {
        "latent_channels": int(getattr(vae_model, "latent_channels", -1)),
        "latent_downsample": int(getattr(vae_model, "latent_downsample", -1)),
        "scale_factor": float(getattr(vae_model, "scale_factor", float("nan"))),
        "shift_factor": float(getattr(vae_model, "shift_factor", float("nan"))),
    }
    mismatches = []
    for key in ("latent_channels", "latent_downsample"):
        if observed[key] != FLUX_VAE_CONTRACT[key]:
            mismatches.append(
                f"{key}={observed[key]!r}, expected {FLUX_VAE_CONTRACT[key]!r}"
            )
    for key in ("scale_factor", "shift_factor"):
        if not np.isclose(
            float(observed[key]), float(FLUX_VAE_CONTRACT[key]), rtol=0.0, atol=1e-6
        ):
            mismatches.append(
                f"{key}={observed[key]!r}, expected {FLUX_VAE_CONTRACT[key]!r}"
            )
    if mismatches:
        raise RuntimeError(
            "BAGEL clean latents are not compatible with the configured FLUX "
            "Diffusion-RM: " + "; ".join(mismatches)
        )
    return observed


def unpack_bagel_latents(
    packed_latents: Sequence[torch.Tensor] | torch.Tensor,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    """Convert packed BAGEL [tokens,64] latents to FLUX [B,16,H/8,W/8]."""

    if isinstance(packed_latents, torch.Tensor):
        packed = packed_latents
        if packed.ndim == 2:
            packed = packed.unsqueeze(0)
    else:
        packed = torch.stack(list(packed_latents), dim=0)
    if packed.ndim != 3 or int(packed.shape[-1]) != 64:
        raise ValueError(
            f"expected BAGEL packed latents [B,tokens,64], got {tuple(packed.shape)}"
        )
    image_h, image_w = (int(value) for value in image_shape)
    grid_h, grid_w = image_h // 16, image_w // 16
    if int(packed.shape[1]) != grid_h * grid_w:
        raise ValueError(
            f"packed token count {packed.shape[1]} does not match image shape "
            f"{tuple(image_shape)}"
        )
    latent = packed.reshape(-1, grid_h, grid_w, 2, 2, 16)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    return latent.reshape(-1, 16, grid_h * 2, grid_w * 2).contiguous()


class GenEvalRewardClient:
    """Small fail-fast client for the standard local GenEval reward service."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:18085",
        *,
        timeout_seconds: float = 120.0,
        batch_size: int = 64,
    ) -> None:
        import requests
        from requests.adapters import HTTPAdapter, Retry

        self.url = str(url).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.batch_size = int(batch_size)
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=False,
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def check_available(self, *, timeout_seconds: float = 2.0) -> None:
        """Fail before model loading when the local evaluator is not listening."""

        parsed = urlparse(self.url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            raise ValueError(f"invalid GenEval URL: {self.url}")
        try:
            with socket.create_connection((host, port), timeout=float(timeout_seconds)):
                pass
        except OSError as error:
            raise RuntimeError(
                f"GenEval service is not listening at {host}:{port}"
            ) from error

    @staticmethod
    def _jpeg_bytes(image: Any) -> bytes:
        if isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            if isinstance(image, torch.Tensor):
                tensor = image.detach().cpu()
                if tensor.ndim == 3 and int(tensor.shape[0]) in (1, 3, 4):
                    tensor = tensor.permute(1, 2, 0)
                array = tensor.numpy()
            else:
                array = np.asarray(image)
            if np.issubdtype(array.dtype, np.floating):
                if float(array.max(initial=0.0)) <= 1.0:
                    array = array * 255.0
                array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8, copy=False)
            pil = Image.fromarray(array).convert("RGB")
        buffer = BytesIO()
        pil.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()

    def score(
        self,
        images: Sequence[Any],
        metadatas: Sequence[Mapping[str, Any]],
        *,
        only_strict: bool = True,
    ) -> torch.Tensor:
        if len(images) != len(metadatas) or not images:
            raise ValueError("GenEval images and metadata must have equal non-zero length")
        scores = []
        for start in range(0, len(images), self.batch_size):
            image_batch = images[start : start + self.batch_size]
            metadata_batch = metadatas[start : start + self.batch_size]
            payload = {
                "images": [self._jpeg_bytes(image) for image in image_batch],
                "meta_datas": list(metadata_batch),
                "only_strict": bool(only_strict),
            }
            try:
                response = self.session.post(
                    self.url,
                    data=pickle.dumps(payload),
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                result = pickle.loads(response.content)
            except Exception as error:
                raise RuntimeError(
                    f"GenEval service unavailable at {self.url}; start the evaluator "
                    "before GRPO training"
                ) from error
            if "scores" not in result:
                raise RuntimeError("GenEval response has no 'scores' field")
            scores.extend(float(value) for value in result["scores"])
        if len(scores) != len(images):
            raise RuntimeError(
                f"GenEval returned {len(scores)} scores for {len(images)} images"
            )
        return torch.tensor(scores, dtype=torch.float32)


class FluxLatentReward:
    """Frozen FLUX Diffusion-RM scorer for clean BAGEL VAE latents."""

    def __init__(
        self,
        *,
        diffusion_rm_repo: str,
        config_path: str,
        checkpoint_path: str,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        noise_level: float = 0.05,
        noise_levels: Sequence[float] | None = None,
        noise_seed: int = 17,
    ) -> None:
        import yaml
        from diffusers import FluxPipeline
        from omegaconf import OmegaConf

        repo = str(Path(diffusion_rm_repo).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from diffusion_rm.models.flux_rm import FLUXRewardModel

        self.device = torch.device(device)
        self.dtype = dtype
        self.noise_levels = tuple(
            float(value)
            for value in (noise_levels if noise_levels is not None else (noise_level,))
        )
        self.noise_seed = int(noise_seed)
        if not self.noise_levels or any(
            not 0.0 <= value <= 1.0 for value in self.noise_levels
        ):
            raise ValueError("FLUX reward noise levels must be non-empty and in [0,1]")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = OmegaConf.create(yaml.safe_load(handle))
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"FLUX reward checkpoint not found: {checkpoint}")

        pipeline = FluxPipeline.from_pretrained(
            config.model.backbone_model_id,
            torch_dtype=dtype,
        )
        pipeline.text_encoder.to(self.device, dtype=dtype)
        pipeline.text_encoder_2.to(self.device, dtype=dtype)
        model = FLUXRewardModel(
            pipeline=pipeline,
            config_model=config.model,
            device=self.device,
            dtype=dtype,
            vae_scale_factor=pipeline.vae_scale_factor,
        )
        model.backbone.load_adapter(
            str(checkpoint / "backbone_lora"), adapter_name="rm_eval"
        )
        model.backbone.set_adapter("rm_eval")
        head_state = torch.load(
            checkpoint / "rm_head.pt", map_location="cpu", weights_only=True
        )
        model.reward_head.load_state_dict(head_state, strict=True)
        self.scheduler = pipeline.scheduler
        self.model = model.eval().requires_grad_(False)

    def _fixed_noise(self, shape, prompt: str) -> torch.Tensor:
        digest = hashlib.blake2b(
            f"{self.noise_seed}\0{prompt}".encode(), digest_size=8
        ).digest()
        seed = int.from_bytes(digest, "little") & ((1 << 63) - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            self.device, self.dtype
        )

    def score(
        self,
        packed_latents: Sequence[torch.Tensor] | torch.Tensor,
        prompts: Sequence[str],
        *,
        image_shape: tuple[int, int],
    ) -> torch.Tensor:
        latents = unpack_bagel_latents(packed_latents, image_shape)
        if int(latents.shape[0]) != len(prompts) or not prompts:
            raise ValueError("FLUX reward latents and prompts must have equal non-zero length")
        latents = latents.to(device=self.device, dtype=self.dtype)
        noises = torch.cat(
            [self._fixed_noise(latents[:1].shape, str(prompt)) for prompt in prompts],
            dim=0,
        )
        with torch.inference_mode():
            text = self.model.encode_prompt([str(prompt) for prompt in prompts])
            scores = []
            sigmas = self.scheduler.sigmas.to(device=self.device, dtype=torch.float32)
            timesteps = self.scheduler.timesteps.to(self.device)
            for noise_level in self.noise_levels:
                index = int(torch.argmin((sigmas - noise_level).abs()))
                sigma = sigmas[index].to(dtype=self.dtype)
                noisy = (1.0 - sigma) * latents + sigma * noises
                scores.append(
                    self.model(
                        latents=noisy,
                        timesteps=timesteps[index].expand(len(prompts)),
                        **text,
                    ).float().flatten()
                )
        return torch.stack(scores, dim=0).mean(dim=0).cpu()


__all__ = [
    "FLUX_VAE_CONTRACT",
    "FluxLatentReward",
    "GenEvalRewardClient",
    "audit_bagel_flux_vae_contract",
    "unpack_bagel_latents",
]
