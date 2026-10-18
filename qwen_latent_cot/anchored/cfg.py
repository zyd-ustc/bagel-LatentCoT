"""CFG combination copied from ``Bagel._forward_flow``."""
from __future__ import annotations

import torch


def combine_cfg(
    conditional: torch.Tensor,
    cfg_text: torch.Tensor | None,
    cfg_image: torch.Tensor | None,
    cfg_text_scale: float,
    cfg_img_scale: float,
    cfg_renorm_type: str,
    cfg_renorm_min: float,
) -> torch.Tensor:
    if cfg_text_scale <= 1.0:
        return conditional
    if cfg_text is None:
        raise ValueError("text CFG scale needs a text-branch velocity")
    if cfg_renorm_type == "text_channel":
        guided = cfg_text + cfg_text_scale * (conditional - cfg_text)
        scale = (torch.norm(conditional, dim=-1, keepdim=True) / (torch.norm(guided, dim=-1, keepdim=True) + 1e-8))
        guided = guided * scale.clamp(min=cfg_renorm_min, max=1.0)
        if cfg_img_scale > 1.0:
            if cfg_image is None:
                raise ValueError("image CFG scale needs an image-branch velocity")
            return cfg_image + cfg_img_scale * (guided - cfg_image)
        return guided
    guided = cfg_text + cfg_text_scale * (conditional - cfg_text)
    if cfg_img_scale > 1.0:
        if cfg_image is None:
            raise ValueError("image CFG scale needs an image-branch velocity")
        guided = cfg_image + cfg_img_scale * (guided - cfg_image)
    if cfg_renorm_type == "global":
        norm_native = torch.norm(conditional)
        norm_guided = torch.norm(guided)
    elif cfg_renorm_type == "channel":
        norm_native = torch.norm(conditional, dim=-1, keepdim=True)
        norm_guided = torch.norm(guided, dim=-1, keepdim=True)
    else:
        raise NotImplementedError(f"{cfg_renorm_type} is not supported")
    scale = (norm_native / (norm_guided + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
    return guided * scale
