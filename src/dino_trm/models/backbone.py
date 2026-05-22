"""Frozen ViT backbone wrapper (DINOv2 now, DINOv3 once access is granted).

The rest of the pipeline never hardcodes the patch count or feature dim; it reads
``num_patches``, ``feature_dim`` and ``grid_size`` off this wrapper. That keeps the
DINOv2 -> DINOv3 swap a one-line config change despite their different patch sizes:

    DINOv2 ViT-S/14 @ 224  -> 16x16 = 256 patches, dim 384
    DINOv3 ViT-S/16 @ 224  -> 14x14 = 196 patches, dim 384
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

# Default development backbone (open / non-gated). Swap to the DINOv3 id below
# once HF gated access + a gated-repo-read token are in place.
DINOV2_S = "facebook/dinov2-small"
DINOV3_S = "facebook/dinov3-vits16-pretrain-lvd1689m"


class FrozenBackbone(nn.Module):
    """Wraps a HF ViT, exposes patch tokens only, frozen + eval + bf16.

    forward(pixel_values) -> (B, N, D) patch features, CLS and register tokens
    dropped. The backbone runs under ``torch.no_grad`` so no activations are kept
    for backprop (it has no trainable params anyway).
    """

    def __init__(
        self,
        model_id: str = DINOV3_S,
        dtype: torch.dtype = torch.bfloat16,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.dtype = dtype
        self.image_size = image_size

        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        cfg = self.model.config
        self.feature_dim: int = cfg.hidden_size
        self.patch_size: int = cfg.patch_size
        self.num_register_tokens: int = int(getattr(cfg, "num_register_tokens", 0) or 0)
        # Number of leading non-patch tokens to drop: CLS + registers.
        self._num_prefix_tokens: int = 1 + self.num_register_tokens

        self.grid_size: int = image_size // self.patch_size
        self.num_patches: int = self.grid_size * self.grid_size

        # ImageNet normalisation stats the processor uses (handy for un-normalising
        # images in visualisations).
        self.image_mean = getattr(self.processor, "image_mean", [0.485, 0.456, 0.406])
        self.image_std = getattr(self.processor, "image_std", [0.229, 0.224, 0.225])

    def train(self, mode: bool = True):  # noqa: D401 - keep backbone in eval always
        """Override so the backbone never leaves eval mode (frozen BN/dropout)."""
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.dtype)
        out = self.model(pixel_values=pixel_values)
        tokens = out.last_hidden_state  # (B, prefix + N, D)
        patches = tokens[:, self._num_prefix_tokens :, :]
        return patches

    def extra_repr(self) -> str:
        return (
            f"model_id={self.model_id}, feature_dim={self.feature_dim}, "
            f"num_patches={self.num_patches}, grid_size={self.grid_size}, "
            f"patch_size={self.patch_size}, registers={self.num_register_tokens}"
        )
