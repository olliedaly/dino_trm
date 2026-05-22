"""Phase 0 sanity check.

Loads the frozen ViT backbone (DINOv2 ViT-S/14 by default; DINOv3 ViT-S/16 once
gated access is in place), runs one synthetic image through it in bf16 on the GPU,
and verifies we get sensible patch features. Confirms the Blackwell (sm_120) +
CUDA 12.8 + bf16 stack works end-to-end before we build anything on top.

Run:  uv run python scripts/sanity_check.py
      uv run python scripts/sanity_check.py facebook/dinov3-vits16-pretrain-lvd1689m
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from PIL import Image

from dino_trm.models.backbone import DINOV2_S, FrozenBackbone

IMG_SIZE = 224


def make_dummy_image() -> Image.Image:
    """Deterministic synthetic RGB image so the check needs no network/data."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else DINOV2_S

    assert torch.cuda.is_available(), "CUDA not available"
    cap = torch.cuda.get_device_capability()
    print(f"device       : {torch.cuda.get_device_name(0)}")
    print(f"capability   : {cap}  (expect (12, 0) for Blackwell sm_120)")
    print(f"torch        : {torch.__version__}")
    if cap[0] < 12:
        print("WARNING: device capability < 12; not the expected Blackwell GPU")

    device = torch.device("cuda")
    print(f"\nloading {model_id} ...")
    backbone = FrozenBackbone(model_id=model_id, image_size=IMG_SIZE).to(device)
    n_params = sum(p.numel() for p in backbone.model.parameters())
    print(f"backbone       : {backbone.extra_repr()}")
    print(f"backbone params: {n_params/1e6:.1f}M (frozen, eval, bf16)")

    image = make_dummy_image()
    inputs = backbone.processor(images=image, return_tensors="pt").to(device)
    print(f"pixel_values   : {tuple(inputs['pixel_values'].shape)} {inputs['pixel_values'].dtype}")

    patches = backbone(inputs["pixel_values"])
    print(
        f"patch tokens   : {tuple(patches.shape)} {patches.dtype}  "
        f"(expect (1, {backbone.num_patches}, {backbone.feature_dim}))"
    )

    ok = (
        patches.shape[0] == 1
        and patches.shape[1] == backbone.num_patches
        and patches.shape[2] == backbone.feature_dim
    )
    has_nan = torch.isnan(patches.float()).any().item()
    print(f"contains NaN   : {has_nan}")

    if ok and not has_nan:
        print("\nPHASE 0 SANITY CHECK: PASS")
        return 0
    print("\nPHASE 0 SANITY CHECK: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
