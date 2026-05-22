"""PASCAL VOC 2012 loader for object-centric training/eval.

Uses the HuggingFace dataset ``nateraw/pascal-voc-2012`` (fields: ``image``,
``mask``), which carries the VOC SegmentationClass palette masks. We:

  * resize images to a square ``image_size`` and normalise with the backbone's
    ImageNet mean/std (explicit transform, so image<->mask alignment is exact and
    independent of the HF processor's resize/crop quirks);
  * convert the RGB palette mask to integer class labels and nearest-neighbour
    downsample to the ViT patch grid, so each patch token has one GT label.

Background is class 0; void/boundary (palette colour (224,224,192)) maps to 255 and
is ignored by metrics. NOTE: these are *semantic* masks, so two instances of the
same class share a label — fine for FG-ARI (fg pixels) and mBO_c, but true mBO_i
would need SegmentationObject masks (swap the dataset when available).
"""

from __future__ import annotations

import random

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DATASET_ID = "nateraw/pascal-voc-2012"
VOID_LABEL = 255

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def voc_colormap(n: int = 256) -> np.ndarray:
    """Standard PASCAL VOC colour map: index -> RGB."""
    def bitget(value: int, idx: int) -> int:
        return (value >> idx) & 1

    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= bitget(c, 0) << (7 - j)
            g |= bitget(c, 1) << (7 - j)
            b |= bitget(c, 2) << (7 - j)
            c >>= 3
        cmap[i] = (r, g, b)
    return cmap


_CMAP = voc_colormap()
# Encode each palette RGB as a single int for fast lookup; map -> label index.
_RGB2LABEL: dict[int, int] = {}
for _idx in range(21):  # 0 background + 1..20 classes
    r, g, b = (int(x) for x in _CMAP[_idx])
    _RGB2LABEL[(r << 16) | (g << 8) | b] = _idx
# Void / boundary colour.
_RGB2LABEL[(224 << 16) | (224 << 8) | 192] = VOID_LABEL


def rgb_mask_to_label(mask_rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 VOC palette mask -> (H, W) int label map (255 = void)."""
    h, w, _ = mask_rgb.shape
    flat = mask_rgb.reshape(-1, 3).astype(np.int64)
    keys = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    out = np.zeros(h * w, dtype=np.int64)
    # Unknown colours (palette interpolation artefacts) -> void.
    for k in np.unique(keys):
        out[keys == k] = _RGB2LABEL.get(int(k), VOID_LABEL)
    return out.reshape(h, w)


class PascalVOC(Dataset):
    """Returns dict(pixel_values=(3,H,W) float, label=(grid,grid) long)."""

    def __init__(
        self,
        split: str = "train",
        image_size: int = 336,
        grid_size: int = 21,
        cache_dir: str | None = None,
        augment: bool = False,
        full_mask: bool = False,
    ) -> None:
        self.ds = load_dataset(DATASET_ID, split=split, cache_dir=cache_dir)
        self.image_size = image_size
        self.grid_size = grid_size
        self.augment = augment
        self.full_mask = full_mask  # also return image-resolution semantic mask for eval
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.ds)

    def _resize_crop(self, pil: Image.Image, resample) -> Image.Image:
        """Aspect-preserving: resize the shorter side to image_size, then centre crop.

        Avoids the object distortion that a naive square resize introduces.
        """
        w, h = pil.size
        scale = self.image_size / min(w, h)
        nw, nh = round(w * scale), round(h * scale)
        pil = pil.resize((nw, nh), resample)
        left = (nw - self.image_size) // 2
        top = (nh - self.image_size) // 2
        return pil.crop((left, top, left + self.image_size, top + self.image_size))

    def __getitem__(self, i: int) -> dict:
        ex = self.ds[i]
        img = ex["image"].convert("RGB")
        label = rgb_mask_to_label(np.asarray(ex["mask"].convert("RGB")))  # (H, W) int
        label_pil = Image.fromarray(label.astype(np.int32), mode="I")

        # Identical geometric transform for image and label so they stay aligned.
        img = self._resize_crop(img, Image.BILINEAR)
        label_pil = self._resize_crop(label_pil, Image.NEAREST)
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label_pil = label_pil.transpose(Image.FLIP_LEFT_RIGHT)

        arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        pixel_values = (arr - self._mean) / self._std

        lab_grid = label_pil.resize((self.grid_size, self.grid_size), Image.NEAREST)
        label = torch.from_numpy(np.asarray(lab_grid, dtype=np.int64))
        out = {"pixel_values": pixel_values, "label": label}
        if self.full_mask:
            # Image-resolution semantic mask for the published-protocol eval.
            out["label_full"] = torch.from_numpy(np.asarray(label_pil, dtype=np.int64))
        return out


def build_loader(
    split: str = "train",
    batch_size: int = 32,
    image_size: int = 336,
    grid_size: int = 21,
    num_workers: int = 4,
    shuffle: bool | None = None,
    cache_dir: str | None = None,
    augment: bool | None = None,
    full_mask: bool = False,
) -> DataLoader:
    if augment is None:
        augment = split == "train"
    ds = PascalVOC(
        split=split,
        image_size=image_size,
        grid_size=grid_size,
        cache_dir=cache_dir,
        augment=augment,
        full_mask=full_mask,
    )
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=num_workers > 0,
    )
