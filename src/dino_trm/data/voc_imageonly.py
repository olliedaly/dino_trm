"""Image-only VOC training set for scaled-up unsupervised training.

Uses `ernestchu/voc2012-image-only` (17,125 images) restricted to the leakage-free
index built by ``scripts/build_train_index.py`` (val images removed). Returns only
``pixel_values`` (training is unsupervised, no masks needed). Same aspect-preserving
crop + horizontal-flip augmentation as the labelled loader, so the recipe matches.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .pascal_voc import IMAGENET_MEAN, IMAGENET_STD

IMAGEONLY_ID = "ernestchu/voc2012-image-only"
INDEX_PATH = "data/train_index.npy"


class VOCImageOnly(Dataset):
    def __init__(
        self,
        image_size: int = 336,
        augment: bool = True,
        index_path: str = INDEX_PATH,
        cache_dir: str | None = None,
    ) -> None:
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"{index_path} missing — run `uv run python scripts/build_train_index.py` first."
            )
        self.ds = load_dataset(IMAGEONLY_ID, split="train", cache_dir=cache_dir)
        self.index = np.load(index_path)
        self.image_size = image_size
        self.augment = augment
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.index)

    def _resize_crop(self, pil: Image.Image) -> Image.Image:
        w, h = pil.size
        scale = self.image_size / min(w, h)
        nw, nh = round(w * scale), round(h * scale)
        pil = pil.resize((nw, nh), Image.BILINEAR)
        left = (nw - self.image_size) // 2
        top = (nh - self.image_size) // 2
        return pil.crop((left, top, left + self.image_size, top + self.image_size))

    def __getitem__(self, i: int) -> dict:
        ex = self.ds[int(self.index[i])]
        img = self._resize_crop(ex["image"].convert("RGB"))
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return {"pixel_values": (arr - self._mean) / self._std}


def build_imageonly_loader(
    batch_size: int = 32,
    image_size: int = 336,
    num_workers: int = 8,
    cache_dir: str | None = None,
) -> DataLoader:
    ds = VOCImageOnly(image_size=image_size, augment=True, cache_dir=cache_dir)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
