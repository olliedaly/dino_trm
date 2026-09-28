"""MOVi-C loader — a synthetic third domain for the object-size analysis.

WHY MOVi-C. DINOSAUR benchmarked on it, ground truth is complete by construction (no
annotation gaps, no connected-component instance derivation), and its images span a
wide range of object sizes, so both ends of the size analysis are populated. MOVi-E has
more objects per frame, but they average out: its PER-IMAGE mean object size varies
much less, and the analysis unit is the image.

GEOMETRY. Frames are square (256x256), so the aspect-preserving resize is a pure
rescale with no centre crop.

Instance ids come straight from the renderer, so no connected-component derivation and
none of the VOC merging ambiguity. Segmentation id k is instance k-1; id 0 is
background. Semantic ids are ShapeNet category indices, used for mBO_c.

Expects ``scripts/prepare_movi.py`` to have been run.
"""

from __future__ import annotations

import glob
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .pascal_voc import IMAGENET_MEAN, IMAGENET_STD

DEFAULT_ROOT = "data/movi/frames"
MIN_INSTANCE_PX = 200      # matches the VOC connected-component floor


class MOVi(Dataset):
    def __init__(self, split: str = "train", image_size: int = 336,
                 root: str = DEFAULT_ROOT, augment: bool = False,
                 return_masks: bool = False) -> None:
        d = os.path.join(root, split)
        self.frames = sorted(f for f in glob.glob(os.path.join(d, "*.png"))
                             if not f.endswith("_seg.png"))
        if not self.frames:
            raise FileNotFoundError(
                f"no frames under {d} — run scripts/prepare_movi.py --split {split}")
        idx_path = os.path.join(d, "index.json")
        self.categories = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
        self.image_size, self.augment, self.return_masks = image_size, augment, return_masks
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> dict:
        path = self.frames[i]
        flip = self.augment and random.random() < 0.5
        img = Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        out = {"pixel_values": (arr - self._mean) / self._std}

        if self.return_masks:
            seg = Image.open(path[:-4] + "_seg.png")
            seg = seg.resize((self.image_size, self.image_size), Image.NEAREST)
            if flip:
                seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
            inst = np.asarray(seg, dtype=np.int64).copy()
            if inst.ndim == 3:
                inst = inst[..., 0]
            ids, counts = np.unique(inst[inst != 0], return_counts=True)
            for j, c in zip(ids, counts):
                if c < MIN_INSTANCE_PX:
                    inst[inst == j] = 0
            # semantic map: instance k carries ShapeNet category categories[k-1]
            cats = self.categories.get(os.path.basename(path).split("_")[0], [])
            sem = np.zeros_like(inst)
            for j in np.unique(inst[inst != 0]):
                sem[inst == j] = (cats[j - 1] + 1) if j - 1 < len(cats) else 1
            out["label_full"] = torch.from_numpy(sem)
            out["inst_full"] = torch.from_numpy(inst)
        return out


def build_movi_loader(split: str = "train", batch_size: int = 32, image_size: int = 336,
                      num_workers: int = 8, root: str = DEFAULT_ROOT,
                      shuffle: bool | None = None, augment: bool | None = None,
                      return_masks: bool | None = None) -> DataLoader:
    is_train = split == "train"
    if augment is None:
        augment = is_train
    if return_masks is None:
        return_masks = not is_train
    if shuffle is None:
        shuffle = is_train
    ds = MOVi(split=split, image_size=image_size, root=root,
              augment=augment, return_masks=return_masks)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=is_train, persistent_workers=num_workers > 0)
