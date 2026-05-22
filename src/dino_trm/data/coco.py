"""COCO 2017 multi-object loader (occlusion-rich training/eval).

Reads the curated subset produced by ``scripts/build_coco_subset.py`` (only the
downloaded JPGs + the saved id lists). COCO is the real-world, heavily occluded
benchmark with a published DINOSAUR baseline on our exact setup — the regime where
the coupled feedback loop and slot→patch cross-attention should actually pay off,
unlike mostly-single-object VOC.

  * train: unsupervised, returns ``pixel_values`` only (feature-reconstruction
    objective needs no labels), same aspect-preserving crop + h-flip as VOC.
  * val: also returns ``label_full`` (semantic category map) and ``inst_full`` (TRUE
    per-object instance map) at image resolution. Unlike VOC we have real instance
    masks, so ``eval_protocol`` uses ``inst_full`` directly instead of deriving
    instances from connected components.

Only the 80 "thing" categories are present in instances_*2017.json (no stuff), so
foreground == objects, matching the DINOSAUR COCO object-discovery protocol.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

from .pascal_voc import IMAGENET_MEAN, IMAGENET_STD

DEFAULT_ROOT = "data/coco"


def _resize_crop(pil: Image.Image, image_size: int, resample) -> Image.Image:
    """Aspect-preserving: resize shorter side to image_size, then centre crop."""
    w, h = pil.size
    scale = image_size / min(w, h)
    nw, nh = round(w * scale), round(h * scale)
    pil = pil.resize((nw, nh), resample)
    left = (nw - image_size) // 2
    top = (nh - image_size) // 2
    return pil.crop((left, top, left + image_size, top + image_size))


class COCOSubset(Dataset):
    def __init__(
        self,
        split: str = "train",
        image_size: int = 336,
        root: str = DEFAULT_ROOT,
        augment: bool = False,
        return_masks: bool = False,
    ) -> None:
        ids_path = os.path.join(root, f"{split}_subset_ids.npy")
        if not os.path.exists(ids_path):
            raise FileNotFoundError(
                f"{ids_path} missing — run `uv run python scripts/build_coco_subset.py` first."
            )
        ann = os.path.join(root, "annotations", f"instances_{split}2017.json")
        self.coco = COCO(ann)
        self.ids = np.load(ids_path).tolist()
        self.img_dir = os.path.join(root, f"{split}2017")
        self.image_size = image_size
        self.augment = augment
        self.return_masks = return_masks
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.ids)

    def _build_masks(self, img_id: int, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        """Full-res semantic (category) and instance (per-object) maps; 0 = background.

        Larger objects are painted first so smaller (often occluding/foreground)
        objects land on top where masks overlap.
        """
        sem = np.zeros((h, w), dtype=np.int32)
        inst = np.zeros((h, w), dtype=np.int32)
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
        anns = sorted(anns, key=lambda a: -a["area"])
        next_id = 1
        for a in anns:
            m = self.coco.annToMask(a).astype(bool)
            if m.shape != (h, w) or not m.any():
                continue
            sem[m] = a["category_id"]
            inst[m] = next_id
            next_id += 1
        return sem, inst

    def __getitem__(self, i: int) -> dict:
        img_id = int(self.ids[i])
        meta = self.coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(self.img_dir, meta["file_name"])).convert("RGB")
        flip = self.augment and random.random() < 0.5

        img_t = _resize_crop(img, self.image_size, Image.BILINEAR)
        if flip:
            img_t = img_t.transpose(Image.FLIP_LEFT_RIGHT)
        arr = torch.from_numpy(np.asarray(img_t, dtype=np.float32) / 255.0).permute(2, 0, 1)
        out = {"pixel_values": (arr - self._mean) / self._std}

        if self.return_masks:
            sem, inst = self._build_masks(img_id, meta["height"], meta["width"])
            sem_pil = _resize_crop(Image.fromarray(sem, mode="I"), self.image_size, Image.NEAREST)
            inst_pil = _resize_crop(Image.fromarray(inst, mode="I"), self.image_size, Image.NEAREST)
            if flip:
                sem_pil = sem_pil.transpose(Image.FLIP_LEFT_RIGHT)
                inst_pil = inst_pil.transpose(Image.FLIP_LEFT_RIGHT)
            out["label_full"] = torch.from_numpy(np.asarray(sem_pil, dtype=np.int64))
            out["inst_full"] = torch.from_numpy(np.asarray(inst_pil, dtype=np.int64))
        return out


def build_coco_loader(
    split: str = "train",
    batch_size: int = 32,
    image_size: int = 336,
    num_workers: int = 8,
    root: str = DEFAULT_ROOT,
    shuffle: bool | None = None,
    augment: bool | None = None,
    return_masks: bool | None = None,
) -> DataLoader:
    is_train = split == "train"
    if augment is None:
        augment = is_train
    if return_masks is None:
        return_masks = not is_train  # val needs masks for eval; train is unsupervised
    if shuffle is None:
        shuffle = is_train
    ds = COCOSubset(
        split=split,
        image_size=image_size,
        root=root,
        augment=augment,
        return_masks=return_masks,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=num_workers > 0,
    )
