"""Build a leakage-free training index for scaled-up unsupervised training.

Training pool = `ernestchu/voc2012-image-only` (17,125 VOC images, image-only).
We must NOT train on the 1,449 images used for evaluation (`nateraw/pascal-voc-2012`
val split). Since the image-only set has no filenames, we exclude val images by
content hash (32x32 grayscale, exact byte match — both datasets use the original VOC
JPEGs so decoded pixels match).

Saves the kept indices to data/train_index.npy. Run once:
    uv run python scripts/build_train_index.py
"""

from __future__ import annotations

import os

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

TRAIN_ID = "ernestchu/voc2012-image-only"
VAL_ID = "nateraw/pascal-voc-2012"
OUT = "data/train_index.npy"


def img_hash(pil: Image.Image) -> bytes:
    a = np.asarray(pil.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.uint8)
    return a.tobytes()


def main() -> None:
    os.makedirs("data", exist_ok=True)

    val = load_dataset(VAL_ID, split="val")
    val_hashes = {img_hash(ex["image"]) for ex in tqdm(val, desc="hashing val")}
    print(f"val images: {len(val)}  unique hashes: {len(val_hashes)}")

    train = load_dataset(TRAIN_ID, split="train")
    keep = []
    excluded = 0
    for i, ex in enumerate(tqdm(train, desc="scanning train")):
        if img_hash(ex["image"]) in val_hashes:
            excluded += 1
        else:
            keep.append(i)

    keep = np.array(keep, dtype=np.int64)
    np.save(OUT, keep)
    print(f"train pool: {len(train)}  excluded (val overlap): {excluded}  kept: {len(keep)}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
