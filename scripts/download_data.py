"""Pre-download PASCAL VOC 2012 into the HuggingFace cache so training starts fast.

Run:  uv run python scripts/download_data.py
"""

from __future__ import annotations

from datasets import load_dataset

from dino_trm.data.pascal_voc import DATASET_ID


def main() -> None:
    for split in ("train", "val"):
        ds = load_dataset(DATASET_ID, split=split)
        print(f"{split}: {len(ds)} examples cached")


if __name__ == "__main__":
    main()
