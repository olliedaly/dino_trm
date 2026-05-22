"""Curate and download a multi-object COCO 2017 subset for occlusion-rich training.

Clean VOC under-tests the coupled feedback loop (mostly single-object scenes). COCO
is the real-world, heavily occluded benchmark with a published DINOSAUR baseline on
*our* setup (frozen DINO -> slot attention -> MLP feature-reconstruction decoder). We
don't need all 118k train images: we keep only **multi-object** images (the regime
where occlusion + relational reasoning matter) and download just those JPGs, so the
footprint stays a few GB instead of 18 GB.

Pipeline:
  1. download + extract the instance annotations (instances_{train,val}2017.json);
  2. select images with >= ``min_instances`` non-crowd instances each at least
     ``min_area_frac`` of the image area (drops tiny specks), capped at N (seeded);
  3. download only the selected JPGs (concurrent, resumable) into data/coco/{split}2017;
  4. save the selected image-id lists to data/coco/{split}_subset_ids.npy.

Run:
    uv run python scripts/build_coco_subset.py                 # defaults: 25k train / 2k val
    uv run python scripts/build_coco_subset.py --n-train 30000 --workers 48
"""

from __future__ import annotations

import argparse
import os
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_file(url: str, dst: str, retries: int = 5) -> bool:
    """Download ``url`` -> ``dst`` (atomic via .part), skipping if it already exists."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    tmp = dst + ".part"
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dst)
            return True
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == retries - 1:
                return False
    return False


def ensure_annotations(root: str) -> None:
    ann_dir = os.path.join(root, "annotations")
    need = [
        os.path.join(ann_dir, "instances_train2017.json"),
        os.path.join(ann_dir, "instances_val2017.json"),
    ]
    if all(os.path.exists(p) for p in need):
        return
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "annotations_trainval2017.zip")
    print(f"downloading annotations -> {zip_path} (~250 MB)")
    if not download_file(ANN_URL, zip_path):
        raise RuntimeError(f"failed to download {ANN_URL}")
    print("extracting instance annotations ...")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith(("instances_train2017.json", "instances_val2017.json")):
                zf.extract(member, root)
    os.remove(zip_path)


def select_subset(
    coco: COCO, min_instances: int, min_area_frac: float, n_cap: int, seed: int
) -> list[int]:
    """Image ids with >= min_instances non-crowd instances each >= min_area_frac of
    the image area. Shuffled with ``seed`` and capped at ``n_cap``."""
    keep: list[int] = []
    for img_id in coco.getImgIds():
        meta = coco.loadImgs(img_id)[0]
        img_area = meta["height"] * meta["width"]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
        big = sum(1 for a in anns if a["area"] >= min_area_frac * img_area)
        if big >= min_instances:
            keep.append(img_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(keep)
    return keep[:n_cap]


def download_split(coco: COCO, ids: list[int], split: str, root: str, workers: int) -> int:
    img_dir = os.path.join(root, f"{split}2017")
    os.makedirs(img_dir, exist_ok=True)
    metas = coco.loadImgs(ids)
    jobs = [
        (m["coco_url"], os.path.join(img_dir, m["file_name"]))
        for m in metas
    ]
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_file, url, dst): dst for url, dst in jobs}
        for f in tqdm(as_completed(futs), total=len(futs), desc=f"{split} imgs"):
            ok += int(f.result())
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/coco")
    ap.add_argument("--n-train", type=int, default=25000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--min-instances", type=int, default=3)
    ap.add_argument("--min-area-frac", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ensure_annotations(args.root)

    for split, n_cap in (("train", args.n_train), ("val", args.n_val)):
        ann = os.path.join(args.root, "annotations", f"instances_{split}2017.json")
        coco = COCO(ann)
        ids = select_subset(
            coco, args.min_instances, args.min_area_frac, n_cap, args.seed
        )
        print(f"{split}: selected {len(ids)} multi-object images "
              f"(>= {args.min_instances} instances >= {args.min_area_frac:.1%} area)")
        ok = download_split(coco, ids, split, args.root, args.workers)
        print(f"{split}: downloaded {ok}/{len(ids)} images")
        np.save(os.path.join(args.root, f"{split}_subset_ids.npy"), np.array(ids))

    print("done.")


if __name__ == "__main__":
    main()
