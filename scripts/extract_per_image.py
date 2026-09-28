"""Dump PER-IMAGE eval metrics + per-image scene covariates for the COCO val subset.

``eval_published.py`` aggregates to a single mean per run, which cannot support any
stratified or paired analysis. This script keeps the per-image vectors so the
crowding analysis can pair images across runs and control for scene properties.

Metrics exactly mirror ``eval_protocol.evaluate_protocol`` (full-res upsample, true
COCO instance masks, void->background, FG-ARI restricted to inst != 0) — verified by
``--check``, which re-aggregates the per-image vectors and compares against the
stored summary_protocol.json means.

Covariates are computed ONCE (they don't depend on the model) inside the eval crop,
so an object cropped out of the 336px centre crop counts for nothing — matching what
the metric actually sees.

Run:
    uv run python scripts/extract_per_image.py --runs trm_t1,trm,mlp_block,baseline_pm
    uv run python scripts/extract_per_image.py --covariates-only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

import numpy as np
import torch
from PIL import Image

from dino_trm.eval_protocol import (_best_overlap, _fg_ari, _upsample_pred,
                                    semantic_to_instance)
from dino_trm.models.full_model import DinoSlotModel

DEVICE = "cuda"
# per-dataset (checkpoint dir, results dir), mirroring scripts/eval_published.py
DIRS = {"coco": ("checkpoints/coco", "results/coco"), "voc": ("checkpoints", "results"),
        "movi": ("checkpoints/movi", "results/movi")}


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def latest_ckpt(ckpt_dir: str, run: str) -> str | None:
    paths = glob.glob(os.path.join(ckpt_dir, f"{run}_epoch*.pt"))
    return max(paths, key=lambda p: int(re.search(r"epoch(\d+)", p).group(1))) if paths else None


# --------------------------------------------------------------------------- #
# covariates
# --------------------------------------------------------------------------- #
def extract_covariates(root: str, image_size: int) -> dict:
    """Per-image scene descriptors, computed inside the eval centre crop.

    occlusion_j = 1 - (visible area of object j in the final paint-over map)
                    / (area of object j rendered alone)
    both measured after the same resize+crop the eval pipeline applies, so crop
    losses cancel and only the annotation-derived paint-over remains. ``_build_masks``
    paints largest-first, so a smaller object landing on top of a larger one is
    exactly the occlusion signal.
    """
    from dino_trm.data.coco import _resize_crop
    from pycocotools.coco import COCO

    coco = COCO(os.path.join(root, "annotations", "instances_val2017.json"))
    ids = np.load(os.path.join(root, "val_subset_ids.npy")).tolist()

    # A DINOv3-S/16 patch at 336px covers 16x16 = 256 px, and the eval upsamples the
    # 21x21 slot grid back to full res — so an object below ~1 patch is close to
    # unrepresentable. n_tiny / n_small count those, to separate "many objects" from
    # "objects too small for the patch grid" (the two are strongly collinear).
    cols = {k: [] for k in ("n_inst", "n_cls", "fg_frac", "mean_obj_size", "med_obj_size",
                            "occ_mean", "occ_area_w", "occ_max", "n_occluded",
                            "n_tiny", "n_small", "frac_tiny", "min_obj_size")}
    for img_id in ids:
        meta = coco.loadImgs(int(img_id))[0]
        h, w = meta["height"], meta["width"]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=int(img_id), iscrowd=False))
        anns = sorted(anns, key=lambda a: -a["area"])

        sem = np.zeros((h, w), dtype=np.int32)
        inst = np.zeros((h, w), dtype=np.int32)
        solo_in_crop, ann_cls = [], []
        next_id = 1
        for a in anns:
            m = coco.annToMask(a).astype(bool)
            if m.shape != (h, w) or not m.any():
                continue
            sem[m] = a["category_id"]
            inst[m] = next_id
            # area of this object alone, after the eval resize+crop
            solo = _resize_crop(Image.fromarray(m.astype(np.int32), mode="I"),
                                image_size, Image.NEAREST)
            solo_in_crop.append(int((np.asarray(solo) > 0).sum()))
            ann_cls.append(a["category_id"])
            next_id += 1

        inst_c = np.asarray(_resize_crop(Image.fromarray(inst, mode="I"),
                                         image_size, Image.NEAREST))
        sem_c = np.asarray(_resize_crop(Image.fromarray(sem, mode="I"),
                                        image_size, Image.NEAREST))

        vis_ids, vis_counts = np.unique(inst_c[inst_c != 0], return_counts=True)
        vis = dict(zip(vis_ids.tolist(), vis_counts.tolist()))

        occs, areas = [], []
        for j, solo_a in enumerate(solo_in_crop, start=1):
            if solo_a <= 0:          # object fell entirely outside the crop
                continue
            v = vis.get(j, 0)
            if v == 0:               # fully hidden by later paint-over, or cropped away
                continue
            occs.append(1.0 - v / solo_a)
            areas.append(solo_a)

        occs_a, areas_a = np.array(occs, dtype=np.float64), np.array(areas, dtype=np.float64)
        n_inst = len(vis)
        cols["n_inst"].append(n_inst)
        cols["n_cls"].append(int(len([c for c in np.unique(sem_c) if c != 0])))
        cols["fg_frac"].append(float((inst_c != 0).mean()))
        sizes = np.array(list(vis.values()), dtype=np.float64)
        cols["mean_obj_size"].append(float(sizes.mean()) if n_inst else 0.0)
        cols["med_obj_size"].append(float(np.median(sizes)) if n_inst else 0.0)
        cols["min_obj_size"].append(float(sizes.min()) if n_inst else 0.0)
        cols["n_tiny"].append(int((sizes < 256).sum()) if n_inst else 0)
        cols["n_small"].append(int((sizes < 1024).sum()) if n_inst else 0)
        cols["frac_tiny"].append(float((sizes < 256).mean()) if n_inst else 0.0)
        cols["occ_mean"].append(float(occs_a.mean()) if occs_a.size else 0.0)
        cols["occ_area_w"].append(float((occs_a * areas_a).sum() / areas_a.sum())
                                  if areas_a.size else 0.0)
        cols["occ_max"].append(float(occs_a.max()) if occs_a.size else 0.0)
        cols["n_occluded"].append(int((occs_a > 0.05).sum()))

    out = {k: np.array(v) for k, v in cols.items()}
    out["img_id"] = np.array(ids)
    return out


# --------------------------------------------------------------------------- #
# per-image metrics
# --------------------------------------------------------------------------- #
def _sem_inst(batch, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Semantic + instance maps for one image, matching evaluate_protocol exactly:
    void->background, true COCO instances when present, else connected components."""
    sem = batch["label_full"][b].numpy().copy()
    sem[sem == 255] = 0
    inst = (batch["inst_full"][b].numpy() if "inst_full" in batch
            else semantic_to_instance(sem))
    return sem, inst


@torch.no_grad()
def per_image_metrics(model, loader, grid: int, size: int) -> dict:
    model.eval()
    mbo_i, mbo_c, fg_ari = [], [], []
    for batch in loader:
        px = batch["pixel_values"].to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(px)
        final = out["masks_list"][-1]
        for b in range(px.shape[0]):
            sem, inst = _sem_inst(batch, b)
            cls_ids = [c for c in np.unique(sem) if c != 0]
            inst_ids = [int(i) for i in np.unique(inst) if i != 0]
            pred = _upsample_pred(final[b].view(-1, grid, grid), size)
            mbo_i.append(_best_overlap(pred, inst_ids, inst))
            mbo_c.append(_best_overlap(pred, cls_ids, sem))
            fg_ari.append(_fg_ari(pred, inst))
    nan = lambda v: np.array([np.nan if x is None else x for x in v], dtype=np.float64)
    return {"mbo_i": nan(mbo_i), "mbo_c": nan(mbo_c), "fg_ari": nan(fg_ari)}


def covariates_from_masks(loader) -> dict:
    """Scene covariates derived from the loader's own GT masks — the only option for
    VOC, whose semantic annotations carry no per-object polygons.

    For COCO this reproduces every field except the occlusion ones, because the COCO
    covariates are already measured on the post-crop paint-over map (``inst_full``),
    which is exactly what this reads. Occlusion needs the individual annotation
    polygons and so stays COCO-only.
    """
    cols = {k: [] for k in ("n_inst", "n_cls", "fg_frac", "mean_obj_size",
                            "med_obj_size", "min_obj_size", "n_tiny", "n_small",
                            "frac_tiny")}
    for batch in loader:
        for b in range(batch["label_full"].shape[0]):
            sem, inst = _sem_inst(batch, b)
            ids, counts = np.unique(inst[inst != 0], return_counts=True)
            sizes = counts.astype(np.float64)
            n = len(ids)
            cols["n_inst"].append(n)
            cols["n_cls"].append(int(len([c for c in np.unique(sem) if c != 0])))
            cols["fg_frac"].append(float((inst != 0).mean()))
            cols["mean_obj_size"].append(float(sizes.mean()) if n else 0.0)
            cols["med_obj_size"].append(float(np.median(sizes)) if n else 0.0)
            cols["min_obj_size"].append(float(sizes.min()) if n else 0.0)
            cols["n_tiny"].append(int((sizes < 256).sum()) if n else 0)
            cols["n_small"].append(int((sizes < 1024).sum()) if n else 0)
            cols["frac_tiny"].append(float((sizes < 256).mean()) if n else 0.0)
    return {k: np.array(v) for k, v in cols.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["coco", "voc"], default="coco")
    ap.add_argument("--runs", type=str, default="trm_t1,trm,mlp_block,baseline_pm")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--coco-root", type=str, default="data/coco")
    ap.add_argument("--covariates-only", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="re-aggregate and compare against summary_protocol.json")
    args = ap.parse_args()

    ckpt_dir, results_dir = DIRS[args.dataset]
    out_dir = os.path.join(results_dir, "per_image")
    os.makedirs(out_dir, exist_ok=True)
    cov_path = os.path.join(out_dir, "covariates.npz")

    if args.dataset == "coco":
        loader = None  # built lazily below; COCO covariates come from annotations
    else:
        from dino_trm.data.pascal_voc import build_loader
        loader = build_loader(split="val", batch_size=args.batch_size, num_workers=4,
                              shuffle=False, full_mask=True)

    if not os.path.exists(cov_path):
        print("extracting covariates ...")
        if args.dataset == "coco":
            cov = extract_covariates(args.coco_root, 336)
        else:
            cov = covariates_from_masks(loader)
        np.savez(cov_path, **cov)
        print(f"wrote {cov_path}  n={len(cov['n_inst'])}  "
              f"median n_inst={np.median(cov['n_inst']):.1f}  "
              f"median obj size={np.median(cov['mean_obj_size'])/256:.1f} patches"
              + (f"  median occ={np.median(cov['occ_mean']):.4f}" if "occ_mean" in cov else ""))
    else:
        print(f"covariates cached at {cov_path}")
    if args.covariates_only:
        return

    if loader is None:
        from dino_trm.data.coco import build_coco_loader
        loader = build_coco_loader(split="val", batch_size=args.batch_size,
                                   num_workers=4, shuffle=False, return_masks=True)
    print(f"dataset={args.dataset}  val images = {len(loader.dataset)}")

    summary = json.load(open(os.path.join(results_dir, "summary_protocol.json")))["ours"]
    for run in args.runs.split(","):
        out_path = os.path.join(out_dir, f"{run}.npz")
        if os.path.exists(out_path) and not args.check:
            print(f"{run}: cached")
            continue
        ck = latest_ckpt(ckpt_dir, run)
        if not ck:
            print(f"{run}: NO CHECKPOINT")
            continue
        c = torch.load(ck, map_location=DEVICE, weights_only=False)
        model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).to(DEVICE)
        model.load_state_dict(c["model"])
        grid, size = model.backbone.grid_size, model.backbone.image_size

        store = {}
        for s in range(args.seeds):
            set_seed(s)
            m = per_image_metrics(model, loader, grid, size)
            for k, v in m.items():
                store[f"{k}_s{s}"] = v
        np.savez(out_path, **store)

        # cross-check: mean over per-image vectors must match the stored summary
        msg = []
        for k in ("mbo_i", "mbo_c", "fg_ari"):
            mine = np.mean([np.nanmean(store[f"{k}_s{s}"]) for s in range(args.seeds)])
            ref = summary.get(run, {}).get(k)
            ok = "" if ref is None else f" (summary {100*ref:.2f}, d={100*(mine-ref):+.3f})"
            msg.append(f"{k}={100*mine:.2f}{ok}")
        print(f"{run}: " + "  ".join(msg))
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
