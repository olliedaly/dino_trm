"""Does VOC's ground-truth derivation explain why the size effect vanishes there?

VOC has no instance annotations, so ``evaluate_protocol`` derives instances with
``semantic_to_instance`` — connected components of the semantic mask, min 200px. That
separates same-class objects when they are spatially disjoint but MERGES them when
they touch, because touching objects form one connected region.

If the recursion's gain on COCO depended on separating touching same-class objects,
VOC's ground truth could not see it, and the flat VOC result would be a measurement
artifact rather than a domain difference.

THIS SCRIPT TESTS IT DIRECTLY: re-score COCO with VOC's GT derivation. Same images,
same models, same predictions — only the ground truth changes.

    fg_ari_true : vs COCO's true per-object annotations        (what we report)
    fg_ari_cc   : vs semantic_to_instance(COCO semantic map)   (VOC's derivation)

and both size covariates, since merging changes object size too:

    size_true   : mean visible area of a true instance
    size_cc     : mean visible area of a CC-derived instance

READING THE RESULT
    rho(cc) collapses toward VOC's -0.06  -> the VOC null is a GT artifact; the COCO
                                            finding stands and VOC simply cannot
                                            measure it.
    rho(cc) stays near -0.31              -> the merged-instance explanation is dead
                                            and the COCO/VOC gap is something else.

Run:
    uv run python scripts/coco_voc_style_gt.py
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import re

import numpy as np
import torch

from dino_trm.eval_protocol import _fg_ari, _upsample_pred, semantic_to_instance
from dino_trm.models.full_model import DinoSlotModel

DEVICE = "cuda"
OUT = "results/coco/per_image/voc_style_gt.npz"
RUNS = ["trm_t1", "trm_t1_seed1", "trm_t1_seed2", "trm", "trm_seed1", "trm_seed2"]


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def latest(run: str) -> str | None:
    p = glob.glob(os.path.join("checkpoints/coco", f"{run}_epoch*.pt"))
    return max(p, key=lambda x: int(re.search(r"epoch(\d+)", x).group(1))) if p else None


@torch.no_grad()
def score(model, loader, grid: int, size: int, gts: list) -> dict:
    """gts is the pre-computed [(inst_true, inst_cc), ...] list, image order."""
    model.eval()
    true_, cc_ = [], []
    i = 0
    for batch in loader:
        px = batch["pixel_values"].to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(px)
        final = out["masks_list"][-1]
        for b in range(px.shape[0]):
            pred = _upsample_pred(final[b].view(-1, grid, grid), size)
            it, ic = gts[i]; i += 1
            true_.append(_fg_ari(pred, it))
            cc_.append(_fg_ari(pred, ic))
    nan = lambda v: np.array([np.nan if x is None else x for x in v], dtype=np.float64)
    return {"true": nan(true_), "cc": nan(cc_)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    from dino_trm.data.coco import build_coco_loader
    loader = build_coco_loader(split="val", batch_size=args.batch_size, num_workers=4,
                               shuffle=False, return_masks=True)

    # Ground truths and size covariates are model-independent: build once.
    print("building both ground-truth variants ...")
    gts, size_true, size_cc, n_true, n_cc = [], [], [], [], []
    for batch in loader:
        for b in range(batch["label_full"].shape[0]):
            sem = batch["label_full"][b].numpy().copy()
            sem[sem == 255] = 0
            it = batch["inst_full"][b].numpy()
            ic = semantic_to_instance(sem)
            gts.append((it, ic))
            for m, sz, nn in ((it, size_true, n_true), (ic, size_cc, n_cc)):
                _, c = np.unique(m[m != 0], return_counts=True)
                sz.append(float(c.mean()) if c.size else 0.0)
                nn.append(int(c.size))
    store = {"size_true": np.array(size_true), "size_cc": np.array(size_cc),
             "n_true": np.array(n_true), "n_cc": np.array(n_cc)}
    print(f"  {len(gts)} images. mean instances: true {store['n_true'].mean():.2f}  "
          f"cc {store['n_cc'].mean():.2f}   "
          f"(cc/true = {store['n_cc'].mean()/store['n_true'].mean():.2f})")
    print(f"  mean object size (patches): true {store['size_true'].mean()/256:.1f}  "
          f"cc {store['size_cc'].mean()/256:.1f}")

    for run in RUNS:
        ck = latest(run)
        if not ck:
            print(f"{run}: NO CHECKPOINT"); continue
        c = torch.load(ck, map_location=DEVICE, weights_only=False)
        model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).to(DEVICE)
        model.load_state_dict(c["model"])
        for s in range(args.seeds):
            set_seed(s)
            m = score(model, loader, model.backbone.grid_size,
                      model.backbone.image_size, gts)
            store[f"{run}_true_s{s}"] = m["true"]
            store[f"{run}_cc_s{s}"] = m["cc"]
        print(f"{run}: FG-ARI true={100*np.nanmean([store[f'{run}_true_s{s}'] for s in range(args.seeds)]):.2f}"
              f"  cc={100*np.nanmean([store[f'{run}_cc_s{s}'] for s in range(args.seeds)]):.2f}")

    np.savez(OUT, **store)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
