"""Does the size effect follow the MODEL or the DATA?

Neither VOC's ground-truth derivation (coco_voc_style_gt.py) nor object count explains
the COCO/VOC gap. What remains splits cleanly in two:

  TRAINING DISTRIBUTION — the COCO subset is curated for >=3 instances, so COCO models
      TRAIN on crowded scenes while VOC models train on mostly-single-object ones. The
      size-specialised behaviour may only develop under crowded training data.
      (Note this is NOT what the earlier slot-competition test ruled out: that
      restricted the COCO EVAL set, leaving training untouched.)

  TEST IMAGES — something about VOC's images or annotations suppresses the effect.

This script separates them by evaluating COCO-TRAINED models on VOC val images:

  effect appears  -> it is a property of the trained model  => training distribution
  effect absent   -> it is a property of the evaluation data => test images

Architectures are identical across the two datasets (6.40M, gnn_cross_attn=true, same
decoder and slot config), because the VOC runs were retrained to the COCO config — so
the same checkpoint class is being moved, not a different model.

CAVEAT: COCO models saw 80 thing-classes, VOC has 20 that largely overlap; the
backbone is frozen DINOv3 and the decoder reconstructs its features, so transfer is
plausible but the absolute numbers will drop. The SHAPE of the size profile is the
signal here, not the level.

Run:  uv run python scripts/transfer_eval.py
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import re

import numpy as np
import torch
from scipy import stats

from dino_trm.eval_protocol import _fg_ari, _upsample_pred, semantic_to_instance
from dino_trm.models.full_model import DinoSlotModel

DEVICE = "cuda"
OUT = "results/per_image/transfer_coco_to_voc.npz"
PAIRS = {"trm_t1": ["trm_t1", "trm_t1_seed1", "trm_t1_seed2"],
         "trm":    ["trm", "trm_seed1", "trm_seed2"]}
EDGES = np.array([8, 16, 32, 64]) * 256
LABELS = ["<8", "8-16", "16-32", "32-64", "64+"]


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def run(model, loader, gts) -> np.ndarray:
    model.eval()
    grid, size = model.backbone.grid_size, model.backbone.image_size
    out, i = [], 0
    for batch in loader:
        px = batch["pixel_values"].to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(px)
        final = o["masks_list"][-1]
        for b in range(px.shape[0]):
            out.append(_fg_ari(_upsample_pred(final[b].view(-1, grid, grid), size), gts[i]))
            i += 1
    return np.array([np.nan if x is None else x for x in out], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    from dino_trm.data.pascal_voc import build_loader
    loader = build_loader(split="val", batch_size=args.batch_size, num_workers=4,
                          shuffle=False, full_mask=True)
    gts = []
    for batch in loader:
        for b in range(batch["label_full"].shape[0]):
            sem = batch["label_full"][b].numpy().copy(); sem[sem == 255] = 0
            gts.append(semantic_to_instance(sem))
    print(f"VOC val: {len(gts)} images")

    store = {}
    for arm, runs in PAIRS.items():
        for r in runs:
            ck = max(glob.glob(f"checkpoints/coco/{r}_epoch*.pt"),
                     key=lambda x: int(re.search(r"epoch(\d+)", x).group(1)))
            c = torch.load(ck, map_location=DEVICE, weights_only=False)
            m = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).to(DEVICE)
            m.load_state_dict(c["model"])
            for s in range(args.seeds):
                set_seed(s)
                store[f"{r}_s{s}"] = run(m, loader, gts)
            print(f"  {r:16s} (COCO-trained) on VOC: FG-ARI="
                  f"{100*np.nanmean([store[f'{r}_s{s}'] for s in range(args.seeds)]):.2f}")
    np.savez(OUT, **store)

    cov = dict(np.load("results/per_image/covariates.npz"))
    size = cov["mean_obj_size"].astype(float)
    b = np.clip(np.searchsorted(EDGES, size, side="right"), 0, 4)
    avg = lambda rs: np.nanmean([store[f"{r}_s{s}"] for r in rs for s in range(args.seeds)], axis=0)
    d = avg(PAIRS["trm"]) - avg(PAIRS["trm_t1"])
    k = ~np.isnan(d) & (size > 0)
    r = stats.spearmanr(d[k], size[k])
    print("\n=== COCO-trained models, VOC images: size profile of T=1 -> T=8 ===")
    print("  " + "  ".join(f"{l}={100*d[k & (b == i)].mean():+6.2f}" for i, l in enumerate(LABELS)))
    print(f"  overall {100*d[k].mean():+.2f}   n={k.sum()}   rho={r.statistic:+.4f}  p={r.pvalue:.1e}")
    print("\n  reference:  COCO-trained on COCO images  rho = -0.302")
    print("              VOC-trained  on VOC images   rho = -0.059")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
