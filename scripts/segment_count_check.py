"""Is the T=1 -> T=8 size effect just "more segments on the foreground"?

FG-ARI rewards finer partitions on images with many small objects and penalises them
on images with a few large ones. So if recursion simply put more slots on the
foreground in cluttered images, the size effect would follow trivially. This re-runs a
T=1 / T=8 checkpoint pair (eval seed 0, the same predictions as fg_ari_s0 in
results/coco/per_image), counts the predicted segments holding >= 2% of each image's
foreground, and checks whether the size trend survives among images where both models
produce the SAME number of foreground segments.

    uv run python scripts/segment_count_check.py --pair trm_t1:trm
    uv run python scripts/segment_count_check.py --pair trm_t1_seed1:trm_seed1

Needs checkpoints/coco and the DINOv3 weights.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from extract_per_image import latest_ckpt, set_seed  # noqa: E402

from dino_trm.data.coco import build_coco_loader  # noqa: E402
from dino_trm.eval_protocol import _fg_ari, _upsample_pred  # noqa: E402
from dino_trm.models.full_model import DinoSlotModel  # noqa: E402


@torch.no_grad()
def run(name: str, loader) -> tuple[np.ndarray, np.ndarray]:
    c = torch.load(latest_ckpt("checkpoints/coco", name), map_location="cuda",
                   weights_only=False)
    model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).cuda()
    model.load_state_dict(c["model"])
    model.eval()
    g, size = model.backbone.grid_size, model.backbone.image_size
    set_seed(0)
    ari, nseg = [], []
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch["pixel_values"].cuda())
        for b in range(batch["pixel_values"].shape[0]):
            inst = batch["inst_full"][b].numpy()
            fg = inst != 0
            pred = _upsample_pred(out["masks_list"][-1][b].view(-1, g, g), size)
            r = _fg_ari(pred, inst)
            ari.append(np.nan if r is None else r)
            _, cnt = np.unique(pred[fg], return_counts=True)
            nseg.append(int((cnt >= 0.02 * fg.sum()).sum()))
    return np.array(ari), np.array(nseg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="trm_t1:trm")
    a, b = ap.parse_args().pair.split(":")
    loader = build_coco_loader(split="val", batch_size=16, num_workers=4, shuffle=False,
                               return_masks=True)
    (fa, na), (fb, nb) = run(a, loader), run(b, loader)
    stored = np.load(f"results/coco/per_image/{a}.npz")["fg_ari_s0"]
    print(f"cross-check vs stored per-image FG-ARI ({a}, eval seed 0): "
          f"max |diff| = {np.nanmax(np.abs(stored - fa)):.2e}")

    cov = np.load("results/coco/per_image/covariates.npz")
    size = cov["mean_obj_size"] / 256.0
    qi = np.searchsorted(np.quantile(size, [0.2, 0.4, 0.6, 0.8]), size)
    d, dn = 100 * (fb - fa), nb - na
    print(f"\nforeground segments per image: {a} {na.mean():.2f}, {b} {nb.mean():.2f}")
    print("Δ segments by size quintile: " + "  ".join(
        f"Q{k+1} {dn[qi == k].mean():+.2f}" for k in range(5)))
    print(f"rho(size, ΔFG-ARI), all images: {spearmanr(size, d, nan_policy='omit')[0]:+.3f}")
    for v in (-1, 0, 1):
        m = dn == v
        print(f"  images with Δsegments = {v:+d} (n={m.sum():4d}): rho "
              f"{spearmanr(size[m], d[m], nan_policy='omit')[0]:+.3f}   "
              f"ΔFG-ARI Q1 {np.nanmean(d[m & (qi == 0)]):+.2f}  "
              f"Q5 {np.nanmean(d[m & (qi == 4)]):+.2f}")


if __name__ == "__main__":
    main()
