"""Published-protocol object-discovery eval (matches the DINOSAUR reference repo).

Differences from the coarse-grid eval in ``eval.py`` (which is fine for the
metric-vs-recursion-depth trend but not comparable to published numbers):

  * Predicted slot masks are upsampled to **image resolution** and the metric is
    computed there (not at the 14²/21² patch grid).
  * GT **instance** masks are derived from the semantic mask via connected components
    (``semantic_to_instance``), exactly as the reference does — VOC's semantic masks
    don't separate instances, so this is how mBO_i / FG-ARI are obtained.
  * Void (255) is folded into background (0), per the reference.

Reports mBO_i (instance), mBO_c (class), and FG-ARI (over instance foreground), the
three numbers the DINOSAUR papers report for PASCAL VOC.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as cc_label
from sklearn.metrics import adjusted_rand_score


def semantic_to_instance(seg: np.ndarray, min_size: int = 200) -> np.ndarray:
    """Split each semantic class into connected components -> instance id map.

    Background stays 0; components smaller than ``min_size`` pixels are dropped.
    """
    inst = np.zeros_like(seg, dtype=np.int32)
    next_id = 1
    for cls in np.unique(seg):
        if cls == 0:
            continue
        comps, n = cc_label(seg == cls)
        for i in range(1, n + 1):
            if (comps == i).sum() >= min_size:
                inst[comps == i] = next_id
                next_id += 1
    return inst


def _upsample_pred(masks_grid: torch.Tensor, size: int) -> np.ndarray:
    """(K, g, g) soft masks -> (size, size) hard slot-id map at image resolution."""
    k, g, _ = masks_grid.shape
    up = F.interpolate(masks_grid[None].float(), size=(size, size), mode="bilinear",
                       align_corners=False)[0]
    return up.argmax(0).cpu().numpy()


def _best_overlap(pred: np.ndarray, gt_ids: list[int], gt: np.ndarray) -> float | None:
    if not gt_ids:
        return None
    pred_ids = np.unique(pred)
    ious = []
    for g in gt_ids:
        gm = gt == g
        best = 0.0
        for p in pred_ids:
            pm = pred == p
            union = np.logical_or(gm, pm).sum()
            if union:
                best = max(best, np.logical_and(gm, pm).sum() / union)
        ious.append(best)
    return float(np.mean(ious))


def _fg_ari(pred: np.ndarray, inst: np.ndarray) -> float | None:
    fg = inst != 0
    if fg.sum() < 2 or len(np.unique(inst[fg])) < 1:
        return None
    return float(adjusted_rand_score(inst[fg], pred[fg]))


def _agg(vals: list[float | None]) -> float:
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else 0.0


@torch.no_grad()
def evaluate_protocol(model, loader, device, max_batches: int | None = None) -> dict:
    """Full-resolution mBO_i / mBO_c / FG-ARI on the final recursion step, plus
    per-step FG-ARI for the depth plot. ``loader`` must yield ``label_full``."""
    model.eval()
    size = model.backbone.image_size
    grid = model.backbone.grid_size
    mbo_i, mbo_c, fg_ari = [], [], []
    per_step_ari = None

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        px = batch["pixel_values"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(px)
        masks_list = out["masks_list"]
        if per_step_ari is None:
            per_step_ari = [[] for _ in masks_list]

        for b in range(px.shape[0]):
            sem = batch["label_full"][b].numpy().copy()
            sem[sem == 255] = 0                       # void -> background
            inst = semantic_to_instance(sem)
            cls_ids = [c for c in np.unique(sem) if c != 0]
            inst_ids = [i for i in np.unique(inst) if i != 0]

            for si, masks in enumerate(masks_list):
                pred = _upsample_pred(masks[b].view(-1, grid, grid), size)
                per_step_ari[si].append(_fg_ari(pred, inst))

            pred = _upsample_pred(masks_list[-1][b].view(-1, grid, grid), size)
            mbo_i.append(_best_overlap(pred, inst_ids, inst))
            mbo_c.append(_best_overlap(pred, cls_ids, sem))
            fg_ari.append(_fg_ari(pred, inst))

    return {
        "mbo_i": _agg(mbo_i),
        "mbo_c": _agg(mbo_c),
        "fg_ari": _agg(fg_ari),
        "fg_ari_per_step": [_agg(s) for s in (per_step_ari or [])],
    }
