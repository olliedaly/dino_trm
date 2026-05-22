"""Object-centric segmentation metrics: FG-ARI and mean Best Overlap (mBO).

All metrics operate at the ViT patch-grid resolution (each patch token has one GT
label from the downsampled VOC mask). Predicted segmentation is the hard per-patch
assignment ``argmax_k`` of the decoder alpha masks.

FG-ARI: Adjusted Rand Index between predicted and GT clusterings over *foreground*
patches only (background label 0 and void 255 excluded) — the standard DINOSAUR
metric. mBO: for each GT segment, the best IoU against any predicted segment,
averaged. With VOC semantic masks this is mBO_c; mBO_i needs instance masks.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from .data.pascal_voc import VOID_LABEL


def _hard_pred(masks: torch.Tensor) -> np.ndarray:
    """(B, K, N) soft masks -> (B, N) int predicted segment ids."""
    return masks.argmax(dim=1).cpu().numpy()


def fg_ari_image(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """ARI over foreground patches (exclude background 0 and void). None if <2 fg."""
    fg = (gt != 0) & (gt != VOID_LABEL)
    if fg.sum() < 2 or len(np.unique(gt[fg])) < 1:
        return None
    return float(adjusted_rand_score(gt[fg], pred[fg]))


def mbo_image(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """Mean best IoU over GT segments (exclude background and void)."""
    gt_ids = [g for g in np.unique(gt) if g not in (0, VOID_LABEL)]
    if not gt_ids:
        return None
    pred_ids = np.unique(pred)
    ious = []
    for g in gt_ids:
        gmask = gt == g
        best = 0.0
        for p in pred_ids:
            pmask = pred == p
            inter = np.logical_and(gmask, pmask).sum()
            union = np.logical_or(gmask, pmask).sum()
            if union > 0:
                best = max(best, inter / union)
        ious.append(best)
    return float(np.mean(ious))


def _aggregate(values: list[float | None]) -> float:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else 0.0


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int | None = None) -> dict:
    """Returns {'fg_ari', 'mbo'} on the final step, plus per-step lists for the
    metric-vs-recursion-depth plot."""
    model.eval()
    fg_final: list[float | None] = []
    mbo_final: list[float | None] = []
    per_step_ari: list[list[float | None]] = None  # filled once n_steps known

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        px = batch["pixel_values"].to(device)
        gt = batch["label"].view(batch["label"].shape[0], -1).numpy()  # (B, N)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(px)
        masks_list = out["masks_list"]  # list over steps of (B, K, N)

        if per_step_ari is None:
            per_step_ari = [[] for _ in masks_list]
        for si, masks in enumerate(masks_list):
            pred = _hard_pred(masks)
            for b in range(pred.shape[0]):
                per_step_ari[si].append(fg_ari_image(pred[b], gt[b]))

        final_pred = _hard_pred(masks_list[-1])
        for b in range(final_pred.shape[0]):
            fg_final.append(fg_ari_image(final_pred[b], gt[b]))
            mbo_final.append(mbo_image(final_pred[b], gt[b]))

    return {
        "fg_ari": _aggregate(fg_final),
        "mbo": _aggregate(mbo_final),
        "fg_ari_per_step": [_aggregate(s) for s in (per_step_ari or [])],
    }
