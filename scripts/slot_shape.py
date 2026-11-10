"""What does recursion change about the slots? Slot size, slot pieces, and two interventions.

Re-runs checkpoints at eval seed 0 (the same predictions as ``*_s0`` in
results/coco/per_image) and writes results/coco/per_image/slot_shape/<run>.npz with,
per image:

  * at every recursion step, using no ground truth:
      max_share_t{t}  largest slot's share of the image
      pieces_t{t}     separate (8-connected) pieces per slot, counting pieces >= 1 patch
                      (256 px), averaged over the slots present
      off_main_t{t}   share of slot pixels outside each slot's biggest piece
  * at the final step, FG-ARI / mBO_i / mBO_c of
      *_orig          the prediction as scored by eval_protocol
      *_split         every piece of every slot relabelled as its own segment
      *_lam{x}        (--sweep) argmax(log m_k + b_k), b_k = -x log(share_k): a per-slot
                      bias that evens slot sizes out; share_k is slot k's share of the image
  * per GT instance at the final step: inst_img, inst_area (patches), inst_iou (best IoU)

Metrics come from contingency tables (same formulas as eval_protocol, incl. sklearn's
ARI); the script checks them against the stored per-image FG-ARI.

    uv run python scripts/slot_shape.py --runs trm_t1,trm_t1_seed1,trm_t1_seed2 --sweep
    uv run python scripts/slot_shape.py --runs trm,trm_seed1,trm_seed2,trm_t2,trm_t4

Needs checkpoints/coco and the DINOv3 weights.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as cc_label

sys.path.insert(0, "scripts")
from extract_per_image import latest_ckpt, set_seed  # noqa: E402

from dino_trm.data.coco import build_coco_loader  # noqa: E402
from dino_trm.models.full_model import DinoSlotModel  # noqa: E402

OUT = "results/coco/per_image/slot_shape"
LAMS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
PATCH = 256          # pixels per 16x16 patch
EIGHT = np.ones((3, 3), int)


def ari(C: np.ndarray) -> float:
    """sklearn's adjusted_rand_score from a contingency table (rows GT, cols pred)."""
    C = C[C.sum(1) > 0][:, C.sum(0) > 0].astype(object)   # python ints: no overflow
    n = int(C.sum())
    ss = int((C ** 2).sum())
    fp = int((C @ C.sum(0)).sum()) - ss
    fn = int((C.T @ C.sum(1)).sum()) - ss
    tp, tn = ss - n, n * n - fp - fn - ss
    if fp == 0 and fn == 0:
        return 1.0
    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


def best_iou(pred: np.ndarray, gt: np.ndarray, n_pred: int):
    """Best IoU over predicted segments for each nonzero GT id, plus the GT areas."""
    p = pred.ravel()
    ids, inv = np.unique(gt.ravel(), return_inverse=True)
    C = np.bincount(inv * n_pred + p, minlength=len(ids) * n_pred).reshape(len(ids), n_pred)
    area = C.sum(0).astype(np.float64)
    Cr = C[ids != 0].astype(np.float64)
    g = Cr.sum(1, keepdims=True)
    iou = Cr / np.maximum(g + area[None] - Cr, 1)
    iou[:, area == 0] = 0
    return iou.max(1), g[:, 0], C[ids != 0]


def score(pred, inst, sem, n_pred) -> dict:
    iou_i, _, Ci = best_iou(pred, inst, n_pred)
    iou_c, _, _ = best_iou(pred, sem, n_pred)
    return {"fg_ari": ari(Ci) if Ci.sum() >= 2 else np.nan,
            "mbo_i": iou_i.mean() if len(iou_i) else np.nan,
            "mbo_c": iou_c.mean() if len(iou_c) else np.nan}


def pieces(pred: np.ndarray, K: int):
    """Per-slot piece counts and the prediction with every piece as its own segment."""
    split = np.zeros_like(pred)
    nxt, n_big, off = 0, [], []
    for k in range(K):
        m = pred == k
        if not m.any():
            continue
        comp, n = cc_label(m, structure=EIGHT)
        sizes = np.bincount(comp.ravel())[1:]
        n_big.append((sizes >= PATCH).sum())
        off.append(1 - sizes.max() / sizes.sum())
        split[m] = comp[m] - 1 + nxt
        nxt += n
    return float(np.mean(n_big)), float(np.mean(off)), split, nxt


@torch.no_grad()
def run(name: str, loader, sweep: bool) -> None:
    c = torch.load(latest_ckpt("checkpoints/coco", name), map_location="cuda",
                   weights_only=False)
    model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).cuda()
    model.load_state_dict(c["model"])
    model.eval()
    g, size = model.backbone.grid_size, model.backbone.image_size
    set_seed(0)
    rec: dict[str, list] = {}
    add = lambda k, v: rec.setdefault(k, []).append(v)  # noqa: E731
    i0 = 0
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch["pixel_values"].cuda())
        ml = out["masks_list"]
        T, K = len(ml), ml[-1].shape[1]
        for b in range(ml[-1].shape[0]):
            inst = batch["inst_full"][b].numpy()
            sem = batch["label_full"][b].numpy().copy()
            sem[sem == 255] = 0
            for t in range(T):
                up = F.interpolate(ml[t][b].view(-1, g, g)[None].float(), size=(size, size),
                                   mode="bilinear", align_corners=False)[0]
                logm = torch.log(up.clamp_min(1e-8))
                pred_t = logm.argmax(0)
                share = torch.bincount(pred_t.ravel(), minlength=K).float() / pred_t.numel()
                pred = pred_t.cpu().numpy()
                nb, off, split, n_split = pieces(pred, K)
                add(f"max_share_t{t}", share.max().item())
                add(f"pieces_t{t}", nb)
                add(f"off_main_t{t}", off)
                if t < T - 1:
                    continue
                for k, v in score(pred, inst, sem, K).items():
                    add(f"{k}_orig", v)
                for k, v in score(split, inst, sem, n_split).items():
                    add(f"{k}_split", v)
                iou, area, _ = best_iou(pred, inst, K)
                rec.setdefault("inst_iou", []).extend(iou)
                rec.setdefault("inst_area", []).extend(area / PATCH)
                rec.setdefault("inst_img", []).extend([i0 + b] * len(iou))
                if sweep:
                    for lam in LAMS:
                        bias = -lam * torch.log(share.clamp_min(1e-3))
                        p2_t = (logm + bias[:, None, None]).argmax(0)
                        add(f"max_share_lam{lam:.1f}",
                            (torch.bincount(p2_t.ravel(), minlength=K).max()
                             / p2_t.numel()).item())
                        for k, v in score(p2_t.cpu().numpy(), inst, sem, K).items():
                            add(f"{k}_lam{lam:.1f}", v)
        i0 += ml[-1].shape[0]
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f"{OUT}/{name}.npz",
                        **{k: np.asarray(v, dtype=np.float32) for k, v in rec.items()})
    stored = np.load(f"results/coco/per_image/{name}.npz")["fg_ari_s0"]
    mine = np.asarray(rec["fg_ari_orig"])
    print(f"{name}: FG-ARI {100 * np.nanmean(mine):.2f}, max |diff| vs stored per-image "
          f"FG-ARI (eval seed 0) {np.nanmax(np.abs(mine - stored)):.1e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--sweep", action="store_true", help="also run the slot-size intervention")
    a = ap.parse_args()
    loader = build_coco_loader(split="val", batch_size=16, num_workers=4, shuffle=False,
                               return_masks=True)
    for r in a.runs.split(","):
        run(r, loader, a.sweep)


if __name__ == "__main__":
    main()
