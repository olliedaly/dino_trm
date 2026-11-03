"""Blind (no-GT) contiguity / size interventions on cached final-step masks.

variants per image:
  s{sig}_l{lam}  Gaussian-smooth the low-res masks (sigma in patches), upsample, then add the
                 slot-size bias -lam*log(share_k) and argmax
  absorb         every non-main piece of a slot is handed to the next-best slot (iterated):
                 all slots contiguous, no extra segments
  split          every piece its own segment
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, label as cc_label

sys.path.insert(0, "scripts")
from slot_shape import ari, best_iou, pieces, EIGHT  # noqa: E402

C = sys.argv[1]
RUN = sys.argv[2]
SIGS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
LAMS = (0.0, 0.2, 0.3, 0.4)
S = 336
torch.set_num_threads(1)
M = np.load(f"{C}/m_{RUN}.npy")
GT = np.load(f"{C}/gt.npz")
INST, SEM = GT["inst"], GT["sem"]


def metrics(pred, inst, sem, n):
    iou_i, _, Ci = best_iou(pred, inst, n)
    iou_c, _, _ = best_iou(pred, sem, n)
    K = int(pred.max()) + 1
    nb, off, _, _ = pieces(pred, K)
    share = np.bincount(pred.ravel()).max() / pred.size
    return dict(fg_ari=ari(Ci) if Ci.sum() >= 2 else np.nan,
                mbo_i=iou_i.mean() if len(iou_i) else np.nan,
                mbo_c=iou_c.mean() if len(iou_c) else np.nan,
                pieces=nb, off=off, max_share=share, nseg=len(np.unique(pred)))


def up(m):
    t = torch.from_numpy(np.ascontiguousarray(m))[None].float()
    return torch.log(F.interpolate(t, size=(S, S), mode="bilinear",
                                   align_corners=False)[0].clamp_min(1e-8)).numpy()


def absorb(logm):
    logm = logm.copy()
    K = logm.shape[0]
    for _ in range(20):
        pred = logm.argmax(0)
        moved = False
        for k in range(K):
            comp, n = cc_label(pred == k, structure=EIGHT)
            if n < 2:
                continue
            main = np.bincount(comp.ravel())[1:].argmax() + 1
            stray = (comp > 0) & (comp != main)
            logm[k][stray] = -np.inf
            moved = True
        if not moved:
            break
    return logm.argmax(0)


def one(i):
    inst, sem = INST[i].astype(np.int64), SEM[i].astype(np.int64)
    K = M.shape[1]
    out = {}
    for sig in SIGS:
        m = M[i] if sig == 0 else np.stack([gaussian_filter(x, sig, mode="nearest") for x in M[i]])
        logm = up(m)
        p0 = logm.argmax(0)
        share = np.bincount(p0.ravel(), minlength=K) / p0.size
        for lam in LAMS:
            bias = -lam * np.log(np.maximum(share, 1e-3))
            p = (logm + bias[:, None, None]).argmax(0)
            for k, v in metrics(p, inst, sem, K).items():
                out[f"{k}_s{sig}_l{lam}"] = v
    logm = up(M[i])
    for k, v in metrics(absorb(logm), inst, sem, K).items():
        out[f"{k}_absorb"] = v
    _, _, split, n = pieces(logm.argmax(0), K)
    for k, v in metrics(split, inst, sem, n).items():
        out[f"{k}_split"] = v
    return out


if __name__ == "__main__":
    with Pool(11) as pool:
        rows = pool.map(one, range(len(M)), chunksize=8)
    np.savez_compressed(f"{C}/d_{RUN}.npz",
                        **{k: np.array([r[k] for r in rows], np.float32) for k in rows[0]})
    ref = np.load(f"results/coco/per_image/slot_shape/{RUN}.npz")["fg_ari_orig"]
    mine = np.array([r["fg_ari_s0.0_l0.0"] for r in rows])
    print(RUN, "max |diff| vs slot_shape orig", np.nanmax(np.abs(mine - ref)), flush=True)
