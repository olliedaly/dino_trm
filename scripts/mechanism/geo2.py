"""Slot geometry + object-pair sharing for a run under several variants.

variants
  final       plain argmax (for T=8 runs also: step0 = the first recursion step)
  lam{x}      sizes evened: argmax(log m_k - x log share_k)
  loc{b}      made local: argmax(log m_k - b d_k^2), d_k = distance (image widths) to the
              centroid of slot k's main piece
per image: max_share, area_cv, pieces, spread (radius of gyration / sqrt(area), area-
weighted over slots), voronoi (pixels whose slot owns the nearest slot centroid), fg_ari,
mbo_i. per object pair (same order as pd_*.npy): share = P(pixels of the two share a slot).
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as cc_label

sys.path.insert(0, "scripts")
from slot_shape import ari, best_iou, EIGHT, PATCH  # noqa: E402

C, RUN, VARIANTS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
S = 336
torch.set_num_threads(1)
try:
    MS = np.load(f"{C}/m_{RUN}_steps.npy", mmap_mode="r")
except FileNotFoundError:
    MS = None
M = np.load(f"{C}/m_{RUN}.npy") if MS is None else None
GT = np.load(f"{C}/gt.npz")
INST = GT["inst"]
YY, XX = np.mgrid[0:S, 0:S] / S


def geometry(p, K):
    area = np.bincount(p.ravel(), minlength=K).astype(float)
    sh = area[area > 0] / p.size
    n_big, spread, cents, ks = [], [], [], []
    for k in np.flatnonzero(area):
        m = p == k
        comp, _ = cc_label(m, structure=EIGHT)
        n_big.append((np.bincount(comp.ravel())[1:] >= PATCH).sum())
        y, x = YY[m], XX[m]
        spread.append(np.sqrt(y.var() + x.var()) / np.sqrt(m.mean()))
        cents.append((y.mean(), x.mean()))
        ks.append(k)
    cents = np.array(cents)
    d = (YY[None] - cents[:, 0, None, None]) ** 2 + (XX[None] - cents[:, 1, None, None]) ** 2
    vor = np.array(ks)[d.argmin(0)]
    return dict(max_share=sh.max(), area_cv=sh.std() / sh.mean(), pieces=np.mean(n_big),
                spread=np.average(spread, weights=sh), voronoi=(vor == p).mean())


def main_piece_d2(p, K):
    d2 = np.zeros((K, S, S))
    for k in range(K):
        m = p == k
        if not m.any():
            continue
        comp, _ = cc_label(m, structure=EIGHT)
        big = comp == np.bincount(comp.ravel())[1:].argmax() + 1
        d2[k] = (YY - YY[big].mean()) ** 2 + (XX - XX[big].mean()) ** 2
    return d2


def one(i):
    inst = INST[i].astype(np.int64)
    fg = inst > 0
    u, gi = np.unique(inst[fg], return_inverse=True)
    a, b = np.triu_indices(len(u), 1)
    out = {}
    for v in VARIANTS:
        m = (MS[i, 0] if v == "step0" else MS[i, -1]) if MS is not None else M[i]
        t = torch.from_numpy(np.array(m, dtype=np.float32))[None]
        logm = torch.log(F.interpolate(t, size=(S, S), mode="bilinear",
                                       align_corners=False)[0].clamp_min(1e-8)).numpy()
        K = logm.shape[0]
        p = logm.argmax(0)
        if v.startswith("lam"):
            share = np.bincount(p.ravel(), minlength=K) / p.size
            p = (logm - float(v[3:]) * np.log(np.maximum(share, 1e-3))[:, None, None]).argmax(0)
        elif v.startswith("loc"):
            p = (logm - float(v[3:]) * main_piece_d2(p, K)).argmax(0)
        r = geometry(p, K)
        iou, _, Ci = best_iou(p, inst, K)
        r["fg_ari"] = ari(Ci) if Ci.sum() >= 2 else np.nan
        r["mbo_i"] = iou.mean() if len(iou) else np.nan
        if len(u) >= 2:
            n = np.bincount(gi * K + p[fg], minlength=len(u) * K).reshape(len(u), K).astype(float)
            P = n / n.sum(1, keepdims=True)
            r["pairs"] = (P @ P.T)[a, b]
        else:
            r["pairs"] = np.zeros(0)
        out[v] = r
    return out


if __name__ == "__main__":
    with Pool(11) as pool:
        rows = pool.map(one, range(len(INST)), chunksize=8)
    save = {}
    for v in VARIANTS:
        for k in rows[0][v]:
            if k == "pairs":
                save[f"{v}|pairs"] = np.concatenate([r[v]["pairs"] for r in rows]).astype(np.float32)
            else:
                save[f"{v}|{k}"] = np.array([r[v][k] for r in rows], np.float32)
    np.savez_compressed(f"{C}/g_{RUN}.npz", **save)
    print(RUN, "done", flush=True)
