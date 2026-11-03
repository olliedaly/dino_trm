"""Pair rates behind FG-ARI, per model (eval seed 0, all val images, foreground pixels only).
whole  = share of same-object pixel pairs that share a slot      (w)
same   = share of pairs from two different same-class objects that share a slot
diff   = share of pairs from two different-class objects that share a slot
q      = share of all foreground pairs that share a slot
FG-ARI = p(w - q) / (1/2 (p + q) - p q)."""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

C = sys.argv[1]
NAMES = sys.argv[2].split(",")
S = 336
torch.set_num_threads(1)
GT = np.load(f"{C}/gt.npz")
INST, SEM = GT["inst"], GT["sem"]
M = {n: np.load(f"{C}/m_{n}.npy", mmap_mode="r") for n in NAMES}


def c2(x):
    return (x * (x - 1) / 2).sum()


def one(i):
    inst, sem = INST[i].astype(np.int64), SEM[i].astype(np.int64)
    fg = inst > 0
    ids, gi = np.unique(inst[fg], return_inverse=True)
    if len(ids) < 2:
        return None
    cls = np.array([np.bincount(sem[inst == j]).argmax() for j in ids])
    n_g = np.bincount(gi, minlength=len(ids)).astype(float)
    N = fg.sum()
    T = N * (N - 1) / 2
    A = c2(n_g)
    sc = cls[:, None] == cls[None, :]
    off = ~np.eye(len(ids), dtype=bool)
    pg = np.outer(n_g, n_g)
    tot_same, tot_diff = pg[sc & off].sum() / 2, pg[~sc].sum() / 2
    out = []
    for n in NAMES:
        t = torch.from_numpy(np.array(M[n][i], dtype=np.float32))[None]
        p = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)[0].argmax(0).numpy()
        K = t.shape[1]
        nk = np.bincount(gi * K + p[fg], minlength=len(ids) * K).reshape(len(ids), K).astype(float)
        a = c2(nk)
        B = c2(nk.sum(0))
        cross = nk @ nk.T                     # pixel pairs (g, h) sharing a slot
        same = cross[sc & off].sum() / 2 / tot_same if tot_same else np.nan
        diff = cross[~sc].sum() / 2 / tot_diff if tot_diff else np.nan
        pp, qq, ww = A / T, B / T, a / A
        ari = (pp * ww - pp * qq) / (0.5 * (pp + qq) - pp * qq)
        out.append([ww, same, diff, qq, ari])
    return out


if __name__ == "__main__":
    with Pool(11) as pool:
        R = [r for r in pool.map(one, range(len(INST)), chunksize=8) if r is not None]
    R = np.array(R)  # (images, models, 5)
    print(f"{len(R)} images with >= 2 objects")
    print(f"{'model':24s}{'whole':>8s}{'same':>8s}{'diff':>8s}{'q':>8s}{'FG-ARI':>8s}")
    for j, n in enumerate(NAMES):
        v = np.nanmean(R[:, j], 0)
        print(f"{n:24s}" + "".join(f"{x:8.3f}" for x in v[:4]) + f"{100 * v[4]:8.1f}")
