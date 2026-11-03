"""Slot attention's own map per epoch for any curve_sa{tag}.npz files (curve_sa_an.py's
measures: FG-ARI of the map, largest slot's share of the image, slots owning >= 1 patch).
Usage: sa_curve_any.py C file[,file...]   prints one row per run and epoch; adds them to
c/sa_curve_any.npz ({run}|e{epoch}|{fg_ari,max_share,n_used})."""
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

C = sys.argv[1]
S = 336
torch.set_num_threads(1)
A = {}
for f in sys.argv[2].split(","):
    Z = np.load(f"{C}/{f}")
    for k in Z.files:
        A[k.rsplit("|", 1)[0]] = Z[k]
KEYS = sorted(A, key=lambda k: (k.split("|e")[0], int(k.split("|e")[1])))
N = A[KEYS[0]].shape[0]
INST = np.load(f"{C}/gt.npz")["inst"][:N]


def ari_table(n):
    n = n.astype(np.float64)
    c2 = lambda x: (x * (x - 1) / 2).sum()  # noqa: E731
    a, b, t = c2(n.sum(1)), c2(n.sum(0)), c2(n)
    tot = n.sum() * (n.sum() - 1) / 2
    e = a * b / tot
    return (t - e) / (0.5 * (a + b) - e) if (0.5 * (a + b) - e) > 0 else 1.0


def one(i):
    inst = INST[i].astype(np.int64)
    fg = inst > 0
    _, gi = np.unique(inst[fg], return_inverse=True)
    out = {}
    for k in KEYS:
        a = A[k][i].astype(np.float32)
        K = a.shape[0]
        tok = a.reshape(K, -1).argmax(0)
        out[f"{k}|max_share"] = np.bincount(tok, minlength=K).max() / tok.size
        out[f"{k}|n_used"] = (np.bincount(tok, minlength=K) > 0).sum()
        if fg.sum() > 1:
            p = F.interpolate(torch.from_numpy(a)[None], size=(S, S), mode="bilinear", align_corners=False)[0].argmax(0).numpy()
            G = gi.max() + 1
            out[f"{k}|fg_ari"] = ari_table(np.bincount(gi * K + p[fg], minlength=G * K).reshape(G, K))
    return out


if __name__ == "__main__":
    with Pool(8) as pool:
        rows = pool.map(one, range(N), chunksize=4)
    out = dict(np.load(f"{C}/sa_curve_any.npz")) if os.path.exists(f"{C}/sa_curve_any.npz") else {}
    print(f"slot attention's own map, {N} val images: FG-ARI | largest slot | slots used")
    for k in KEYS:
        v = [np.nanmean([r.get(f"{k}|{x}", np.nan) for r in rows]) for x in ("fg_ari", "max_share", "n_used")]
        out.update({f"{k}|{x}": val for x, val in zip(("fg_ari", "max_share", "n_used"), v)})
        print(f"  {k:22s} {100 * v[0]:5.1f} | {v[1]:.2f} | {v[2]:.1f}")
    np.savez(f"{C}/sa_curve_any.npz", **out)
