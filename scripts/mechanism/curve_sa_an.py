"""Slot attention's own map over training (c/curve_sa.npz from curve_sa.py): when does it
stop segmenting in T=8? Per epoch, mean over 3 seeds, on the first N val images:
FG-ARI of the map, the largest slot's share of the image (argmax over slots), and slots
owning >= 1 patch."""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

C = sys.argv[1]
S = 336
torch.set_num_threads(1)
Z = np.load(f"{C}/curve_sa.npz")
KEYS = sorted({k.rsplit("|", 1)[0] for k in Z.files})  # "{run}|e{epoch}"
EPOCHS = sorted({int(k.split("|e")[1]) for k in KEYS})
A = {k: Z[f"{k}|attn"] for k in KEYS}
N = A[KEYS[0]].shape[0]
INST = np.load(f"{C}/gt.npz")["inst"][:N]


def ari_table(n):
    n = n.astype(np.float64)
    c2 = lambda x: (x * (x - 1) / 2).sum()  # noqa: E731
    a, b, t = c2(n.sum(1)), c2(n.sum(0)), c2(n)
    tot = n.sum() * (n.sum() - 1) / 2
    e = a * b / tot
    return (t - e) / (0.5 * (a + b) - e) if (0.5 * (a + b) - e) > 0 else 1.0  # sklearn's convention


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
    P = {k: np.array([r.get(k, np.nan) for r in rows]) for k in {k for r in rows for k in r}}
    np.savez(f"{C}/curve_sa_sum.npz", **{f"{k}|{x}": np.nanmean(P[f"{k}|{x}"]) for k in KEYS for x in ("fg_ari", "max_share", "n_used")})
    SEEDS = ["", "_seed1", "_seed2"]
    print(f"slot attention's own map, {N} val images, mean over 3 seeds (per-seed values in brackets for max share)")
    print(f"{'epoch':>5s} | {'FG-ARI T=1':>10s} {'T=8':>6s} | {'max share T=1':>13s} {'T=8':>28s} | {'slots used T=1':>14s} {'T=8':>5s}")
    for e in EPOCHS:
        if not all(f"{r}{s}|e{e}" in KEYS for r in ("trm_t1", "trm") for s in SEEDS):
            continue
        v = {(r, st): [np.nanmean(P[f"{r}{s}|e{e}|{st}"]) for s in SEEDS] for r in ("trm_t1", "trm") for st in ("fg_ari", "max_share", "n_used")}
        print(f"{e:5d} | {100 * np.mean(v['trm_t1', 'fg_ari']):10.1f} {100 * np.mean(v['trm', 'fg_ari']):6.1f} |"
              f" {np.mean(v['trm_t1', 'max_share']):13.2f} {np.mean(v['trm', 'max_share']):6.2f}"
              f" ({' '.join(f'{x:.2f}' for x in v['trm', 'max_share'])}) |"
              f" {np.mean(v['trm_t1', 'n_used']):14.1f} {np.mean(v['trm', 'n_used']):5.1f}")
