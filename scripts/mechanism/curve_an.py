"""Learning curves from c/curve.npz (curve.py): is T=8's higher reconstruction error a
training-budget artifact, and is it a fit (train images) or a generalisation (val) gap?
Per epoch, mean over 3 seeds: val and train MSE of T=1 and T=8 (final step), their ratio,
and on the val images the locality (far same-class sharing), whole and FG-ARI."""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

C = sys.argv[1]
S = 336
torch.set_num_threads(1)
Z = np.load(f"{C}/curve.npz")
KEYS = sorted({k.rsplit("|", 2)[0] for k in Z.files})  # "{run}|e{epoch}"
EPOCHS = sorted({int(k.split("|e")[1]) for k in KEYS})
N = Z[f"{KEYS[0]}|val|masks"].shape[0]
GT = np.load(f"{C}/gt.npz")
INST, SEM = GT["inst"][:N], GT["sem"][:N]
PD = np.load(f"{C}/pd_trm.npy")
IMG = PD[:, 0].astype(int)
START = np.r_[0, np.cumsum(np.bincount(IMG, minlength=len(GT["inst"])))]
MASKS = {k: Z[f"{k}|val|masks"] for k in KEYS}


def ari_table(n):
    n = n.astype(np.float64)
    c2 = lambda x: (x * (x - 1) / 2).sum()  # noqa: E731
    a, b, t = c2(n.sum(1)), c2(n.sum(0)), c2(n)
    tot = n.sum() * (n.sum() - 1) / 2
    e = a * b / tot
    return (t - e) / (0.5 * (a + b) - e) if (0.5 * (a + b) - e) > 0 else 1.0  # sklearn's convention


def one(i):
    inst, sem = INST[i].astype(np.int64), SEM[i].astype(np.int64)
    fg = inst > 0
    ids, gi = np.unique(inst[fg], return_inverse=True)
    G = len(ids)
    out = {}
    if not G:
        return out
    cls = np.array([np.bincount(sem[inst == j]).argmax() for j in ids])
    n_g = np.bincount(gi, minlength=G).astype(float)
    a_, b_ = np.triu_indices(G, 1)
    gap = np.zeros((G, G))
    gap[a_, b_] = PD[START[i]:START[i + 1], 2]
    gap = gap + gap.T
    far = (cls[:, None] == cls[None, :]) & ~np.eye(G, dtype=bool) & (gap > 112)
    tot_far = np.outer(n_g, n_g)[far].sum() / 2
    c2 = lambda x: (x * (x - 1) / 2).sum()  # noqa: E731
    for k in KEYS:
        t = torch.from_numpy(MASKS[k][i].astype(np.float32))[None]
        K = t.shape[1]
        p = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)[0].argmax(0).numpy()
        nk = np.bincount(gi * K + p[fg], minlength=G * K).reshape(G, K).astype(float)
        out[f"{k}|fg_ari"] = ari_table(nk)
        out[f"{k}|whole"] = c2(nk) / c2(n_g) if c2(n_g) else np.nan
        out[f"{k}|farC"] = (nk @ nk.T)[far].sum() / 2 / tot_far if tot_far else np.nan
    return out


if __name__ == "__main__":
    with Pool(8) as pool:
        rows = pool.map(one, range(N), chunksize=4)
    P = {k: np.array([r.get(k, np.nan) for r in rows]) for k in {k for r in rows for k in r}}
    SEEDS = ["", "_seed1", "_seed2"]
    ok = [all(f"{r}{s}|e{e}" in KEYS for s in SEEDS for r in ("trm_t1", "trm")) for e in EPOCHS]
    print(f"{N} val and {N} train images; mean over 3 seeds; T=8/T=1 = mean of per-seed ratios")
    print(f"{'epoch':>5s} | {'val MSE T=1':>11s} {'T=8':>7s} {'T=8/T=1':>8s} | {'train MSE T=1':>13s} {'T=8':>7s} {'T=8/T=1':>8s} |"
          f" {'farC T=1':>8s} {'T=8':>6s} | {'whole T=1':>9s} {'T=8':>6s} | {'FG-ARI T=1':>10s} {'T=8':>6s}")
    for e, good in zip(EPOCHS, ok):
        if not good:
            continue
        mse = {(r, sp): np.array([Z[f"{r}{s}|e{e}|{sp}|mse"][:, -1].mean() for s in SEEDS])
               for r in ("trm_t1", "trm") for sp in ("val", "train")}
        st = {(r, x): np.mean([np.nanmean(P[f"{r}{s}|e{e}|{x}"]) for s in SEEDS])
              for r in ("trm_t1", "trm") for x in ("farC", "whole", "fg_ari")}
        print(f"{e:5d} | {1000 * mse['trm_t1', 'val'].mean():11.2f} {1000 * mse['trm', 'val'].mean():7.2f}"
              f" {(mse['trm', 'val'] / mse['trm_t1', 'val']).mean():8.3f} |"
              f" {1000 * mse['trm_t1', 'train'].mean():13.2f} {1000 * mse['trm', 'train'].mean():7.2f}"
              f" {(mse['trm', 'train'] / mse['trm_t1', 'train']).mean():8.3f} |"
              f" {st['trm_t1', 'farC']:8.3f} {st['trm', 'farC']:6.3f} | {st['trm_t1', 'whole']:9.3f} {st['trm', 'whole']:6.3f} |"
              f" {100 * st['trm_t1', 'fg_ari']:10.2f} {100 * st['trm', 'fg_ari']:6.2f}")
    summ = {}
    for k in KEYS:
        summ[f"{k}|val_mse"] = Z[f"{k}|val|mse"][:, -1].mean()
        summ[f"{k}|train_mse"] = Z[f"{k}|train|mse"][:, -1].mean()
        for x in ("farC", "whole", "fg_ari"):
            summ[f"{k}|{x}"] = np.nanmean(P[f"{k}|{x}"])
    np.savez(f"{C}/curve_sum.npz", **summ)
    last = [e for e, g in zip(EPOCHS, ok) if g][-2:]
    print(f"\nchange from epoch {last[0]} to {last[1]} (val MSE x1000, per seed):")
    for r in ("trm_t1", "trm"):
        d = [1000 * (Z[f"{r}{s}|e{last[1]}|val|mse"][:, -1].mean() - Z[f"{r}{s}|e{last[0]}|val|mse"][:, -1].mean()) for s in SEEDS]
        print(f"  {r:7s} " + " ".join(f"{x:+.3f}" for x in d))
    print("val MSE per step inside T=8 at the last epoch (x1000, mean of seeds):",
          np.round(1000 * np.mean([Z[f"trm{s}|e{last[1]}|val|mse"].mean(0) for s in SEEDS], 0), 2))
