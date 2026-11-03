"""Do slots that read the same patches merge when the module is iterated?

The module's cross-attention is a softmax over patches for each slot separately: unlike
slot attention, slots do not compete for patches. Two slots whose queries match the same
patches read the same values and get similar updates.
1. Read overlap between slots: for every pair of slots, the Bhattacharyya coefficient of
   their cross-attention distributions over the 441 patches (mean of the 4 cross-attention
   calls in a step), from c/sa_{run}.npz 'ca' (first and last step). Mean over pairs, per
   model and step; 3 seeds where available.
2. T=1 iterated at test time (c/rx_trm_t1{s}_n8.npz): does a pair's step-1 read overlap
   predict how much the two slots converge by step 8 (cosine of slot vectors, step 8 minus
   step 1), beyond how similar they were at step 1? Rank correlations and a linear fit on
   pair level, 3 seeds.
3. What the overlap is made of: for each pair, the distance between the two slots' read
   centroids and the DINO-feature cosine of what they read (attention-weighted mean features)."""
import sys

import numpy as np
from scipy.stats import spearmanr

C = sys.argv[1]
SEEDS = ["", "_seed1", "_seed2"]
iu = np.triu_indices(7, 1)
g = 21
yy, xx = np.divmod(np.arange(g * g), g)


def overlap(ca):
    """ca (N, L, K, P) -> (N, pairs) Bhattacharyya coefficient of the layer-mean reads."""
    a = ca.astype(np.float32).mean(1)
    a = a / a.sum(-1, keepdims=True)
    r = np.sqrt(a)
    bc = np.einsum("nkp,njp->nkj", r, r)
    return bc[:, iu[0], iu[1]]


def centroid(ca):
    a = ca.astype(np.float32).mean(1)
    a = a / a.sum(-1, keepdims=True)
    return np.stack([(a * yy).sum(-1), (a * xx).sum(-1)], -1)  # (N, K, 2)


print("1. Read overlap between slots (Bhattacharyya coefficient, mean over the 21 slot pairs; 1 = identical reads)")
for name, run, seeds in [("T=1", "trm_t1", SEEDS), ("mlp_block", "mlp_block", SEEDS), ("T=2", "trm_t2", [""]), ("T=4", "trm_t4", [""]),
                         ("T=8", "trm", SEEDS)]:
    first, last = [], []
    for s in seeds:
        ca = np.load(f"{C}/sa_{run}{s}.npz")["ca"]
        first.append(overlap(ca[:, 0]).mean())
        last.append(overlap(ca[:, -1]).mean())
    print(f"   {name:10s} first step {np.mean(first):.3f} ({' '.join(f'{x:.3f}' for x in first)})   last step {np.mean(last):.3f} ({' '.join(f'{x:.3f}' for x in last)})")

print("\n2. T=1 iterated at test time: step-1 read overlap vs convergence of the slot pair by step 8")
for s in SEEDS:
    ov = overlap(np.load(f"{C}/sa_trm_t1{s}.npz")["ca"][:, 0]).ravel()
    sl = np.load(f"{C}/rx_trm_t1{s}_n8.npz")["slots"].astype(np.float32)  # (N, 8, K, D)
    y1 = np.load(f"{C}/sa_trm_t1{s}.npz")["y"][:, 0].astype(np.float32)
    assert np.allclose(y1, sl[:, 0], atol=0.05), "slot order differs between the two extractions"
    n = sl / np.linalg.norm(sl, axis=-1, keepdims=True)
    cos = np.einsum("ntkd,ntjd->ntkj", n, n)[:, :, iu[0], iu[1]]  # (N, 8, pairs)
    c1, c8 = cos[:, 0].ravel(), cos[:, 7].ravel()
    conv = c8 - c1
    X = np.c_[np.ones_like(ov), ov, c1]
    beta = np.linalg.lstsq(X, conv, rcond=None)[0]
    zs = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    bstd = np.linalg.lstsq(np.c_[np.ones_like(ov), zs(ov), zs(c1)], zs(conv), rcond=None)[0]
    hi = ov > np.quantile(ov, 0.9)
    lo = ov < np.quantile(ov, 0.1)
    print(f"   seed{s or '_seed0'}: rho(overlap, convergence) {spearmanr(ov, conv)[0]:+.2f}; rho(step-1 similarity, convergence) {spearmanr(c1, conv)[0]:+.2f}; "
          f"standardised fit: overlap {bstd[1]:+.2f}, step-1 similarity {bstd[2]:+.2f}")
    print(f"      top-10% overlap pairs: cos {c1[hi].mean():.2f} -> {c8[hi].mean():.2f};  bottom-10%: {c1[lo].mean():.2f} -> {c8[lo].mean():.2f};  all: {c1.mean():.2f} -> {c8.mean():.2f}")

print("\n3. What overlapping reads are made of (seed 0, first step): pairs binned by read overlap")
for name, run in [("T=1", "trm_t1"), ("T=8", "trm")]:
    ca = np.load(f"{C}/sa_{run}.npz")["ca"][:, 0]
    ov = overlap(ca)
    cen = centroid(ca)
    d = np.linalg.norm(cen[:, iu[0]] - cen[:, iu[1]], axis=-1)
    q = np.quantile(ov, [0, 0.5, 0.9, 1])
    print(f"   {name}: read-centroid distance (patches) by overlap bin <50% / 50-90% / >90%: " +
          " / ".join(f"{d[(ov >= q[k]) & (ov <= q[k + 1])].mean():.2f}" for k in range(3)) + f";  all pairs {d.mean():.2f}")
