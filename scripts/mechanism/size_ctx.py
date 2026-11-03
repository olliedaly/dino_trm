"""Is the object-size slope of the FG-ARI change (README section 3) a proxy for same-class context?"""
import sys

import numpy as np

C = sys.argv[1]
D = dict(np.load(f"{C}/mech.npz"))
inst = np.load(f"{C}/gt.npz")["inst"]
size = np.array([np.bincount(x[x > 0])[np.unique(x[x > 0])].mean() / 256 if (x > 0).any() else np.nan for x in inst])
SEEDS = ["", "_seed1", "_seed2"]
d = np.mean([D[f"t8{s}|step8|fg_ari"] - D[f"t1{s}|dec|fg_ari"] for s in SEEDS], 0) * 100
opp = D["gt|opp_far"]
groups = {"all": np.ones(len(d), bool), "no same-class pair": D["gt|has_sameC"] == 0,
          "same-class, none far": (D["gt|has_sameC"] == 1) & (D["gt|has_farC"] == 0),
          "has far same-class pair": D["gt|has_farC"] == 1}
bands = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 1e9)]
print("dFG-ARI (T=8 - T=1, 3 seed pairs) by mean object size (patches); n in brackets")
print(f"  {'':24s}" + "".join(f"{f'{lo:g}-{hi:g}' if hi < 1e9 else f'{lo:g}+':>14s}" for lo, hi in bands))
for g, m in groups.items():
    cells = []
    for lo, hi in bands:
        k = m & (size >= lo) & (size < hi) & ~np.isnan(d)
        cells.append(f"{d[k].mean():+6.2f} ({k.sum():4d})" if k.sum() >= 15 else f"{'-':>6s} ({k.sum():4d})")
    print(f"  {g:24s}" + "".join(f"{c:>14s}" for c in cells))
print("share of images with a far same-class pair, by size band:",
      " ".join(f"{np.nanmean(D['gt|has_farC'][(size >= lo) & (size < hi)]):.2f}" for lo, hi in bands))
ok = ~np.isnan(d) & ~np.isnan(size) & ~np.isnan(opp)
X1 = np.c_[np.ones(ok.sum()), np.log(size[ok])]
X2 = np.c_[X1, opp[ok], D["gt|has_sameC"][ok]]
X3 = np.c_[np.ones(ok.sum()), opp[ok], D["gt|has_sameC"][ok]]
for lab, X in [("log size only", X1), ("same-class context only", X3), ("both", X2)]:
    beta, *_ = np.linalg.lstsq(X, d[ok], rcond=None)
    r = d[ok] - X @ beta
    print(f"  OLS {lab:24s} R2 {1 - r.var() / d[ok].var():.4f}  coefs {np.round(beta, 3)}")
