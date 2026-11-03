"""Sufficiency test at inference: does making T=1's masks local (geo2.py loc{b}: argmax of
log m_k - b d_k^2, d_k = distance to the centroid of slot k's main piece) reproduce T=8's
FG-ARI pattern? Per image group, FG-ARI change from T=1's own masks (3 seeds), next to the
T=8 - T=1 change; plus the mean sharing of far (> 112 px) same-class object pairs."""
import sys

import numpy as np

C = sys.argv[1]
M = np.load(f"{C}/mech.npz")
PD = np.load(f"{C}/pd_trm.npy")
FAR = (PD[:, 1] == 1) & (PD[:, 2] > 112)
SEEDS = ["", "_seed1", "_seed2"]
G1 = {s: np.load(f"{C}/g_trm_t1{s}.npz") for s in SEEDS}
G8 = {s: np.load(f"{C}/g_trm{s}.npz") for s in SEEDS}
groups = {"no same-class pair": M["gt|has_sameC"] == 0,
          "same-class, none far": (M["gt|has_sameC"] == 1) & (M["gt|has_farC"] == 0),
          "far same-class pair": M["gt|has_farC"] == 1, "all": np.ones(2000, bool)}
print(f"{'variant':12s} {'far pair sharing':>16s} | " + " | ".join(f"{g:>20s}" for g in groups) + " |  mBO_i")
base = {s: G1[s]["final|fg_ari"] for s in SEEDS}
for name, src in [("T=1", "final"), *[(f"T=1 loc{b}", f"loc{b}") for b in (2, 5, 10, 20, 40)], ("T=8", None)]:
    if src is None:
        fg = np.mean([G8[s]["final|fg_ari"] - base[s] for s in SEEDS], 0)
        fp = np.mean([G8[s]["final|pairs"][FAR].mean() for s in SEEDS])
        mb = np.mean([G8[s]["final|mbo_i"].mean() - G1[s]["final|mbo_i"].mean() for s in SEEDS])
    else:
        fg = np.mean([G1[s][f"{src}|fg_ari"] - base[s] for s in SEEDS], 0)
        fp = np.mean([G1[s][f"{src}|pairs"][FAR].mean() for s in SEEDS])
        mb = np.mean([G1[s][f"{src}|mbo_i"].mean() - G1[s]["final|mbo_i"].mean() for s in SEEDS])
    print(f"{name:12s} {fp:16.3f} | " + " | ".join(f"{100 * np.nanmean(fg[m]):+20.2f}" for m in groups.values()) + f" | {100 * mb:+6.2f}")
