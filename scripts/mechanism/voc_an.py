"""Does the COCO mechanism predict the VOC result?"""
import sys

import numpy as np

C = sys.argv[1]
V = dict(np.load(f"{C}/voc_mech.npz"))
D = dict(np.load(f"{C}/mech.npz"))
SEEDS = ["", "_seed1", "_seed2"]
rng = np.random.default_rng(0)


def grp(X):
    return {"no same-class pair": X["gt|has_sameC"] == 0,
            "same-class, none far": (X["gt|has_sameC"] == 1) & (X["gt|has_farC"] == 0),
            "has far same-class pair": X["gt|has_farC"] == 1}


def dvec(X, a, b, st):
    return np.mean([X[f"{b.format(s=s)}|{st}"] - X[f"{a.format(s=s)}|{st}"] for s in SEEDS], 0)


def ci(x):
    x = x[~np.isnan(x)]
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)]
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(x)


gc, gv = grp(D), grp(V)
print("1) image mix (share of images with >= 1 object)")
for g in gc:
    nc, nv = np.isfinite(D["gt|has_sameC"]).sum(), np.isfinite(V["gt|has_sameC"]).sum()
    print(f"  {g:24s} COCO {gc[g].sum() / nc:5.1%}   VOC {gv[g].sum() / nv:5.1%}")
print(f"  objects per image: COCO median {np.median(D['gt|n_obj']):.0f}, VOC median {np.median(V['gt|n_obj']):.0f}")

for lab, X, a, b, groups in [("COCO models on COCO images", D, "t1{s}|dec", "t8{s}|step8", gc),
                             ("COCO models on VOC images", V, "coco_trm_t1{s}", "coco_trm{s}", gv)]:
    print(f"\n2) {lab}: T=8 - T=1, by group (3 seed pairs; image-bootstrap CI)")
    for st in ["fg_ari", "ari_all", "mbo_i"]:
        d = dvec(X, a, b, st) * 100
        cells = []
        for g, m in groups.items():
            mu, lo, hi, n = ci(d[m])
            cells.append(f"{g} {mu:+.2f} [{lo:+.2f},{hi:+.2f}] n={n}")
        mu, lo, hi, n = ci(d)
        print(f"  {st:7s} overall {mu:+.2f} [{lo:+.2f},{hi:+.2f}] | " + " | ".join(cells))
    for st in ["farC", "nearC", "whole", "bgmix"]:
        A = np.nanmean([np.nanmean(X[f"{a.format(s=s)}|{st}"]) for s in SEEDS])
        B = np.nanmean([np.nanmean(X[f"{b.format(s=s)}|{st}"]) for s in SEEDS])
        print(f"  {st:7s} T=1 {A:.3f} -> T=8 {B:.3f}")

print("\n3) prediction: VOC's overall FG-ARI change from COCO's per-group effects x VOC's image mix")
dc = dvec(D, "t1{s}|dec", "t8{s}|step8", "fg_ari") * 100
eff = {g: np.nanmean(dc[m]) for g, m in gc.items()}
nv = sum(m.sum() for m in gv.values())
pred = sum(eff[g] * gv[g].sum() / nv for g in gv)
dv = dvec(V, "coco_trm_t1{s}", "coco_trm{s}", "fg_ari") * 100
print(f"  predicted {pred:+.2f}   observed (COCO models on VOC) {np.nanmean(dv):+.2f}")

print("\n4) VOC-trained models (1 seed): T=1 -> T=8")
for st in ["fg_ari", "ari_all", "mbo_i", "mbo_c", "farC", "nearC", "whole", "bgmix"]:
    k = 100 if st in ("fg_ari", "ari_all", "mbo_i", "mbo_c") else 1
    a, b = V[f"voc_trm_t1_m|{st}"], V[f"voc_trm_m|{st}"]
    cells = []
    for g, m in gv.items():
        cells.append(f"{g} {k * np.nanmean(b[m] - a[m]):+.3f}")
    print(f"  {st:7s} T=1 {k * np.nanmean(a):7.3f} T=8 {k * np.nanmean(b):7.3f} diff {k * np.nanmean(b - a):+.3f} | " + " | ".join(cells))
mse = {t: np.load(f"{C}/voc_{t}.npz")["mse"].mean() for t in ["voc_trm_t1_m", "voc_trm_m", "voc_baseline_pm_m"]}
print("  VOC recon MSE:", {k: round(float(v), 4) for k, v in mse.items()})
