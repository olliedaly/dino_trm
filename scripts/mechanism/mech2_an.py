"""Dose-response, mlp_block, test-time depth, and per-image reconstruction test."""
import sys

import numpy as np

C = sys.argv[1]
D = {**dict(np.load(f"{C}/mech.npz")), **dict(np.load(f"{C}/mech2.npz"))}
SEEDS = ["", "_seed1", "_seed2"]
ST = ["fg_ari", "ari_all", "mbo_i", "mbo_c", "whole", "farC", "nearC", "touchC", "diffC", "bgmix"]
SC = {"fg_ari": 100, "ari_all": 100, "mbo_i": 100, "mbo_c": 100}


def row(lab, parts, mse=None):
    v = [np.nanmean([np.nanmean(D[f"{p}|{st}"]) for p in parts]) * SC.get(st, 1) for st in ST]
    print(f"  {lab:26s}" + "".join(f"{x:7.2f}" if j < 4 else f"{x:7.3f}" for j, x in enumerate(v))
          + (f"  {mse:.4f}" if mse is not None else ""))


def mse(files, step=-1):
    return np.mean([np.load(f"{C}/rx_{f}.npz")["mse"][:, step].mean() for f in files])


print(f"  {'':26s} FG-ARI ARI_all  mBO_i  mBO_c  whole   farC  nearC touchC  diffC  bgmix  recon MSE")
print("1) dose-response in trained depth (seed 0 only for every row, to match T=2/T=4)")
row("T=1", ["t1|dec"], mse(["trm_t1"]))
row("T=2", ["t2|step2"], mse(["trm_t2"]))
row("T=4", ["t4|step4"], mse(["trm_t4"]))
row("T=8", ["t8|step8"], mse(["trm"]))
print("2) non-recursive module (mlp_block, 3 seeds) vs T=1 and T=8 (3 seeds)")
row("mlp_block", [f"mlp{s}|dec" for s in SEEDS], mse([f"mlp_block{s}" for s in SEEDS]))
row("T=1", [f"t1{s}|dec" for s in SEEDS], mse([f"trm_t1{s}" for s in SEEDS]))
row("T=8", [f"t8{s}|step8" for s in SEEDS], mse([f"trm{s}" for s in SEEDS]))
print("3) test-time depth (3 seeds): T=1 model run for more steps; T=8 model run to 16")
for n in (2, 4, 8):
    row(f"T=1 model, {n} steps", [f"t1{s}|tt{n}" for s in SEEDS], mse([f"trm_t1{s}_n{n}" for s in SEEDS]))
row("T=8 model, 16 steps", [f"t8{s}|tt16" for s in SEEDS], mse([f"trm{s}_n16" for s in SEEDS]))
print("4) T=2 and T=4 step by step (seed 0)")
for t in (1, 2):
    row(f"T=2 step {t}", [f"t2|step{t}"], mse(["trm_t2"], t - 1))
for t in (1, 2, 3, 4):
    row(f"T=4 step {t}", [f"t4|step{t}"], mse(["trm_t4"], t - 1))

print("\n5) where is reconstruction lost? per-image MSE change T=8 - T=1 (3 seed pairs), by GT content")
dm = np.mean([np.load(f"{C}/rx_trm{s}.npz")["mse"][:, -1] - np.load(f"{C}/rx_trm_t1{s}.npz")["mse"][:, -1] for s in SEEDS], 0)
base = np.mean([np.load(f"{C}/rx_trm_t1{s}.npz")["mse"][:, -1] for s in SEEDS], 0)
rng = np.random.default_rng(0)
groups = {"no same-class pair": D["gt|has_sameC"] == 0,
          "same-class pairs, none far": (D["gt|has_sameC"] == 1) & (D["gt|has_farC"] == 0),
          "has far same-class pair": D["gt|has_farC"] == 1,
          "no objects": np.isnan(D["gt|has_sameC"])}
for g, m in groups.items():
    if m.sum() == 0:
        continue
    x = dm[m]
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)]
    print(f"  {g:28s} n={m.sum():4d}  T=1 MSE {base[m].mean():.4f}  change {x.mean():+.4f} "
          f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]  relative {100 * x.mean() / base[m].mean():+.1f}%")
opp = D["gt|opp_far"]
ok = ~np.isnan(opp)
print(f"  correlation(far-same-class opportunity, MSE change): {np.corrcoef(opp[ok], dm[ok])[0, 1]:+.3f}")
nobj = D["gt|n_obj"]
for lo, hi in [(0, 1), (2, 3), (4, 6), (7, 10), (11, 99)]:
    m = (nobj >= lo) & (nobj <= hi)
    print(f"  objects {lo:2d}-{hi:2d}: n={m.sum():4d}  MSE change {dm[m].mean():+.4f} ({100 * dm[m].mean() / base[m].mean():+.1f}%)")
