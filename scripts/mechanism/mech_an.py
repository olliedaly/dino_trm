"""Tables from c/mech.npz: T=1 vs T=8 with seed spread and image-bootstrap CIs."""
import sys

import numpy as np

C = sys.argv[1]
D = dict(np.load(f"{C}/mech.npz"))
SEEDS = ["", "_seed1", "_seed2"]
rng = np.random.default_rng(0)
BOOT = rng.integers(0, 2000, size=(2000, 2000))


def per_seed(fmt, stat):
    return np.array([D[f"{fmt.format(s=s)}|{stat}"] for s in SEEDS])  # (3, 2000)


def diff(stat, a="t1{s}|dec", b="t8{s}|step8", mask=None):
    A, B = per_seed(a, stat), per_seed(b, stat)
    d = (B - A).mean(0)
    ok = ~np.isnan(d) if mask is None else (~np.isnan(d)) & mask
    idx = np.where(ok)[0]
    bs = [np.nanmean(d[idx[BOOT[k, :len(idx)] % len(idx)]]) for k in range(1000)]
    seed_means = np.nanmean((B - A)[:, ok], 1)
    return np.nanmean(A[:, ok]), np.nanmean(B[:, ok]), np.nanmean(d[ok]), np.percentile(bs, [2.5, 97.5]), seed_means, ok.sum()


print("1) T=1 -> T=8 (3 seed pairs; CI = image bootstrap of the seed-averaged per-image difference)")
SC = {"fg_ari": 100, "ari_all": 100, "mbo_i": 100, "mbo_c": 100}
for st in ["fg_ari", "ari_all", "mbo_i", "mbo_c", "whole", "farC", "nearC", "touchC", "sameC", "diffC", "q", "bgmix"]:
    k = SC.get(st, 1)
    a, b, d, ci, sm, n = diff(st)
    print(f"  {st:7s} T=1 {k * a:7.3f}  T=8 {k * b:7.3f}  diff {k * d:+7.3f} [{k * ci[0]:+.3f}, {k * ci[1]:+.3f}]  per seed {' '.join(f'{k * x:+.3f}' for x in sm)}  n={n}")

print("\n2) where does the FG-ARI gain come from? images split by GT content (diff T=8 - T=1)")
groups = {"no same-class pair": D["gt|has_sameC"] == 0,
          "same-class pairs, none far": (D["gt|has_sameC"] == 1) & (D["gt|has_farC"] == 0),
          "has far same-class pair": D["gt|has_farC"] == 1}
for g, m in groups.items():
    row = []
    for st in ["fg_ari", "ari_all", "mbo_i", "mbo_c"]:
        a, b, d, ci, sm, n = diff(st, mask=m)
        row.append(f"{st} {100 * d:+.2f} [{100 * ci[0]:+.2f},{100 * ci[1]:+.2f}]")
    print(f"  {g:28s} n={m.sum():4d} | " + " | ".join(row))
opp = D["gt|opp_far"]
d = (per_seed("t8{s}|step8", "fg_ari") - per_seed("t1{s}|dec", "fg_ari")).mean(0)
ok = ~np.isnan(d) & ~np.isnan(opp)
q = np.quantile(opp[ok & (opp > 0)], [0, .25, .5, .75, 1])
print("  by share of foreground pixel pairs that are far same-class (quartiles of images with any):")
for lo, hi in zip(q[:-1], q[1:]):
    m = ok & (opp > lo) & (opp <= hi)
    print(f"    opp {lo:.3f}-{hi:.3f}: n={m.sum():4d}  dFG-ARI {100 * d[m].mean():+.2f}")
print(f"  correlation(opp, dFG-ARI) over all images: {np.corrcoef(opp[ok], d[ok])[0, 1]:+.3f}")

print("\n3) inside T=8, step by step (mean of 3 seeds)")
print("  step  FG-ARI ARI_all  mBO_i  mBO_c  whole   farC  nearC touchC  diffC   bgmix")
rows = [("T=1", "t1{s}|dec")] + [(f"{t}", f"t8{{s}}|step{t}") for t in range(1, 9)]
for lab, fmt in rows:
    v = [np.nanmean(per_seed(fmt, st)) * SC.get(st, 1) for st in ["fg_ari", "ari_all", "mbo_i", "mbo_c", "whole", "farC", "nearC", "touchC", "diffC", "bgmix"]]
    print(f"  {lab:5s}" + "".join(f"{x:7.2f}" if j < 4 else f"{x:7.3f}" for j, x in enumerate(v)))

print("\n4) other partitions (mean of seeds where available)")
for lab, fmts in [("baseline decoder", [f"base{s}|dec" for s in SEEDS]), ("baseline map", [f"base{s}|map" for s in SEEDS]),
                  ("sa_rec_1step decoder", ["sarec|dec"]), ("sa_rec_1step map", ["sarec|map"])]:
    v = [np.nanmean([np.nanmean(D[f"{f}|{st}"]) for f in fmts]) * SC.get(st, 1) for st in ["fg_ari", "ari_all", "mbo_i", "mbo_c", "whole", "farC", "nearC", "touchC", "diffC", "bgmix"]]
    print(f"  {lab:21s}" + "".join(f"{x:7.2f}" if j < 4 else f"{x:7.3f}" for j, x in enumerate(v)))
