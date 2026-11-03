"""Recursion controls against T=1 and T=8 (eval seed 0, all val images).

For each control (c/mechc_{ctrl}{seed}.npz and c/rx_{ctrl}{seed}.npz, from post_control.sh),
the per-image difference to T=1 and to T=8 of the same training seed, averaged over the
seeds available, with an image-bootstrap 95% CI. "share" = (ctrl - T=1) / (T=8 - T=1): how
much of recursion's effect the control reproduces. Also the control's own step-by-step
trajectory, and its FG-ARI change from T=1 by image group.
Usage: ctrl_an.py C trm_1step,trm_final"""
import os
import sys

import numpy as np

C = sys.argv[1]
D = dict(np.load(f"{C}/mech.npz"))
SEEDS = ["", "_seed1", "_seed2"]
ST = ["fg_ari", "ari_all", "mbo_i", "mbo_c", "whole", "farC", "nearC", "touchC", "diffC", "q", "bgmix"]
SC = {"fg_ari": 100, "ari_all": 100, "mbo_i": 100, "mbo_c": 100}
BOOT = np.random.default_rng(0).integers(0, 2000, size=(1000, 2000))
GROUPS = {"no same-class pair": D["gt|has_sameC"] == 0,
          "same-class pairs, none far": (D["gt|has_sameC"] == 1) & (D["gt|has_farC"] == 0),
          "has far same-class pair": D["gt|has_farC"] == 1}


def ci(d, mask=None):
    ok = ~np.isnan(d) if mask is None else ~np.isnan(d) & mask
    idx = np.flatnonzero(ok)
    bs = [d[idx[BOOT[k, :len(idx)] % len(idx)]].mean() for k in range(len(BOOT))]
    return d[ok].mean(), np.percentile(bs, [2.5, 97.5])


for ctrl in sys.argv[2].split(","):
    seeds = [s for s in SEEDS if os.path.exists(f"{C}/mechc_{ctrl}{s}.npz")]
    if not seeds:
        print(f"{ctrl}: no results yet")
        continue
    M = {s: dict(np.load(f"{C}/mechc_{ctrl}{s}.npz")) for s in seeds}
    T = max(int(k.split("|")[1][4:]) for k in M[seeds[0]] if k.startswith(f"{ctrl}|step"))

    def get(which, s, st, t=None):
        if which == "t1":
            return D[f"t1{s}|dec|{st}"]
        if which == "t8":
            return D[f"t8{s}|step{t or 8}|{st}"]
        return M[s][f"{ctrl}{s}|step{t or T}|{st}"]

    def mse(which, s, t=-1):
        f = {"t1": f"trm_t1{s}", "t8": f"trm{s}", "c": f"{ctrl}{s}"}[which]
        return np.load(f"{C}/rx_{f}.npz")["mse"][:, t].astype(np.float64)

    print(f"\n=== {ctrl}: seeds {[s or '_seed0' for s in seeds]} vs T=1 / T=8 of the same seeds ===")
    print(f"  {'stat':8s} {'T=1':>7s} {'T=8':>7s} {'ctrl':>7s} | {'ctrl-T=1 [95% CI]':>26s} | {'ctrl-T=8 [95% CI]':>26s} | share")
    for st in ST + ["mse"]:
        if st == "mse":
            v = {w: np.mean([mse(w, s) for s in seeds], 0) for w in ("t1", "t8", "c")}
            k = 1000
        else:
            v = {w: np.mean([get(w, s, st) for s in seeds], 0) for w in ("t1", "t8", "c")}
            k = SC.get(st, 1)
        m = {w: np.nanmean(x) for w, x in v.items()}
        d1, c1 = ci(v["c"] - v["t1"])
        d8, c8 = ci(v["c"] - v["t8"])
        share = (m["c"] - m["t1"]) / (m["t8"] - m["t1"]) if abs(m["t8"] - m["t1"]) > 1e-9 else np.nan
        print(f"  {st:8s} {k * m['t1']:7.3f} {k * m['t8']:7.3f} {k * m['c']:7.3f} | {k * d1:+7.3f} [{k * c1[0]:+7.3f},{k * c1[1]:+7.3f}] |"
              f" {k * d8:+7.3f} [{k * c8[0]:+7.3f},{k * c8[1]:+7.3f}] | {share:+.2f}")
    print("  (mse x1000)")
    print(f"  step by step (ctrl | T=8, mean of the same seeds):")
    print(f"  {'step':5s} {'FG-ARI':>15s} {'whole':>15s} {'farC':>15s} {'bgmix':>15s} {'mse x1000':>15s}")
    for t in range(1, T + 1):
        row = []
        for st in ("fg_ari", "whole", "farC", "bgmix"):
            k = SC.get(st, 1)
            a = np.mean([np.nanmean(get("c", s, st, t)) for s in seeds]) * k
            b = np.mean([np.nanmean(get("t8", s, st, t)) for s in seeds]) * k
            row.append(f"{a:7.3f}|{b:7.3f}")
        a = np.mean([mse("c", s, t - 1).mean() for s in seeds]) * 1000
        b = np.mean([mse("t8", s, t - 1).mean() for s in seeds]) * 1000
        row.append(f"{a:7.3f}|{b:7.3f}")
        print(f"  {t:<5d} " + " ".join(f"{r:>15s}" for r in row))
    print("  FG-ARI / ARI counting background change by image group (ctrl - T=1 | T=8 - T=1):")
    for g, msk in GROUPS.items():
        out = []
        for st in ("fg_ari", "ari_all"):
            dc, cc = ci(np.mean([get("c", s, st) - get("t1", s, st) for s in seeds], 0), msk)
            d8, _ = ci(np.mean([get("t8", s, st) - get("t1", s, st) for s in seeds], 0), msk)
            out.append(f"{st} {100 * dc:+.2f} [{100 * cc[0]:+.2f},{100 * cc[1]:+.2f}] | {100 * d8:+.2f}")
        print(f"    {g:28s} n={msk.sum():4d}  " + "   ".join(out))
