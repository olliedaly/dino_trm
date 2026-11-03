"""Summary of handover.py: the output's dependence on slot attention's slots and slot
attention's slot concentration by epoch (3-seed means for T=1 and T=8), slot attention's
gradient before and after its collapse, and the warm starts.
Reads every c/handover_*.npz. Usage: handover_sum.py C"""
import glob
import sys

import numpy as np

C = sys.argv[1]
R = {}
for f in sorted(glob.glob(f"{C}/handover_*.npz")):
    if f.endswith("_test.npz"):
        continue
    Z = np.load(f)
    R.update({k: Z[k] for k in Z.files})


def rel(r, e):
    k = f"{r}|e{e}"
    if f"{k}|err" not in R:
        return np.nan
    m, mr, mo = R[f"{k}|err"][:, -1].mean(), R[f"{k}|err_roll"][:, -1].mean(), R[f"{k}|err_other"].mean()
    return (mr - m) / (mo - m)


def get(r, e, key):
    k = f"{r}|e{e}|{key}"
    return float(R[k]) if k in R else np.nan


S3 = ["", "_seed1", "_seed2"]
ROWS = [("T=1 (3 seeds)", ["trm_t1" + s for s in S3]), ("T=2", ["trm_t2"]), ("T=4", ["trm_t4"]),
        ("T=8 (3 seeds)", ["trm" + s for s in S3]), ("T=8, final-step loss only", ["trm_final"]),
        ("T=8, 1-step gradients", ["trm_1step"]), ("mlp_block (no recursion)", ["mlp_block"]),
        ("VOC T=8 with cross-attention", ["voc:trm_m"]), ("VOC T=8 without cross-attention", ["voc:trm"])]
EP = [0, 1, 2, 3, 4, 5, 7, 11, 19, 29]
for title, fn in (("dependence on slot attention's slots", rel),
                  ("slot attention's slot concentration (1/7 = even)", lambda r, e: get(r, e, "conc"))):
    print(f"\n{title}\n{'epoch':32s}" + "".join(f"{e:>6d}" for e in EP))
    for lab, runs in ROWS:
        print(f"{lab:32s}" + "".join(f"{np.nanmean([fn(r, e) for r in runs]) if not all(np.isnan([fn(r, e) for r in runs])) else np.nan:6.2f}" for e in EP))
    for r in ["trm", "trm_seed1", "trm_seed2"]:
        print(f"  {r:30s}" + "".join(f"{fn(r, e):6.2f}" for e in EP))

print("\nslot attention's gradient (rms, coherence) by epoch")
for lab, runs in ROWS:
    for r in runs:
        eps = sorted({int(k.split("|")[1][1:]) for k in R if k.startswith(r + "|") and k.endswith("g_sa_coh")})
        print(f"{r:14s}" + " ".join(f"e{e}:{np.sqrt(get(r, e, 'g_sa_sq')):.2f}/{get(r, e, 'g_sa_coh'):+.3f}" for e in eps))

print("\nwarm starts: dependence and concentration, epochs 0-5")
for r in ["ws_trme2_T8_lr30", "ws_trme2_T8_frozen_lr30", "ws_trme2_T1", "ws_trm_t1e2_T8_lr30", "ws_trm_t1e2_T8", "ws_trm_t1_T8"]:
    print(f"{r:26s} start " + " ".join(f"{rel(r, e):.2f}" for e in range(6)) + " | conc " + " ".join(f"{get(r, e, 'conc'):.3f}" for e in range(6)))
print("start points: T=8 e2 %.2f, T=1 e2 %.2f, T=1 e29 %.2f" % (rel("trm", 2), rel("trm_t1", 2), rel("trm_t1", 29)))
