"""Report: is the module's output slot attention's own result? (means over seeds)"""
import sys

import numpy as np

C = sys.argv[1]
CFG = {"baseline": ["baseline_pm", "baseline_pm_seed1", "baseline_pm_seed2"],
       "mlp_block": ["mlp_block", "mlp_block_seed1", "mlp_block_seed2"],
       "T=1": ["trm_t1", "trm_t1_seed1", "trm_t1_seed2"],
       "T=2": ["trm_t2"], "T=4": ["trm_t4"],
       "T=8": ["trm", "trm_seed1", "trm_seed2"]}
PD = np.load(f"{C}/pd_trm.npy")
FARSC = (PD[:, 1] == 1) & (PD[:, 2] > 112)
FARDC = (PD[:, 1] == 0) & (PD[:, 2] > 112)
TOUCH_SC = (PD[:, 1] == 1) & (PD[:, 2] <= 1.5)


def get(run, part, m):
    d = np.load(f"{C}/an_{run}.npz")
    if m == "farSC":
        return d[f"{part}|pairs"][FARSC].mean()
    if m == "touchSC":
        return d[f"{part}|pairs"][TOUCH_SC].mean()
    return np.nanmean(d[f"{part}|{m}"])


def final(run):
    return [k.split("|")[0] for k in np.load(f"{C}/an_{run}.npz") if k.startswith("t")][-1]


M = ["fg_ari", "mbo_i", "mbo_c", "farSC", "touchSC", "max_share"]
print("1) slot attention's own partition (sa) vs the decoded output (fin), mean over seeds")
print(f"{'':10s}{'part':5s}" + "".join(f"{m:>9s}" for m in M) + f"{'ARI(fin,sa)':>13s}{'same idx':>10s}")
for c, runs in CFG.items():
    for part in ("sa", "fin"):
        v = [np.mean([get(r, final(r) if part == "fin" else part, m) for r in runs]) for m in M]
        a = np.mean([get(r, "sa", "ari_fin") for r in runs])
        s = np.mean([get(r, "sa", "idx_fin") for r in runs])
        print(f"{c:10s}{part:5s}" + "".join(f"{x:9.3f}" for x in v)
              + (f"{a:13.3f}{s:10.3f}" if part == "fin" else ""))

print("\n2) inside T=8, per step (mean over 3 seeds)")
print(f"{'step':6s}{'fg_ari':>8s}{'mbo_i':>8s}{'farSC':>8s}{'ARI(step,sa)':>14s}{'ARI(step,fin)':>15s}{'|y-x|/|x|':>11s}")
for t in range(1, 9):
    f = lambda m: np.mean([get(r, f"t{t}", m) for r in CFG["T=8"]])  # noqa: E731
    mv = np.mean([np.load(f"{C}/an_{r}.npz")["slot|rel_move"][:, t - 1].mean() for r in CFG["T=8"]])
    print(f"{t:<6d}{f('fg_ari'):8.3f}{f('mbo_i'):8.3f}{f('farSC'):8.3f}{f('ari_sa'):14.3f}{f('ari_fin'):15.3f}{mv:11.3f}")
for c in ("T=1", "mlp_block", "T=2", "T=4"):
    mv = np.mean([np.load(f"{C}/an_{r}.npz")["slot|rel_move"][:, -1].mean() for r in CFG[c]])
    print(f"   {c}: final |y-x|/|x| = {mv:.3f}")

print("\n3) agreement between models' partitions: mean per-image ARI (all pixels)")
X = np.load(f"{C}/xm.npz")
names, A = list(X["names"]), X["ari"].mean(0)


def block(ca, cb, kind_a="fin", kind_b="fin"):
    v = [A[names.index(f"{a}|{kind_a}"), names.index(f"{b}|{kind_b}")]
         for a in CFG[ca] for b in CFG[cb] if not (a == b and kind_a == kind_b)]
    return np.mean(v)


cs = list(CFG)
print("final partitions")
print(f"{'':10s}" + "".join(f"{c:>10s}" for c in cs))
for a in cs:
    print(f"{a:10s}" + "".join(f"{block(a, b):10.3f}" for b in cs))
print("slot-attention partitions")
print(f"{'':10s}" + "".join(f"{c:>10s}" for c in cs))
for a in cs:
    print(f"{a:10s}" + "".join(f"{block(a, b, 'sa', 'sa'):10.3f}" for b in cs))
print("final partition (row) vs baseline's slot-attention partition:",
      {a: round(block(a, "baseline", "fin", "sa"), 3) for a in cs})

print("\n4) cross-attention: share of each slot's attention on each group / that group's area share")
G = ["own obj", "SC near", "SC far", "diff cls", "bg"]
print(f"{'':16s}" + "".join(f"{g:>10s}" for g in G) + f"{'own SA reg':>12s}{'|ca|/|h|':>10s}")
for c in ("mlp_block", "T=1", "T=2", "T=4", "T=8"):
    for s, sname in ((0, "first"), (1, "last")):
        if c in ("mlp_block", "T=1") and s == 0:
            continue
        en, own, nr = [], [], []
        for r in CFG[c]:
            d = np.load(f"{C}/ca_{r}.npz")
            m, a = d["mass"][:, s], d["area"]
            en.append([m[a[:, j] > 0, j].sum() / a[a[:, j] > 0, j].sum() for j in range(5)])
            own.append(d["own_sa"][:, s].mean())
            n = np.load(f"{C}/sa_{r}.npz")["nrm"]          # (N, T, L, 3)
            nr.append(n[:, 0 if s == 0 else -1, :, 1].mean())
        print(f"{c + ' ' + sname:16s}" + "".join(f"{x:10.2f}" for x in np.mean(en, 0))
              + f"{np.mean(own):12.3f}{np.mean(nr):10.3f}")
