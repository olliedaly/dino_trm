"""Learning curves of single runs side by side (first N val images, eval seed 0): val MSE,
far same-class sharing, whole and FG-ARI of the decoder masks (curve*.npz), and slot
attention's own map (curve_sa*.npz): its FG-ARI and largest-slot share.
Usage: curve_cmp.py C run:tag,run:tag,...   (tag "" = curve.npz, "_1step" = curve_1step.npz)"""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

C = sys.argv[1]
SPEC = [x.split(":") for x in sys.argv[2].split(",")]
S = 336
torch.set_num_threads(1)
CUR = {tag: np.load(f"{C}/curve{tag}.npz") for _, tag in SPEC}
SA = {tag: np.load(f"{C}/curve_sa{tag}.npz") for _, tag in SPEC}
EP = {}
for run, tag in SPEC:
    e1 = {int(k.split("|e")[1].split("|")[0]) for k in CUR[tag].files if k.startswith(f"{run}|e")}
    e2 = {int(k.split("|e")[1].split("|")[0]) for k in SA[tag].files if k.startswith(f"{run}|e")}
    EP[run] = sorted(e1 | e2)
DM = {(run, e): CUR[tag][f"{run}|e{e}|val|masks"] for run, tag in SPEC for e in EP[run] if f"{run}|e{e}|val|masks" in CUR[tag].files}
AM = {(run, e): SA[tag][f"{run}|e{e}|attn"] for run, tag in SPEC for e in EP[run] if f"{run}|e{e}|attn" in SA[tag].files}
N = next(iter(DM.values())).shape[0]
GT = np.load(f"{C}/gt.npz")
INST, SEM = GT["inst"][:N], GT["sem"][:N]
PD = np.load(f"{C}/pd_trm.npy")
START = np.r_[0, np.cumsum(np.bincount(PD[:, 0].astype(int), minlength=len(GT["inst"])))]


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
    for src, arrs in (("dec", DM), ("sa", AM)):
        for key, m in arrs.items():
            t = torch.from_numpy(m[i].astype(np.float32))[None]
            K = t.shape[1]
            p = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)[0].argmax(0).numpy()
            nk = np.bincount(gi * K + p[fg], minlength=G * K).reshape(G, K).astype(float)
            out[(src, key, "fg_ari")] = ari_table(nk)
            if src == "dec":
                out[(src, key, "whole")] = c2(nk) / c2(n_g) if c2(n_g) else np.nan
                out[(src, key, "farC")] = (nk @ nk.T)[far].sum() / 2 / tot_far if tot_far else np.nan
            else:
                tok = m[i].reshape(K, -1).argmax(0)
                out[(src, key, "max_share")] = np.bincount(tok, minlength=K).max() / tok.size
    return out


if __name__ == "__main__":
    with Pool(8) as pool:
        rows = pool.map(one, range(N), chunksize=4)
    P = {k: np.nanmean([r.get(k, np.nan) for r in rows]) for k in {k for r in rows for k in r}}
    runs = [r for r, _ in SPEC]
    cols = [("val MSE x1000", None), ("dec FG-ARI", ("dec", "fg_ari")), ("whole", ("dec", "whole")),
            ("far same-class", ("dec", "farC")), ("SA-map FG-ARI", ("sa", "fg_ari")), ("SA largest slot", ("sa", "max_share"))]
    print(f"{N} val images; columns per block: " + " | ".join(runs))
    for lab, key in cols:
        print(f"\n{lab}")
        for e in sorted({e for r in runs for e in EP[r]}):
            vals = []
            for run, tag in SPEC:
                if key is None:
                    k = f"{run}|e{e}|val|mse"
                    vals.append(f"{1000 * CUR[tag][k][:, -1].mean():8.2f}" if k in CUR[tag].files else f"{'':8s}")
                else:
                    v = P.get((key[0], (run, e), key[1]))
                    sc = 100 if key[1] == "fg_ari" else 1
                    vals.append(f"{sc * v:8.3f}" if v is not None else f"{'':8s}")
            print(f"  e{e:<3d} " + " | ".join(vals))
