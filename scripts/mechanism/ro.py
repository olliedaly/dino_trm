"""Attention map vs decoder masks for the same slots (eval seed 0, all val images).
Per image: label changes between neighbouring patches (21x21 grid; map, decoder, GT),
share of pixels where the two readouts disagree, which one is right there (pixel's GT
instance == its slot's majority instance, background counted), and whether the
disagreement sits within one patch (16 px) of a GT boundary."""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

C = sys.argv[1]
S = 336
torch.set_num_threads(1)
GT = np.load(f"{C}/gt.npz")
INST = GT["inst"]
RUNS = {"baseline_pm": ("it_baseline_pm.npz", "attn_3", "m_3", None),
        "sa_rec_1step": ("it_sa_rec_1step.npz", "attn", "m", 7)}
D = {}
for r, (f, a, m, j) in RUNS.items():
    z = np.load(f"{C}/{f}")
    D[r] = (z[a] if j is None else z[a][:, j], z[m] if j is None else z[m][:, j])


def changes(lab):
    return ((lab[1:] != lab[:-1]).mean() + (lab[:, 1:] != lab[:, :-1]).mean()) / 2


def up(x):
    t = torch.from_numpy(np.asarray(x, np.float32))[None]
    return F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)[0].argmax(0).numpy()


def right(p, inst):
    K = p.max() + 1
    ids, gi = np.unique(inst, return_inverse=True)
    n = np.bincount(p.ravel() * len(ids) + gi.ravel(), minlength=K * len(ids)).reshape(K, len(ids))
    return ids[n.argmax(1)][p] == inst


def one(i):
    inst = INST[i].astype(np.int64)
    bnd = np.zeros_like(inst, bool)
    bnd[1:] |= inst[1:] != inst[:-1]
    bnd[:, 1:] |= inst[:, 1:] != inst[:, :-1]
    near = distance_transform_edt(~bnd) <= 16
    out = [changes(inst[8::16, 8::16]), near.mean()]
    for r in RUNS:
        am, dm = D[r][0][i], D[r][1][i]
        pm, pd = up(am), up(dm)
        dis = pm != pd
        rm, rd = right(pm, inst), right(pd, inst)
        out += [changes(np.asarray(am, np.float32).argmax(0)), changes(np.asarray(dm, np.float32).argmax(0)),
                dis.mean(), rm[dis].mean() if dis.any() else np.nan, rd[dis].mean() if dis.any() else np.nan,
                near[dis].mean() if dis.any() else np.nan, rm.mean(), rd.mean()]
    return out


if __name__ == "__main__":
    with Pool(11) as pool:
        R = np.array(pool.map(one, range(len(INST)), chunksize=8))
    print(f"GT: label changes between neighbouring patches {R[:, 0].mean():.3f}; pixels within 16 px of a GT boundary {R[:, 1].mean():.0%}")
    for k, r in enumerate(RUNS):
        v = np.nanmean(R[:, 2 + 8 * k: 10 + 8 * k], 0)
        print(f"{r:13s} label changes: map {v[0]:.3f}, decoder {v[1]:.3f} | pixels where they disagree {v[2]:.0%} | "
              f"right there: map {v[3]:.0%}, decoder {v[4]:.0%} | disagreement within 16 px of GT boundary {v[5]:.0%} | "
              f"right overall: map {v[6]:.1%}, decoder {v[7]:.1%}")
