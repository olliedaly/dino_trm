"""What does each GT object share its slot with? (final masks, eval seed 0)

For a random pixel of object o and a random other pixel of the same slot, the probability
that the other pixel is: o itself, another object of o's class, an object of another class,
or background. FG-ARI only sees the first three; IoU (mBO) sees all four.
Also per object: best-slot IoU, precision (object share of that slot), recall.
Writes c/mates_{run}.npz: per object (n_obj, 4) mates, iou, prec, rec, area (px).
Plus GT stats: pieces per object (8-connected, >= 1 patch).
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as cc_label

C, RUN = sys.argv[1], sys.argv[2]
S, PATCH = 336, 256
EIGHT = np.ones((3, 3), int)
torch.set_num_threads(1)
M = np.load(f"{C}/m_{RUN}.npy", mmap_mode="r")
GT = np.load(f"{C}/gt.npz")
INST, SEM = GT["inst"], GT["sem"]


def one(i):
    inst, sem = INST[i].astype(np.int64), SEM[i].astype(np.int64)
    t = torch.from_numpy(np.array(M[i], dtype=np.float32))[None]
    p = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)[0].argmax(0).numpy()
    K = t.shape[1]
    ids, gi = np.unique(inst, return_inverse=True)          # ids[0] == 0 is background
    gi = gi.reshape(inst.shape)
    n = np.bincount(gi.ravel() * K + p.ravel(), minlength=len(ids) * K).reshape(len(ids), K).astype(float)
    if ids[0] != 0 or len(ids) < 2:
        return np.zeros((0, 10))
    cls = np.array([np.bincount(sem[inst == j]).argmax() for j in ids])
    slot_tot = n.sum(0)
    rows = []
    for o in range(1, len(ids)):
        same = (cls == cls[o]) & (np.arange(len(ids)) != o)
        same[0] = False
        diff = (cls != cls[o])
        diff[0] = False
        w = n[o] / n[o].sum()                               # where o's pixels are
        mates = np.stack([n[o], n[same].sum(0), n[diff].sum(0), n[0]]) / np.maximum(slot_tot, 1)
        k = (n[o] / (n[o].sum() + slot_tot - n[o])).argmax()
        comp, nc = cc_label(inst == ids[o], structure=EIGHT)
        pieces = (np.bincount(comp.ravel())[1:] >= PATCH).sum()
        rows.append(np.r_[mates @ w, n[o, k] / (n[o].sum() + slot_tot[k] - n[o, k]),
                          n[o, k] / slot_tot[k], n[o, k] / n[o].sum(), n[o].sum(), pieces, i])
    return np.array(rows)


if __name__ == "__main__":
    with Pool(11) as pool:
        R = np.concatenate(pool.map(one, range(len(INST)), chunksize=8))
    np.savez_compressed(f"{C}/mates_{RUN}.npz", mates=R[:, :4], iou=R[:, 4], prec=R[:, 5],
                        rec=R[:, 6], area=R[:, 7], pieces=R[:, 8], img=R[:, 9])
    print(RUN, R.shape, flush=True)
