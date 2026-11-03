"""Object pairs: does a model put two objects in the same slot because of class or distance?

For every pair of GT objects in an image: same class?, gap between them (px), and the
probability that a random pixel of one and a random pixel of the other share a slot,
for the T=1 model, the T=8 model's first step and its final step.
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

C, T1, T8 = sys.argv[1], sys.argv[2], sys.argv[3]
S = 336
torch.set_num_threads(1)
M1 = np.load(f"{C}/m_{T1}.npy")
M8 = np.load(f"{C}/m_{T8}_steps.npy", mmap_mode="r")
GT = np.load(f"{C}/gt.npz")
INST, SEM = GT["inst"], GT["sem"]


def pred(m):
    m = torch.from_numpy(np.array(m, dtype=np.float32))[None]
    return F.interpolate(m, size=(S, S), mode="bilinear", align_corners=False)[0].argmax(0).numpy()


def share(p, gi, n_obj, K=7):
    n = np.bincount(gi * K + p, minlength=n_obj * K).reshape(n_obj, K).astype(float)
    P = n / n.sum(1, keepdims=True)
    return P @ P.T


def one(i):
    inst, sem = INST[i].astype(np.int64), SEM[i].astype(np.int64)
    fg = inst > 0
    u, gi = np.unique(inst[fg], return_inverse=True)
    if len(u) < 2:
        return np.zeros((0, 7))
    cls = np.bincount(gi * 256 + sem[fg], minlength=len(u) * 256).reshape(len(u), 256).argmax(1)
    gap = np.zeros((len(u), len(u)))
    for a, ida in enumerate(u):
        d = distance_transform_edt(inst != ida)
        for b, idb in enumerate(u):
            if b > a:
                gap[a, b] = gap[b, a] = d[inst == idb].min()
    sh = [share(pred(m)[fg], gi, len(u)) for m in (M1[i], M8[i, 0], M8[i, -1])]
    a, b = np.triu_indices(len(u), 1)
    return np.stack([np.full(len(a), i), cls[a] == cls[b], gap[a, b],
                     sh[0][a, b], sh[1][a, b], sh[2][a, b], np.full(len(a), len(u))], 1)


if __name__ == "__main__":
    with Pool(11) as pool:
        rows = np.concatenate(pool.map(one, range(len(M1)), chunksize=8))
    np.save(f"{C}/pd_{T8}.npy", rows.astype(np.float32))
    print(T8, rows.shape, flush=True)
