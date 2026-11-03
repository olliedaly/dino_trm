"""Across runs: per-image ARI (all pixels) between every pair of partitions.

Partitions: each run's final decoder masks ("fin") and its slot attention attention ("sa").
Writes c/xm.npz: names (P,), ari (N, P, P).
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from slot_shape import ari  # noqa: E402

C, RUNS = sys.argv[1], sys.argv[2].split(",")
S, K = 336, 7
torch.set_num_threads(1)
SRC = []
for r in RUNS:
    d = np.load(f"{C}/sa_{r}.npz")
    SRC += [(f"{r}|fin", d["m"][:, -1]), (f"{r}|sa", d["attn"])]


def one(i):
    ps = []
    for _, m in SRC:
        t = torch.from_numpy(np.asarray(m[i], np.float32))[None]
        ps.append(F.interpolate(t, size=(S, S), mode="bilinear",
                                align_corners=False)[0].argmax(0).numpy().ravel())
    A = np.eye(len(ps), dtype=np.float32)
    for a in range(len(ps)):
        for b in range(a + 1, len(ps)):
            c = np.bincount(ps[a] * K + ps[b], minlength=K * K).reshape(K, K)
            A[a, b] = A[b, a] = ari(c)
    return A


if __name__ == "__main__":
    with Pool(11) as pool:
        A = np.stack(pool.map(one, range(len(SRC[0][1])), chunksize=8))
    np.savez_compressed(f"{C}/xm.npz", names=np.array([n for n, _ in SRC]), ari=A)
    print("xm", A.shape, flush=True)
