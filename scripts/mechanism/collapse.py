"""Do slots stay distinct when the module is iterated? Per step: mean pairwise cosine
similarity of the 7 slots, number of slots that own >= 1 patch (argmax of decoder masks),
and how far each slot's mask centroid moves from the previous step (patches)."""
import sys

import numpy as np

C = sys.argv[1]
g = np.stack(np.mgrid[:21, :21], -1).reshape(441, 2).astype(np.float32)


def stats(f):
    z = np.load(f"{C}/rx_{f}.npz")
    S, M = z["slots"].astype(np.float32), z["masks"].astype(np.float32)   # (N, T, K, D), (N, T, K, 21, 21)
    N, T, K, _ = S.shape
    Sn = S / np.linalg.norm(S, axis=-1, keepdims=True)
    cos = np.einsum("ntkd,ntjd->ntkj", Sn, Sn)
    off = ~np.eye(K, dtype=bool)
    sim = cos[:, :, off].mean(-1).mean(0)
    am = M.reshape(N, T, K, 441).argmax(2)                         # (N, T, 441)
    used = np.array([[len(np.unique(am[n, t])) for t in range(T)] for n in range(N)]).mean(0)
    w = M.reshape(N, T, K, 441)
    w = w / w.sum(-1, keepdims=True)
    cen = w @ g                                                     # (N, T, K, 2)
    move = np.r_[np.nan, np.linalg.norm(np.diff(cen, axis=1), axis=-1).mean((0, 2))]
    return sim, used, move


for lab, f in [("T=1 model iterated to 8 steps", "trm_t1_n8"), ("T=8 model run to 16 steps", "trm_n16")]:
    sim, used, move = stats(f)
    print(lab)
    print("  slot cosine similarity  " + " ".join(f"{x:.2f}" for x in sim))
    print("  slots owning a patch    " + " ".join(f"{x:.1f}" for x in used))
    print("  centroid move (patches) " + " ".join("  - " if np.isnan(x) else f"{x:.2f}" for x in move))
