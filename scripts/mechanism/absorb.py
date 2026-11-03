"""Partial contiguity: hand stray pieces of a slot to a different slot, two ways.

  next{A}  stray pieces smaller than A px go to the model's next-best slot (logits)
  nbr{A}   stray pieces smaller than A px go to the slot that surrounds them most
           (majority over the piece's outer border): blind, no use of the model's ranking
"""
import sys
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import binary_dilation, label as cc_label

sys.path.insert(0, "scripts")
sys.path.insert(0, sys.argv[1].rsplit("/", 1)[0])
from dumb import metrics, up, EIGHT  # noqa: E402

C, RUN = sys.argv[1], sys.argv[2]
AS = (256, 512, 1024, 2048, 4096, 10 ** 9)
M = np.load(f"{C}/m_{RUN}.npy")
GT = np.load(f"{C}/gt.npz")
INST, SEM = GT["inst"], GT["sem"]


def strays(pred, k, A):
    comp, n = cc_label(pred == k, structure=EIGHT)
    if n < 2:
        return []
    sz = np.bincount(comp.ravel())[1:]
    main = sz.argmax() + 1
    return [comp == j for j in range(1, n + 1) if j != main and sz[j - 1] < A]


def nextbest(logm, A):
    logm = logm.copy()
    for _ in range(20):
        pred, moved = logm.argmax(0), False
        for k in range(logm.shape[0]):
            for s in strays(pred, k, A):
                logm[k][s] = -np.inf
                moved = True
        if not moved:
            break
    return logm.argmax(0)


def neighbour(pred, A):
    pred = pred.copy()
    for _ in range(20):
        moved = False
        for k in range(int(pred.max()) + 1):
            for s in strays(pred, k, A):
                ring = binary_dilation(s, EIGHT) & ~s
                pred[s] = np.bincount(pred[ring]).argmax()
                moved = True
        if not moved:
            break
    return pred


def one(i):
    inst, sem = INST[i].astype(np.int64), SEM[i].astype(np.int64)
    K = M.shape[1]
    logm = up(M[i])
    p0 = logm.argmax(0)
    out = {}
    for A in AS:
        for nm, p in ((f"next{A}", nextbest(logm, A)), (f"nbr{A}", neighbour(p0, A))):
            for k, v in metrics(p, inst, sem, K).items():
                out[f"{k}_{nm}"] = v
    return out


if __name__ == "__main__":
    with Pool(11) as pool:
        rows = pool.map(one, range(len(M)), chunksize=8)
    np.savez_compressed(f"{C}/a_{RUN}.npz",
                        **{k: np.array([r[k] for r in rows], np.float32) for k in rows[0]})
    print(RUN, "done", flush=True)
