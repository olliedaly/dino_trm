"""Pair counts behind FG-ARI, split by pair type, per image and recursion step.

Over foreground pixels, every pixel pair is one of
  within-object (same instance), same-class (different instances of one category),
  different-class.
For each type we store how many pairs there are and how many share a slot, plus
FG-ARI (sklearn), mBO_i and mBO_c. Output c/p_{run}.npz: arrays (images, steps).
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from slot_shape import ari, best_iou  # noqa: E402

C, RUN = sys.argv[1], sys.argv[2]
S = 336
torch.set_num_threads(1)
try:
    M = np.load(f"{C}/m_{RUN}_steps.npy", mmap_mode="r")
except FileNotFoundError:
    M = np.load(f"{C}/m_{RUN}.npy")[:, None]
GT = np.load(f"{C}/gt.npz")
INST, SEM = GT["inst"], GT["sem"]
KEYS = ("W", "Wa", "SC", "SCa", "DC", "DCa", "fg_ari", "mbo_i", "mbo_c", "max_share")
c2 = lambda x: (x * (x - 1) / 2).sum()  # noqa: E731


def counts(pred, inst, sem, K):
    fg = inst > 0
    g, s, p = inst[fg], sem[fg], pred[fg]
    u, gi = np.unique(g, return_inverse=True)
    n = np.bincount(gi * K + p, minlength=len(u) * K).reshape(len(u), K).astype(float)
    cls = np.bincount(gi * 256 + s, minlength=len(u) * 256).reshape(len(u), 256).argmax(1)
    _, ci = np.unique(cls, return_inverse=True)
    m = np.zeros((ci.max() + 1, K))
    np.add.at(m, ci, n)
    W, Wa, Mc, Ma = c2(n), c2(n.sum(1)), c2(m), c2(m.sum(1))
    Q, T = c2(n.sum(0)), len(g) * (len(g) - 1) / 2
    return W, Wa, Mc - W, Ma - Wa, Q - Mc, T - Ma


def one(i):
    inst, sem = INST[i].astype(np.int64), SEM[i].astype(np.int64)
    out = []
    for t in range(M.shape[1]):
        m = torch.from_numpy(np.array(M[i, t], dtype=np.float32))[None]
        pred = F.interpolate(m, size=(S, S), mode="bilinear", align_corners=False)[0].argmax(0).numpy()
        K = m.shape[1]
        iou_i, _, Ci = best_iou(pred, inst, K)
        iou_c, _, _ = best_iou(pred, sem, K)
        out.append(counts(pred, inst, sem, K) + (
            ari(Ci) if Ci.sum() >= 2 else np.nan,
            iou_i.mean() if len(iou_i) else np.nan,
            iou_c.mean() if len(iou_c) else np.nan,
            np.bincount(pred.ravel()).max() / pred.size))
    return np.array(out)


if __name__ == "__main__":
    with Pool(11) as pool:
        rows = np.stack(pool.map(one, range(len(M)), chunksize=8))   # (N, T, F)
    np.savez_compressed(f"{C}/p_{RUN}.npz", **{k: rows[:, :, j] for j, k in enumerate(KEYS)})
    ref = np.load(f"results/coco/per_image/slot_shape/{RUN}.npz")["fg_ari_orig"]
    print(RUN, "steps", rows.shape[1], "max |diff| final-step FG-ARI vs stored",
          np.nanmax(np.abs(rows[:, -1, 6] - ref)), flush=True)
