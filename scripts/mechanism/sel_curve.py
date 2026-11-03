"""Class selectivity over training: when do T=1 and T=8 part?

For each saved epoch in the curve caches (final-step decoder masks of the first 512 val
images), the token-pair profile of bgloc.py: share of token pairs in one slot for pairs in
one object, in two same-class objects and in two different-class objects, by distance.
Selectivity = same-class / different-class sharing (3-7 and 7-14 tokens).
Usage: sel_curve.py C "file:run[,run...];file:run..."  (run names may use {s} for the seed
suffix; mean over the seeds present). Writes c/sel_curve{SEL_TAG}.npz. SEL_KEY=attn reads
slot attention's own maps from curve_sa*.npz instead of the decoder masks."""
import os
import sys

import numpy as np

C = sys.argv[1]
SEEDS = ["", "_seed1", "_seed2"]
GT = np.load(f"{C}/gt.npz")
N, g = 512, 21
MKEY = os.environ.get("SEL_KEY", "val|masks")
INST, SEM = GT["inst"][:N], GT["sem"][:N]
P = INST.shape[1] // g
tok_inst = np.zeros((N, g * g), np.int64)
tok_cls = np.zeros((N, g * g), np.int64)
for i in range(N):
    blk = INST[i].reshape(g, P, g, P).transpose(0, 2, 1, 3).reshape(g * g, P * P).astype(np.int64)
    sblk = SEM[i].reshape(g, P, g, P).transpose(0, 2, 1, 3).reshape(g * g, P * P).astype(np.int64)
    for t in range(g * g):
        tok_inst[i, t] = np.bincount(blk[t]).argmax()
        tok_cls[i, t] = np.bincount(sblk[t][blk[t] == tok_inst[i, t]]).argmax() if tok_inst[i, t] else 0
yy, xx = np.divmod(np.arange(g * g), g)
iu = np.triu_indices(g * g, 1)
dist = np.sqrt((yy[iu[0]] - yy[iu[1]]) ** 2 + (xx[iu[0]] - xx[iu[1]]) ** 2)
binidx = np.digitize(dist, [3, 7, 14], right=True)
tok_inst, tok_cls = tok_inst.astype(np.int16), tok_cls.astype(np.int16)
A, B = tok_inst[:, iu[0]], tok_inst[:, iu[1]]
CA, CB = tok_cls[:, iu[0]], tok_cls[:, iu[1]]
TYP = np.full(A.shape, -1, np.int8)
TYP[(A > 0) & (A == B)] = 0  # same object
TYP[(A > 0) & (B > 0) & (A != B) & (CA == CB)] = 1  # same class
TYP[(A > 0) & (B > 0) & (CA != CB)] = 2  # different class
KEY = TYP * 4 + binidx[None].astype(np.int8)
del A, B, CA, CB
CNT = np.array([np.bincount(KEY[i][TYP[i] >= 0], minlength=12) for i in range(N)]).sum(0)


def rates(masks):
    lab = np.asarray(masks, np.float32).reshape(N, -1, g * g).argmax(1)
    share = lab[:, iu[0]] == lab[:, iu[1]]
    s = np.bincount(KEY[(TYP >= 0) & share], minlength=12)
    return (s / np.maximum(CNT, 1)).reshape(3, 4)


out = {}
for spec in sys.argv[2].split(";"):
    f, runs = spec.split(":")
    Z = np.load(f"{C}/{f}")
    for run in runs.split(","):
        names = [run.format(s=s) for s in SEEDS] if "{s}" in run else [run]
        eps = sorted({int(k.split("|")[1][1:]) for k in Z.keys() if k.split("|")[0] in names and k.endswith(MKEY)}, key=int)
        print(f"{run} ({f}): epoch | sel 3-7 / 7-14 | same obj >14 | same class >14 | diff class <=3")
        for e in eps:
            r = [rates(Z[f"{n}|e{e}|{MKEY}"]) for n in names if f"{n}|e{e}|{MKEY}" in Z]
            out[f"{run}|e{e}"] = np.array(r)
            m = np.mean(r, 0)
            sel = [x[1] / x[2] for x in r]
            print(f"  e{e:<3d} {np.mean([s[1] for s in sel]):.2f} / {np.mean([s[2] for s in sel]):.2f}"
                  f"   ({' '.join(f'{s[1]:.2f}' for s in sel)})"
                  f" | {m[0, 3]:.3f} | {m[1, 3]:.3f} | {m[2, 0]:.3f}", flush=True)
np.savez(f"{C}/sel_curve{os.environ.get('SEL_TAG', '')}.npz", **out)
