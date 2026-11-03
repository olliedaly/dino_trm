"""How does slot sharing change when a model's module is iterated at test time?
T=1 run for 8 steps (it was trained for 1) and T=8 run for 16 (trained for 8): token-pair
sharing by type and distance at several steps, and the same-class / different-class ratio.
Built from bgloc.py (same token labels and bins).
Original bgloc.py docstring:
Is T=8's locality specific to objects, or general? Token-level (21x21) slot sharing by
distance for pairs of background tokens, and for pairs of tokens of one object, of two
same-class objects and of two different-class objects (all val images, eval seed 0, final
decoder masks). A token's GT is its pixels' majority instance (0 = background).
Distance bins in tokens (1 token = 16 px): 1-3, 3-7, 7-14, > 14.
Prints the 3-seed means for T=1 and T=8 (and references) and the T=8 - T=1 difference."""
import os
import sys

import numpy as np

C = sys.argv[1]
SEEDS = ["", "_seed1", "_seed2"]
GT = np.load(f"{C}/gt.npz")
INST, SEM = GT["inst"], GT["sem"]
N, S = INST.shape[0], INST.shape[1]
g = 21
P = S // g
# token GT: majority instance and its class
tok_inst = np.zeros((N, g * g), np.int64)
tok_cls = np.zeros((N, g * g), np.int64)
for i in range(N):
    blk = INST[i].reshape(g, P, g, P).transpose(0, 2, 1, 3).reshape(g * g, P * P).astype(np.int64)
    sblk = SEM[i].reshape(g, P, g, P).transpose(0, 2, 1, 3).reshape(g * g, P * P).astype(np.int64)
    for t in range(g * g):
        v = np.bincount(blk[t])
        tok_inst[i, t] = v.argmax()
        tok_cls[i, t] = np.bincount(sblk[t][blk[t] == tok_inst[i, t]]).argmax() if tok_inst[i, t] else 0
yy, xx = np.divmod(np.arange(g * g), g)
D = np.sqrt((yy[:, None] - yy[None]) ** 2 + (xx[:, None] - xx[None]) ** 2)
BINS = [(1, 3), (3, 7), (7, 14), (14, 99)]
iu = np.triu_indices(g * g, 1)
dist = D[iu]
binidx = np.digitize(dist, [3, 7, 14], right=True)  # (0, 3], (3, 7], (7, 14], > 14


def profile(labels):
    """labels (N, g*g) argmax slot per token -> sums[type, bin], counts[type, bin]."""
    types = ["bg-bg", "same object", "same class", "diff class"]
    s = np.zeros((4, len(BINS)))
    n = np.zeros((4, len(BINS)))
    for i in range(N):
        a, b = tok_inst[i][iu[0]], tok_inst[i][iu[1]]
        ca, cb = tok_cls[i][iu[0]], tok_cls[i][iu[1]]
        share = labels[i][iu[0]] == labels[i][iu[1]]
        typ = np.full(len(a), -1)
        typ[(a == 0) & (b == 0)] = 0
        typ[(a > 0) & (a == b)] = 1
        typ[(a > 0) & (b > 0) & (a != b) & (ca == cb)] = 2
        typ[(a > 0) & (b > 0) & (ca != cb)] = 3
        ok = (typ >= 0) & (binidx >= 0)
        np.add.at(s, (typ[ok], binidx[ok]), share[ok])
        np.add.at(n, (typ[ok], binidx[ok]), 1)
    return types, s, n


def labels_of(f, t=None):
    m = np.load(f"{C}/{f}", mmap_mode="r")
    m = m["masks"] if f.endswith(".npz") else m
    m = np.asarray(m[:, t] if t is not None else m, dtype=np.float32)
    return m.reshape(N, m.shape[1], g * g).argmax(1)



SETS = {"T=1 iterated": [(f"rx_trm_t1{s}_n8.npz", (0, 1, 3, 7)) for s in SEEDS],
        "T=8 iterated": [(f"rx_trm{s}_n16.npz", (0, 7, 15)) for s in SEEDS]}
if len(sys.argv) > 2:  # other sets as "name=file pattern:steps;..." ({s} = seed suffix; seeds whose file exists)
    SETS = {}
    for spec in sys.argv[2].split(";"):
        name, rest = spec.rsplit("=", 1)
        pat, steps = rest.rsplit(":", 1)
        ts = tuple(int(t) for t in steps.split(","))
        files = dict.fromkeys(pat.format(s=s) for s in SEEDS if os.path.exists(f"{C}/{pat.format(s=s)}"))
        SETS[name] = [(f, ts) for f in files]
for name, files in SETS.items():
    steps = files[0][1]
    rates = np.mean([[(lambda r: r[1] / np.maximum(r[2], 1))(profile(labels_of(f, t))) for t in ts] for f, ts in files], 0)
    types = ["bg-bg", "same object", "same class", "diff class"]
    print(f"{name} ({len(files)} seeds), share of token pairs in one slot; bins <=3 / 3-7 / 7-14 / >14 tokens", flush=True)
    for j, t in enumerate(steps):
        print(f"  step {t + 1}:")
        for k, ty in enumerate(types):
            print(f"     {ty:12s} " + "  ".join(f"{rates[j, k, b]:.3f}" for b in range(4)))
        print(f"     {'same/diff':12s} " + "  ".join(f"{rates[j, 2, b] / rates[j, 3, b]:.2f}" for b in range(4)), flush=True)
