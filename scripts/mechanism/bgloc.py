"""Is T=8's locality specific to objects, or general? Token-level (21x21) slot sharing by
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


RUNS = {"T=1": [(f"rx_trm_t1{s}.npz", -1) for s in SEEDS], "T=8": [(f"rx_trm{s}.npz", -1) for s in SEEDS],
        "baseline": [(f"rx_baseline_pm{s}.npz", -1) for s in SEEDS], "mlp_block": [(f"rx_mlp_block{s}.npz", -1) for s in SEEDS]}
if len(sys.argv) > 2 and sys.argv[2]:
    # extra runs as name=pattern, {s} = seed suffix ("", "_seed1", "_seed2"); a bare name
    # means rx_trm_t1{s}_{name}.npz (the loc_mse.py variants). Missing seeds are skipped.
    for extra in sys.argv[2].split(","):
        name, pat = extra.split("=") if "=" in extra else (extra, f"trm_t1{{s}}_{extra}")
        RUNS[name] = [(f"rx_{pat.format(s=s)}.npz", -1) for s in SEEDS if os.path.exists(f"{C}/rx_{pat.format(s=s)}.npz")]
res, per_seed = {}, {}
for name, files in RUNS.items():
    rates = []
    for f, t in files:
        types, s, n = profile(labels_of(f, t))
        rates.append(s / np.maximum(n, 1))
    per_seed[name] = np.array(rates)
    res[name] = np.mean(rates, 0)
    print(f"{name}:", flush=True)
    for k, ty in enumerate(types):
        print(f"   {ty:12s} " + "  ".join(f"{lo}-{hi if hi < 99 else ''}: {res[name][k, j]:.3f}" for j, (lo, hi) in enumerate(BINS)), flush=True)
print("T=8 - T=1 (relative):")
for k, ty in enumerate(types):
    print(f"   {ty:12s} " + "  ".join(f"{(res['T=8'][k, j] / res['T=1'][k, j] - 1) * 100:+5.0f}%" for j in range(len(BINS))))
for name in RUNS:
    if name not in ("T=1", "T=8", "baseline", "mlp_block"):
        print(f"{name} ({len(RUNS[name])} seeds) vs T=1 / vs T=8 (relative, mean of seeds):")
        for k, ty in enumerate(types):
            print(f"   {ty:12s} " + "  ".join(f"{(res[name][k, j] / res['T=1'][k, j] - 1) * 100:+5.0f}% / {(res[name][k, j] / res['T=8'][k, j] - 1) * 100:+5.0f}%" for j in range(len(BINS))))
print("T=8 - T=1 per seed pair (relative):")
for k, ty in enumerate(types):
    r = per_seed["T=8"][:, k] / per_seed["T=1"][:, k] - 1
    print(f"   {ty:12s} " + "  ".join("/".join(f"{100 * x:+.0f}" for x in r[:, j]) + "%" for j in range(len(BINS))))
np.savez(f"{C}/bgloc{os.environ.get('BGLOC_TAG', '')}.npz", names=list(res), vals=np.array(list(res.values())),
         **{f"seeds|{k}": v for k, v in per_seed.items()})
