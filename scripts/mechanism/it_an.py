"""Score the partitions in c/it_{run}.npz (from sa_iter.py), per image.

iters    sa{n}, dec{n} for each test-time iteration count n; agreement: dec{n} vs sa{n},
         and sa{n} vs sa3 (the trained setting)
sa_rec   sa{t}, dec{t} for each trained iteration t, and sa16, dec16 (16 iterations at
         test time); agreement: dec vs sa of the same iteration, sa{t} vs the final sa
coupled  sa{t} (slot-attention call t: 0 = initial, t = the re-binding used at step t) and
         t{t} (decoded output after step t); agreement: t{t} vs sa{t}, and t{t} vs final
Writes c/itan_{run}.npz with keys "{partition}|{metric}".
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from slot_shape import ari, score  # noqa: E402

C, KIND, RUN = sys.argv[1], sys.argv[2], sys.argv[3]
S = 336
torch.set_num_threads(1)
D = dict(np.load(f"{C}/it_{RUN}.npz"))
GT = np.load(f"{C}/gt.npz")
INST, SEM = GT["inst"], GT["sem"]
if KIND == "iters":
    NS = sorted(int(k[5:]) for k in D if k.startswith("attn_"))
    PARTS = {**{f"sa{n}": ("attn_%d" % n, None) for n in NS}, **{f"dec{n}": ("m_%d" % n, None) for n in NS}}
    REFS = {**{f"dec{n}": [f"sa{n}"] for n in NS}, **{f"sa{n}": ["sa3"] for n in NS}}
elif KIND == "sa_rec":
    T = D["m"].shape[1]
    PARTS = {**{f"sa{t}": ("attn", t - 1) for t in range(1, T + 1)},
             **{f"dec{t}": ("m", t - 1) for t in range(1, T + 1)},
             "sa16": ("attn_16", None), "dec16": ("m_16", None)}
    REFS = {**{f"dec{t}": [f"sa{t}"] for t in list(range(1, T + 1)) + [16]},
            **{f"sa{t}": [f"sa{T}"] for t in list(range(1, T + 1)) + [16]}}
else:
    T = D["m"].shape[1]
    PARTS = {**{f"sa{t}": ("attn", t) for t in range(T + 1)}, **{f"t{t}": ("m", t - 1) for t in range(1, T + 1)}}
    REFS = {f"t{t}": [f"sa{t}", f"t{T}"] for t in range(1, T + 1)}
    REFS.update({f"sa{t}": [f"t{T}"] for t in range(T + 1)})


def up(m):
    t = torch.from_numpy(np.asarray(m, np.float32))[None]
    return F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)[0].argmax(0).numpy()


def one(i):
    inst, sem = INST[i].astype(np.int64), SEM[i].astype(np.int64)
    fg = inst > 0
    u, gi = np.unique(inst[fg], return_inverse=True)
    a, b = np.triu_indices(len(u), 1)
    ps = {k: up(D[arr][i] if j is None else D[arr][i, j]) for k, (arr, j) in PARTS.items()}
    K = D[PARTS[next(iter(PARTS))][0]].shape[-3]
    out = {}
    for name, p in ps.items():
        r = score(p, inst, sem, K)
        r["max_share"] = np.bincount(p.ravel(), minlength=K).max() / p.size
        for ref in REFS.get(name, []):
            c = np.bincount(ps[ref].ravel() * K + p.ravel(), minlength=K * K).reshape(K, K)
            r[f"ari_{ref}"] = ari(c)
        n = np.bincount(gi * K + p[fg], minlength=len(u) * K).reshape(len(u), K).astype(float)
        P = n / n.sum(1, keepdims=True)
        r["pairs"] = (P @ P.T)[a, b]
        out[name] = r
    return out


if __name__ == "__main__":
    with Pool(11) as pool:
        rows = pool.map(one, range(len(INST)), chunksize=8)
    save = {}
    for v in rows[0]:
        for k in rows[0][v]:
            if k == "pairs":
                save[f"{v}|pairs"] = np.concatenate([r[v]["pairs"] for r in rows]).astype(np.float32)
            else:
                save[f"{v}|{k}"] = np.array([r[v][k] for r in rows], np.float32)
    np.savez_compressed(f"{C}/itan_{RUN}.npz", **save)
    print(RUN, "done", flush=True)
