"""Per image: each partition of a run scored, and compared with slot attention's own one.

partitions: sa (slot attention's attention), mx (decoder applied to slot attention's slots),
t1..tT (decoder masks after each module step), noca (cross-attention updates zeroed).
For each: fg_ari, mbo_i, mbo_c, max_share, object-pair sharing (pd order), and agreement
with the final partition and with sa (ARI over all pixels, and same-slot-index share).
Writes c/an_{run}.npz with keys "{partition}|{metric}".
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from slot_shape import ari, score  # noqa: E402

C, RUN = sys.argv[1], sys.argv[2]
S = 336
torch.set_num_threads(1)
D = dict(np.load(f"{C}/sa_{RUN}.npz"))
T, K = D["m"].shape[1:3]
GT = np.load(f"{C}/gt.npz")
INST, SEM = GT["inst"], GT["sem"]


def up(m):
    t = torch.from_numpy(np.asarray(m, np.float32))[None]
    return F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)[0].argmax(0).numpy()


def agree(a, b):
    c = np.bincount(a.ravel() * K + b.ravel(), minlength=K * K).reshape(K, K)
    return ari(c), (a == b).mean()


def one(i):
    inst, sem = INST[i].astype(np.int64), SEM[i].astype(np.int64)
    fg = inst > 0
    u, gi = np.unique(inst[fg], return_inverse=True)
    a, b = np.triu_indices(len(u), 1)
    parts = {"sa": up(D["attn"][i]), "mx": up(D["mx"][i])}
    parts.update({f"t{t + 1}": up(D["m"][i, t]) for t in range(T)})
    if "m_noca" in D:
        parts["noca"] = up(D["m_noca"][i])
    fin = parts[f"t{T}"]
    out = {}
    for name, p in parts.items():
        r = score(p, inst, sem, K)
        r["max_share"] = np.bincount(p.ravel(), minlength=K).max() / p.size
        r["ari_fin"], r["idx_fin"] = agree(fin, p)
        r["ari_sa"], r["idx_sa"] = agree(parts["sa"], p)
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
    x, y = D["x"].astype(np.float32), D["y"].astype(np.float32)          # (N,K,D), (N,T,K,D)
    save["slot|rel_move"] = (np.linalg.norm(y - x[:, None], axis=-1)
                             / np.linalg.norm(x, axis=-1)[:, None]).mean(-1)   # (N, T)
    save["slot|cos"] = ((y * x[:, None]).sum(-1) / np.linalg.norm(y, axis=-1)
                        / np.linalg.norm(x, axis=-1)[:, None]).mean(-1)
    np.savez_compressed(f"{C}/an_{RUN}.npz", **save)
    print(RUN, "done", flush=True)
