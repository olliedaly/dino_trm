"""Per-image stats on VOC val (same definitions as mech.py; object gaps computed here).
Writes c/voc_mech.npz."""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, "scripts")
from slot_shape import score  # noqa: E402

C = sys.argv[1]
S = 336
torch.set_num_threads(1)
GT = np.load(f"{C}/voc_gt.npz")
INST, SEM = GT["inst"], GT["sem"]
TAGS = [f"coco_trm_t1{s}" for s in ["", "_seed1", "_seed2"]] + [f"coco_trm{s}" for s in ["", "_seed1", "_seed2"]] + \
       ["voc_trm_t1_m", "voc_trm_m", "voc_baseline_pm_m"]
M = {t: np.load(f"{C}/voc_{t}.npz")["masks"] for t in TAGS}


def ari_table(n):
    n = n.astype(np.float64)
    c2 = lambda x: (x * (x - 1) / 2).sum()  # noqa: E731
    a, b, t = c2(n.sum(1)), c2(n.sum(0)), c2(n)
    tot = n.sum() * (n.sum() - 1) / 2
    e = a * b / tot
    return (t - e) / (0.5 * (a + b) - e) if (0.5 * (a + b) - e) > 0 else np.nan


def one(i):
    inst, sem = INST[i].astype(np.int64), SEM[i].astype(np.int64)
    fg = inst > 0
    ids, gi = np.unique(inst[fg], return_inverse=True)
    G = len(ids)
    out = {"gt|n_obj": G}
    if G:
        cls = np.array([np.bincount(sem[inst == j]).argmax() for j in ids])
        n_g = np.bincount(gi, minlength=G).astype(float)
        gap = np.zeros((G, G))
        for a in range(G):
            d = distance_transform_edt(inst != ids[a])
            for b in range(a + 1, G):
                gap[a, b] = gap[b, a] = d[inst == ids[b]].min()
        sc = (cls[:, None] == cls[None, :]) & ~np.eye(G, dtype=bool)
        grp = {"sameC": sc, "farC": sc & (gap > 112), "nearC": sc & (gap > 1.5) & (gap <= 112),
               "touchC": sc & (gap <= 1.5), "diffC": cls[:, None] != cls[None, :]}
        pg = np.outer(n_g, n_g)
        tot = {k: pg[m].sum() / 2 for k, m in grp.items()}
        nfg = fg.sum()
        out["gt|opp_far"] = tot["farC"] / (nfg * (nfg - 1) / 2) if nfg > 1 else np.nan
        out["gt|has_sameC"] = float(sc.any())
        out["gt|has_farC"] = float(grp["farC"].any())
    for name in TAGS:
        t = torch.from_numpy(np.array(M[name][i], dtype=np.float32))[None]
        K = t.shape[1]
        p = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)[0].argmax(0).numpy()
        for k, v in score(p, inst, sem, K).items():
            out[f"{name}|{k}"] = v
        allids, ag = np.unique(inst, return_inverse=True)
        out[f"{name}|ari_all"] = ari_table(np.bincount(ag.ravel() * K + p.ravel(), minlength=len(allids) * K).reshape(len(allids), K))
        if not G:
            continue
        nk = np.bincount(gi * K + p[fg], minlength=G * K).reshape(G, K).astype(float)
        cross = nk @ nk.T
        c2 = lambda x: (x * (x - 1) / 2).sum()  # noqa: E731
        A = c2(n_g)
        out[f"{name}|whole"] = c2(nk) / A if A else np.nan
        for k, msk in grp.items():
            out[f"{name}|{k}"] = cross[msk].sum() / 2 / tot[k] if tot[k] else np.nan
        bg = np.bincount(p[~fg], minlength=K).astype(float)
        out[f"{name}|bgmix"] = ((nk * (bg / np.maximum(nk.sum(0) + bg, 1))).sum(1) / n_g).mean()
    return out


if __name__ == "__main__":
    with Pool(11) as pool:
        rows = pool.map(one, range(len(INST)), chunksize=4)
    keys = sorted({k for r in rows for k in r})
    np.savez_compressed(f"{C}/voc_mech.npz", **{k: np.array([r.get(k, np.nan) for r in rows], np.float32) for k in keys})
    print("voc mech done", len(rows), "images", len(keys), "keys")
