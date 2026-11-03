"""Per-image partition statistics for the mechanism tests (eval seed 0, all val images).

For every partition (model readout, or trm step): FG-ARI, full ARI (background counted as
one GT segment), mBO_i, mBO_c, and pixel-pair rates against GT:
  whole  same-object pairs sharing a slot              (w)
  sameC  pairs from two different same-class objects sharing a slot (all distances)
  farC / nearC / touchC  the same, split by object gap (> 112 px / 1.5-112 / <= 1.5)
  diffC  pairs from two different-class objects sharing a slot
  bgmix  share of an object's slot that is background, averaged over object pixels
  q      foreground pairs sharing a slot
Per image GT 'opportunity': share of foreground pixel pairs that are far same-class pairs.
Writes c/mech2.npz (partitions from the rx_*.npz extractions): key "{partition}|{stat}" -> (2000,) float32, plus "gt|..." keys."""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from slot_shape import score  # noqa: E402

C = sys.argv[1]
S = 336
torch.set_num_threads(1)
GT = np.load(f"{C}/gt.npz")
INST, SEM = GT["inst"], GT["sem"]
PD = np.load(f"{C}/pd_trm.npy")
IMG = PD[:, 0].astype(int)
START = np.r_[0, np.cumsum(np.bincount(IMG, minlength=len(INST)))]
SEEDS = ["", "_seed1", "_seed2"]
PARTS = {}  # eval seeds 1 and 2
for e in (1, 2):
    for s in SEEDS:
        PARTS[f"t1{s}|e{e}"] = (f"rx_trm_t1{s}_e{e}.npz", 0)
        PARTS[f"t8{s}|e{e}"] = (f"rx_trm{s}_e{e}.npz", 7)
    PARTS[f"base|e{e}"] = (f"rx_baseline_pm_e{e}.npz", 0)
ARR = {f: np.load(f"{C}/{f}")["masks"] for f, _ in set(PARTS.values())}


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
    out = {}
    if G:
        cls = np.array([np.bincount(sem[inst == j]).argmax() for j in ids])
        n_g = np.bincount(gi, minlength=G).astype(float)
        a_, b_ = np.triu_indices(G, 1)
        pd = PD[START[i]:START[i + 1]]
        assert len(pd) == len(a_)
        gap = np.zeros((G, G)); gap[a_, b_] = pd[:, 2]; gap = gap + gap.T
        sc = (cls[:, None] == cls[None, :]) & ~np.eye(G, dtype=bool)
        grp = {"sameC": sc, "farC": sc & (gap > 112), "nearC": sc & (gap > 1.5) & (gap <= 112),
               "touchC": sc & (gap <= 1.5), "diffC": cls[:, None] != cls[None, :]}
        pg = np.outer(n_g, n_g)
        tot = {k: pg[m].sum() / 2 for k, m in grp.items()}
        nfg = fg.sum()
        out["gt|opp_far"] = tot["farC"] / (nfg * (nfg - 1) / 2)
        out["gt|n_obj"] = G
        out["gt|has_sameC"] = float(sc.any())
        out["gt|has_farC"] = float(grp["farC"].any())
    for name, (f, j) in PARTS.items():
        m = ARR[f][i, j]
        t = torch.from_numpy(np.array(m, dtype=np.float32))[None]
        K = t.shape[1]
        p = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)[0].argmax(0).numpy()
        r = score(p, inst, sem, K)
        for k, v in r.items():
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
        out[f"{name}|q"] = c2(nk.sum(0)) / (nfg * (nfg - 1) / 2)
        for k, msk in grp.items():
            out[f"{name}|{k}"] = cross[msk].sum() / 2 / tot[k] if tot[k] else np.nan
        bg = np.bincount(p[~fg], minlength=K).astype(float)
        slot_tot = nk.sum(0) + bg
        out[f"{name}|bgmix"] = ((nk * (bg / np.maximum(slot_tot, 1))).sum(1) / n_g).mean()
    return out


if __name__ == "__main__":
    with Pool(11) as pool:
        rows = pool.map(one, range(len(INST)), chunksize=4)
    keys = sorted({k for r in rows for k in r})
    np.savez_compressed(f"{C}/mech3.npz", **{k: np.array([r.get(k, np.nan) for r in rows], np.float32) for k in keys})
    print("mech done", len(keys), "keys", flush=True)
