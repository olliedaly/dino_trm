"""Where does each slot's cross-attention read from? Token-level (21x21), per run.

For every slot whose final region holds foreground, its main object is the GT instance it
overlaps most. Each token is then: own object, same class near (gap <= 7 tokens), same class
far, different class, background. We sum the slot's head-averaged cross-attention over each
group (mean over the step's blocks), for the first and last step, and also record each
group's area share (what uniform attention would give).
Writes c/ca_{run}.npz: mass (n_slots, 2, 5), area (n_slots, 5), own_sa (n_slots, 2).
"""
import sys
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import distance_transform_edt

C, RUN = sys.argv[1], sys.argv[2]
D = dict(np.load(f"{C}/sa_{RUN}.npz"))
GT = np.load(f"{C}/gt.npz")
INST, SEM = GT["inst"], GT["sem"]
g = D["attn"].shape[-1]
FAR = 7


def major(lbl):
    p = lbl.reshape(g, 16, g, 16).transpose(0, 2, 1, 3).reshape(g * g, 256)
    ids, inv = np.unique(p, return_inverse=True)
    cnt = np.zeros((g * g, len(ids)), np.int32)
    np.add.at(cnt, (np.repeat(np.arange(g * g), 256), inv.ravel()), 1)
    return ids[cnt.argmax(1)]


def one(i):
    inst = INST[i].astype(np.int64)
    it = major(inst)                                   # (g*g,) instance per token
    sem = SEM[i].astype(np.int64)
    fin = D["m"][i, -1].astype(np.float32).reshape(-1, g * g).argmax(0)
    sa = D["attn"][i].astype(np.float32).reshape(-1, g * g).argmax(0)
    ca = D["ca"][i].astype(np.float32).mean(1)         # (2, K, g*g) mean over blocks
    cls = {j: np.bincount(sem[inst == j]).argmax() for j in np.unique(it) if j}
    rows = []
    for k in range(ca.shape[1]):
        reg = (fin == k) & (it > 0)
        if not reg.any():
            continue
        o = np.bincount(it[reg]).argmax()
        d = distance_transform_edt((it != o).reshape(g, g)).ravel()
        grp = np.full(g * g, 4)                        # background
        for j, c in cls.items():
            if j == o:
                grp[it == j] = 0
            elif c == cls[o]:
                grp[it == j] = 1 if d[it == j].min() <= FAR else 2
            else:
                grp[it == j] = 3
        mass = np.stack([np.bincount(grp, weights=ca[s, k], minlength=5) for s in (0, 1)])
        area = np.bincount(grp, minlength=5) / grp.size
        own_sa = ca[:, k][:, sa == k].sum(1)
        rows.append((mass, area, own_sa))
    return rows


if __name__ == "__main__":
    with Pool(11) as pool:
        rows = [r for rs in pool.map(one, range(len(INST)), chunksize=8) for r in rs]
    np.savez_compressed(f"{C}/ca_{RUN}.npz", mass=np.stack([r[0] for r in rows]),
                        area=np.stack([r[1] for r in rows]), own_sa=np.stack([r[2] for r in rows]))
    print(RUN, "slots", len(rows), flush=True)
