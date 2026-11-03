"""Is the probe shift (less class/size, more position) only a by-product of separating
same-class objects? Re-run the probe.py probes for T=1 and T=8 (3 seeds) keeping the
out-of-fold prediction for every object, then split the objects by whether their class
occurs elsewhere in the image. For objects that are the only one of their class, separating
same-class objects cannot change what their slot holds.
Writes c/probe_oof.npz and prints, per subset: class accuracy, position R^2 and size R^2
(within the subset), T=8 - T=1 averaged over the 3 seed pairs with an image-bootstrap CI."""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

C = sys.argv[1]


def softmax_probe(Xtr, ytr, Xte, n_cls, wd=1e-3):  # as in probe.py
    Xtr, Xte = torch.from_numpy(Xtr), torch.from_numpy(Xte)
    W = torch.zeros(Xtr.shape[1], n_cls, requires_grad=True)
    b = torch.zeros(n_cls, requires_grad=True)
    y = torch.from_numpy(ytr)
    opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xtr @ W + b, y) + wd * W.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return (Xte @ W + b).argmax(1).numpy()


z = np.load(f"{C}/probe_assign.npz")
rows, best, names = z["rows"], z["best"], list(z["names"])
GT = np.load(f"{C}/gt.npz")
INST, SEM = GT["inst"], GT["sem"]
img, cls, pos, size = rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 2:4], rows[:, 4]

# Rebuild the object order of probe.py (ids >= 256 px, sorted) to flag lone-class objects.
alone = []
for i in np.unique(img):
    inst, sem = INST[i].astype(np.int64), SEM[i].astype(np.int64)
    ids, counts = np.unique(inst[inst > 0], return_counts=True)
    ocls = np.array([np.bincount(sem[inst == j]).argmax() for j in ids])
    for j, c in zip(ids, counts):
        if c >= 256:
            alone.append((ocls == ocls[ids == j][0]).sum() == 1)
alone = np.array(alone)
assert len(alone) == len(rows)
torch.set_num_threads(8)
keep = np.isin(cls, [c for c, n in zip(*np.unique(cls, return_counts=True)) if n >= 40])
cmap = {c: k for k, c in enumerate(np.unique(cls[keep]))}
folds = list(GroupKFold(5).split(rows, groups=img))


def ridge_pred(Xtr, ytr, Xte, alpha=10.0):
    A = np.c_[Xtr, np.ones(len(Xtr))]
    reg = alpha * np.eye(A.shape[1])
    reg[-1, -1] = 0
    return np.c_[Xte, np.ones(len(Xte))] @ np.linalg.solve(A.T @ A + reg, A.T @ ytr)


oof = {}
for s in ["", "_seed1", "_seed2"]:
    for m, f in (("T=1", f"rx_trm_t1{s}.npz"), ("T=8", f"rx_trm{s}.npz")):
        name = m + s
        slots = np.load(f"{C}/{f}")["slots"][:, -1].astype(np.float32)
        X = slots[img, best[:, names.index(name)]]
        cor = np.full(len(rows), np.nan)
        pp = np.zeros_like(pos)
        sp = np.zeros_like(size)
        for tr, te in folds:
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            ktr, kte = keep[tr], keep[te]
            ytr = np.array([cmap[c] for c in cls[tr][ktr]])
            yte = np.array([cmap[c] for c in cls[te][kte]])
            cor[te[kte]] = softmax_probe(Xtr[ktr], ytr, Xte[kte], len(cmap)) == yte
            pp[te] = ridge_pred(Xtr, pos[tr], Xte)
            sp[te] = ridge_pred(Xtr, size[tr], Xte)
        oof[name] = (cor, pp, sp)
        print(name, "class", np.nanmean(cor).round(3), flush=True)
np.savez(f"{C}/probe_oof.npz", alone=alone, keep=keep,
         **{f"{k}|{j}": v for k, t in oof.items() for j, v in zip(("cor", "pos", "size"), t)})


def stats(name, sel):
    cor, pp, sp = oof[name]
    k = sel & keep
    r2p = np.mean([1 - ((pos[sel, d] - pp[sel, d]) ** 2).sum() / ((pos[sel, d] - pos[sel, d].mean()) ** 2).sum() for d in (0, 1)])
    r2s = 1 - ((size[sel] - sp[sel]) ** 2).sum() / ((size[sel] - size[sel].mean()) ** 2).sum()
    return np.array([cor[k].mean(), r2p, r2s])


def diff(sel):
    return np.mean([stats("T=8" + s, sel) - stats("T=1" + s, sel) for s in ["", "_seed1", "_seed2"]], 0)


rng = np.random.default_rng(0)
U = np.unique(img)
by_img = {i: np.flatnonzero(img == i) for i in U}
for label, sel in (("all objects", np.ones(len(rows), bool)), ("only one of its class in the image", alone),
                   ("has a same-class object in the image", ~alone)):
    t1 = np.mean([stats("T=1" + s, sel) for s in ["", "_seed1", "_seed2"]], 0)
    t8 = np.mean([stats("T=8" + s, sel) for s in ["", "_seed1", "_seed2"]], 0)
    d = t8 - t1
    bs = []
    for _ in range(200):
        idx = np.concatenate([by_img[i] for i in rng.choice(U, len(U))])
        m = np.zeros(len(rows), int)
        np.add.at(m, idx, 1)
        # weighted resample: duplicate objects by their bootstrap multiplicity
        rep = np.repeat(np.arange(len(rows)), m)
        bsel = sel[rep]
        vals = []
        for s in ["", "_seed1", "_seed2"]:
            v = []
            for mm in ("T=8", "T=1"):
                cor, pp, sp = oof[mm + s]
                c, p_, s_ = cor[rep][bsel & keep[rep]], pp[rep][bsel], sp[rep][bsel]
                P, Z = pos[rep][bsel], size[rep][bsel]
                v.append(np.array([c.mean(),
                                   np.mean([1 - ((P[:, q] - p_[:, q]) ** 2).sum() / ((P[:, q] - P[:, q].mean()) ** 2).sum() for q in (0, 1)]),
                                   1 - ((Z - s_) ** 2).sum() / ((Z - Z.mean()) ** 2).sum()]))
            vals.append(v[0] - v[1])
        bs.append(np.mean(vals, 0))
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    print(f"{label} (n={sel.sum()}):")
    for k, nm in enumerate(("class accuracy", "position R2", "size R2")):
        print(f"   {nm:15s} T=1 {t1[k]:.3f}  T=8 {t8[k]:.3f}  diff {d[k]:+.3f} [{lo[k]:+.3f}, {hi[k]:+.3f}]", flush=True)
