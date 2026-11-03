"""What do the slots know about their objects? Linear probes (eval seed 0, all val images).
Each GT object (>= 1 patch) takes the slot whose decoder mask has the best IoU with it.
Probes on that slot vector, 5-fold CV grouped by image, features standardised per fold:
  class      softmax regression (L-BFGS, weight decay 1e-3) over classes with >= 40 objects (accuracy)
  position   ridge regression of the object's centroid (x, y in [0, 1]) (R^2)
  size       ridge regression of log object area (R^2)
Also: share of objects whose best slot is shared with another object of the same class."""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

C = sys.argv[1]
S = 336
torch.set_num_threads(1)
SEEDS = ["", "_seed1", "_seed2"]
MODELS = {}
for s in SEEDS:
    MODELS[f"baseline{s}"] = (f"rx_baseline_pm{s}.npz", 0)
    MODELS[f"mlp_block{s}"] = (f"rx_mlp_block{s}.npz", 0)
    MODELS[f"T=1{s}"] = (f"rx_trm_t1{s}.npz", 0)
    MODELS[f"T=8 step1{s}"] = (f"rx_trm{s}.npz", 0)
    MODELS[f"T=8{s}"] = (f"rx_trm{s}.npz", 7)
MODELS["sa_rec_1step"] = ("rx_sa_rec_1step.npz", 7)
GT = np.load(f"{C}/gt.npz")
INST, SEM = GT["inst"], GT["sem"]
MASKS = {}
for name, (f, t) in MODELS.items():
    MASKS[name] = np.load(f"{C}/{f}")["masks"][:, t]


def one(i):
    inst, sem = INST[i].astype(np.int64), SEM[i].astype(np.int64)
    ids, counts = np.unique(inst[inst > 0], return_counts=True)
    ids = ids[counts >= 256]
    if not len(ids):
        return None
    rows = []
    yy, xx = np.mgrid[:S, :S] / S
    for j in ids:
        m = inst == j
        rows.append([i, np.bincount(sem[m]).argmax(), xx[m].mean(), yy[m].mean(), np.log(m.sum())])
    rows = np.array(rows)
    best, shared = [], []
    for name in MODELS:
        t = torch.from_numpy(np.array(MASKS[name][i], dtype=np.float32))[None]
        K = t.shape[1]
        p = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)[0].argmax(0).numpy()
        slot_area = np.bincount(p.ravel(), minlength=K)
        b = []
        for j in ids:
            inter = np.bincount(p[inst == j], minlength=K)
            b.append((inter / ((inst == j).sum() + slot_area - inter)).argmax())
        b = np.array(b)
        cls = rows[:, 1]
        shared.append([((b == b[k]) & (cls == cls[k])).sum() > 1 for k in range(len(b))])
        best.append(b)
    return rows, np.array(best).T, np.array(shared).T


def softmax_probe(Xtr, ytr, Xte, n_cls, wd=1e-3):
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


def ridge_r2(Xtr, ytr, Xte, yte, alpha=10.0):
    A = np.c_[Xtr, np.ones(len(Xtr))]
    reg = alpha * np.eye(A.shape[1])
    reg[-1, -1] = 0
    beta = np.linalg.solve(A.T @ A + reg, A.T @ ytr)
    pred = (np.c_[Xte, np.ones(len(Xte))] @ beta).reshape(len(Xte), -1)
    yte = yte.reshape(len(yte), -1)
    return np.mean([1 - ((yte[:, k] - pred[:, k]) ** 2).sum() / ((yte[:, k] - yte[:, k].mean()) ** 2).sum()
                    for k in range(yte.shape[1])])


if __name__ == "__main__":
    import os
    if os.path.exists(f"{C}/probe_assign.npz"):
        z = np.load(f"{C}/probe_assign.npz")
        rows, best, shared = z["rows"], z["best"], z["shared"]
    else:
        with Pool(11) as pool:
            out = [o for o in pool.map(one, range(len(INST)), chunksize=8) if o is not None]
        rows = np.concatenate([o[0] for o in out])
        best = np.concatenate([o[1] for o in out])
        shared = np.concatenate([o[2] for o in out])
        np.savez(f"{C}/probe_assign.npz", rows=rows, best=best, shared=shared, names=list(MODELS))
    MASKS.clear()
    torch.set_num_threads(8)
    img, cls, pos, size = rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 2:4], rows[:, 4]
    keep_cls = np.isin(cls, [c for c, n in zip(*np.unique(cls, return_counts=True)) if n >= 40])
    print(f"{len(rows)} objects in {len(np.unique(img))} images; class probe on {keep_cls.sum()} objects, "
          f"{len(np.unique(cls[keep_cls]))} classes; majority-class accuracy "
          f"{np.bincount(cls[keep_cls]).max() / keep_cls.sum():.3f}", flush=True)
    folds = list(GroupKFold(5).split(rows, groups=img))
    res = {}
    for j, (name, (f, t)) in enumerate(MODELS.items()):
        slots = np.load(f"{C}/{f}")["slots"][:, t].astype(np.float32)
        X = slots[img, best[:, j]]
        acc, pr, sr = [], [], []
        cmap = {c: k for k, c in enumerate(np.unique(cls[keep_cls]))}
        for tr, te in folds:
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            ktr, kte = keep_cls[tr], keep_cls[te]
            ytr = np.array([cmap[c] for c in cls[tr][ktr]])
            yte = np.array([cmap[c] for c in cls[te][kte]])
            acc.append((softmax_probe(Xtr[ktr], ytr, Xte[kte], len(cmap)) == yte).mean())
            pr.append(ridge_r2(Xtr, pos[tr], Xte, pos[te]))
            sr.append(ridge_r2(Xtr, size[tr], Xte, size[te]))
        res[name] = (np.mean(acc), np.mean(pr), np.mean(sr), shared[:, j].mean())
        print(f"  {name:16s} class acc {np.mean(acc):.3f}  position R2 {np.mean(pr):.3f}  size R2 {np.mean(sr):.3f}  "
              f"objects sharing their slot with a same-class object {shared[:, j].mean():.3f}", flush=True)
    np.savez(f"{C}/probe.npz", names=list(res), vals=np.array(list(res.values())))
