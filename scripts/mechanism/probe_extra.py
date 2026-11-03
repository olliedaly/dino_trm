"""The probe.py probes (class accuracy, position R^2, size R^2 on each object's best-IoU slot)
for extra models, next to T=1 and T=8 of the same training seed on identical folds.
Reuses the object list in c/probe_assign.npz (same objects, same order as probe.py).
Usage: probe_extra.py C "name=rx file:step,..."   e.g. "prior b=40=rx_trm_t1_lp40.npz:0"
Each model is compared with T=1 / T=8 of the seed its rx file belongs to (suffix _seed1/_seed2).
Writes c/probe_extra.npz."""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

C = sys.argv[1]
S = 336
torch.set_num_threads(1)
EXTRA = {}
for spec in sys.argv[2].split(","):
    name, rest = spec.rsplit("=", 1)
    f, t = rest.rsplit(":", 1)
    EXTRA[name] = (f, int(t))
Z = np.load(f"{C}/probe_assign.npz")
ROWS, BEST, NAMES = Z["rows"], Z["best"], list(Z["names"])
GT = np.load(f"{C}/gt.npz")
INST = GT["inst"]
MASKS = {name: np.load(f"{C}/{f}")["masks"][:, t] for name, (f, t) in EXTRA.items()}


def one(i):
    inst = INST[i].astype(np.int64)
    ids, counts = np.unique(inst[inst > 0], return_counts=True)
    ids = ids[counts >= 256]  # as in probe.py
    if not len(ids):
        return None
    best = []
    for name in EXTRA:
        t = torch.from_numpy(np.array(MASKS[name][i], dtype=np.float32))[None]
        K = t.shape[1]
        p = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)[0].argmax(0).numpy()
        slot_area = np.bincount(p.ravel(), minlength=K)
        best.append([(np.bincount(p[inst == j], minlength=K) / ((inst == j).sum() + slot_area
                      - np.bincount(p[inst == j], minlength=K))).argmax() for j in ids])
    return np.array(best).T


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


def ridge_r2(Xtr, ytr, Xte, yte, alpha=10.0):  # as in probe.py
    A = np.c_[Xtr, np.ones(len(Xtr))]
    reg = alpha * np.eye(A.shape[1])
    reg[-1, -1] = 0
    beta = np.linalg.solve(A.T @ A + reg, A.T @ ytr)
    pred = (np.c_[Xte, np.ones(len(Xte))] @ beta).reshape(len(Xte), -1)
    yte = yte.reshape(len(yte), -1)
    return np.mean([1 - ((yte[:, k] - pred[:, k]) ** 2).sum() / ((yte[:, k] - yte[:, k].mean()) ** 2).sum()
                    for k in range(yte.shape[1])])


if __name__ == "__main__":
    with Pool(8) as pool:
        out = [o for o in pool.map(one, range(len(INST)), chunksize=8) if o is not None]
    best_x = np.concatenate(out)
    assert len(best_x) == len(ROWS), (len(best_x), len(ROWS))
    MASKS.clear()
    torch.set_num_threads(8)
    img, cls, pos, size = ROWS[:, 0].astype(int), ROWS[:, 1].astype(int), ROWS[:, 2:4], ROWS[:, 4]
    keep = np.isin(cls, [c for c, n in zip(*np.unique(cls, return_counts=True)) if n >= 40])
    cmap = {c: k for k, c in enumerate(np.unique(cls[keep]))}
    folds = list(GroupKFold(5).split(ROWS, groups=img))

    def probe(f, t, best):
        X = np.load(f"{C}/{f}")["slots"][:, t].astype(np.float32)[img, best]
        acc, pr, sr = [], [], []
        for tr, te in folds:
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            ytr = np.array([cmap[c] for c in cls[tr][keep[tr]]])
            yte = np.array([cmap[c] for c in cls[te][keep[te]]])
            acc.append((softmax_probe(Xtr[keep[tr]], ytr, Xte[keep[te]], len(cmap)) == yte).mean())
            pr.append(ridge_r2(Xtr, pos[tr], Xte, pos[te]))
            sr.append(ridge_r2(Xtr, size[tr], Xte, size[te]))
        return np.mean(acc), np.mean(pr), np.mean(sr)

    res, done = {}, {}
    for j, (name, (f, t)) in enumerate(EXTRA.items()):
        s = next((x for x in ("_seed1", "_seed2") if x in f), "")
        for ref, rf, rt in ((f"T=1{s}", f"rx_trm_t1{s}.npz", 0), (f"T=8{s}", f"rx_trm{s}.npz", 7)):
            if ref not in done:
                done[ref] = probe(rf, rt, BEST[:, NAMES.index(ref)])
                print(f"  {ref:16s} class acc {done[ref][0]:.3f}  position R2 {done[ref][1]:.3f}  size R2 {done[ref][2]:.3f}", flush=True)
        res[name] = probe(f, t, best_x[:, j])
        print(f"  {name:16s} class acc {res[name][0]:.3f}  position R2 {res[name][1]:.3f}  size R2 {res[name][2]:.3f}", flush=True)
    res.update(done)
    np.savez(f"{C}/probe_extra.npz", names=list(res), vals=np.array(list(res.values())))
