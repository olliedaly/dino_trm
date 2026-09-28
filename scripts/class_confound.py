"""Is the "object size" effect actually an object CLASS effect?

In COCO, size and category are confounded: small objects are disproportionately
bottles, birds, cups and traffic lights; large ones are people, trains, beds and
dining tables. Every size result so far controlled for the NUMBER of classes
(`n_cls`) but never for class IDENTITY — so "recursion helps small objects" and
"recursion helps the categories that happen to be small" are not yet separable.

Object count turned out to be a proxy for object size; class could be one too, so it
is tested directly rather than assumed.

Tests:
  C1  within-class trends  — the size trend computed INSIDE images dominated by a
                             single frequent category. If size is the driver it
                             survives within "person" images, within "car" images,
                             and so on. If it is class, it vanishes there.
  C2  class-demeaned       — subtract each dominant class's mean delta and mean size
                             rank, then correlate the residuals. Removes all
                             between-class variance in one step.
  C3  class composition    — how strongly size and class are actually tied, to show
                             the confound is real rather than hypothetical.

Run:  uv run python scripts/class_confound.py
"""

from __future__ import annotations

import os

import numpy as np
from scipy import stats

from size_profile import load

PI = "results/coco/per_image"
T1 = "trm_t1+trm_t1_seed1+trm_t1_seed2"
T8 = "trm+trm_seed1+trm_seed2"
CACHE = os.path.join(PI, "dominant_class.npz")


def build_class_info() -> dict:
    """Per-image dominant category (by visible area in the eval crop) + its share."""
    from pycocotools.coco import COCO
    from PIL import Image
    from dino_trm.data.coco import _resize_crop

    root = "data/coco"
    coco = COCO(os.path.join(root, "annotations", "instances_val2017.json"))
    ids = np.load(os.path.join(root, "val_subset_ids.npy")).tolist()
    dom, share, names = [], [], {}
    for c in coco.loadCats(coco.getCatIds()):
        names[c["id"]] = c["name"]

    for img_id in ids:
        meta = coco.loadImgs(int(img_id))[0]
        h, w = meta["height"], meta["width"]
        sem = np.zeros((h, w), dtype=np.int32)
        for a in sorted(coco.loadAnns(coco.getAnnIds(imgIds=int(img_id), iscrowd=False)),
                        key=lambda a: -a["area"]):
            m = coco.annToMask(a).astype(bool)
            if m.shape == (h, w) and m.any():
                sem[m] = a["category_id"]
        sem_c = np.asarray(_resize_crop(Image.fromarray(sem, mode="I"), 336, Image.NEAREST))
        cls, cnt = np.unique(sem_c[sem_c != 0], return_counts=True)
        if cls.size:
            dom.append(int(cls[cnt.argmax()])); share.append(float(cnt.max() / cnt.sum()))
        else:
            dom.append(0); share.append(0.0)
    return {"dom": np.array(dom), "share": np.array(share),
            "cat_ids": np.array(list(names)), "cat_names": np.array(list(names.values()))}


def main() -> None:
    if not os.path.exists(CACHE):
        print("building per-image dominant-class info ...")
        np.savez(CACHE, **build_class_info())
    ci = dict(np.load(CACHE, allow_pickle=True))
    name = {int(i): str(n) for i, n in zip(ci["cat_ids"], ci["cat_names"])}

    cov = dict(np.load(os.path.join(PI, "covariates.npz")))
    size = cov["mean_obj_size"].astype(float)
    d = load(PI, T8, "fg_ari", 3) - load(PI, T1, "fg_ari", 3)
    k = ~np.isnan(d) & (size > 0)
    dom = ci["dom"]

    print(f"n={k.sum()}   overall rho(delta, size) = "
          f"{stats.spearmanr(d[k], size[k]).statistic:+.4f}\n")

    print("=" * 72)
    print("C3  IS THE CONFOUND REAL? mean object size by dominant class")
    print("=" * 72)
    freq = [(c, int((dom == c).sum())) for c in np.unique(dom) if c != 0]
    freq = sorted(freq, key=lambda t: -t[1])[:12]
    for c, n in freq:
        m = (dom == c) & k
        print(f"  {name.get(c,'?'):16s} n={n:4d}   mean size {size[m].mean()/256:6.1f} patches"
              f"   mean delta {100*d[m].mean():+6.2f}")
    sizes_by_class = [size[(dom == c) & k].mean() for c, _ in freq]
    print(f"\n  spread of class mean sizes: {min(sizes_by_class)/256:.1f} to "
          f"{max(sizes_by_class)/256:.1f} patches — the confound is real.\n")

    print("=" * 72)
    print("C1  SIZE TREND WITHIN EACH DOMINANT CLASS (>=80 images)")
    print("=" * 72)
    print(f"  {'class':>16s}{'n':>6s}{'rho':>10s}{'p':>10s}{'delta':>9s}")
    rows = []
    for c, n in [(c, n) for c, n in freq if n >= 80]:
        m = (dom == c) & k
        r = stats.spearmanr(d[m], size[m])
        rows.append(r.statistic)
        print(f"  {name.get(c,'?'):>16s}{int(m.sum()):6d}{r.statistic:+10.4f}"
              f"{r.pvalue:10.3f}{100*d[m].mean():+9.2f}")
    if rows:
        print(f"\n  {sum(1 for r in rows if r < 0)}/{len(rows)} classes negative; "
              f"median within-class rho = {np.median(rows):+.4f}")

    print("\n" + "=" * 72)
    print("C2  CLASS-DEMEANED (all between-class variance removed)")
    print("=" * 72)
    rd, rs = stats.rankdata(d[k]), stats.rankdata(size[k])
    dk = dom[k]
    for arr in (rd, rs):
        for c in np.unique(dk):
            m = dk == c
            if m.sum() > 1:
                arr[m] -= arr[m].mean()
    r = stats.pearsonr(rd, rs)
    print(f"  residual correlation after removing each class's mean: "
          f"rho={r[0]:+.4f}  p={r[1]:.2e}")
    print(f"  raw:                                                  "
          f"rho={stats.spearmanr(d[k], size[k]).statistic:+.4f}")


if __name__ == "__main__":
    main()
