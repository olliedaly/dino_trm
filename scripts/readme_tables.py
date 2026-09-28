"""Print every number quoted in README.md, straight from the stored per-image vectors.

    uv run python scripts/readme_tables.py

Inputs are ``results/coco/per_image/*.npz`` and ``results/per_image/*.npz`` (VOC),
written by ``scripts/extract_per_image.py``: for each run, per-image mBO_i / mBO_c /
FG-ARI on the full val set under 3 evaluation seeds (slot init is random at inference).
Nothing here reads a hand-written summary. Parameter counts are read from the
checkpoints when ``checkpoints/`` is present (it is not in the repo).

Conventions: a run's score is the mean over images, averaged over its 3 eval seeds.
"±" and "[min, max]" are over TRAINING seeds (or training-seed pairings), which is
the variation that matters; eval-seed noise is ~10x smaller and is printed separately.
"""

from __future__ import annotations

import glob
import itertools
import os

import numpy as np
from scipy.stats import spearmanr

COCO, VOC = "results/coco/per_image/", "results/per_image/"
M = ("mbo_i", "mbo_c", "fg_ari")
ARMS = {  # 3 training seeds each
    "baseline_pm": ["baseline_pm", "baseline_pm_seed1", "baseline_pm_seed2"],
    "mlp_block": ["mlp_block", "mlp_block_seed1", "mlp_block_seed2"],
    "trm_t1": ["trm_t1", "trm_t1_seed1", "trm_t1_seed2"],
    "trm": ["trm", "trm_seed1", "trm_seed2"],
}
SINGLE = ["baseline", "baseline_s7_seed1", "baseline_s9", "baseline_s11", "baseline_s15",
          "baseline_dec2048", "baseline_dinosaur", "baseline_pm13", "trm_s11", "coupled",
          "coupled_dec2048", "trm_t2", "trm_t4"]


def load(path: str, run: str, metric: str) -> np.ndarray:
    """(3 eval seeds, n images)"""
    d = np.load(f"{path}{run}.npz")
    return np.stack([d[f"{metric}_s{s}"] for s in range(3)])


def img(path, run, metric):            # per-image, eval-seed averaged
    return np.nanmean(load(path, run, metric), axis=0)


def agg(path, run, metric):            # scalar score, in points
    return 100 * np.nanmean(img(path, run, metric))


def params(run: str, ckdir: str = "checkpoints/coco") -> str:
    paths = glob.glob(f"{ckdir}/{run}_epoch*.pt")
    if not paths:
        return "   -"
    import torch
    sd = torch.load(paths[0], map_location="cpu", weights_only=False)["model"]
    n = sum(v.numel() for k, v in sd.items() if not k.startswith("backbone."))
    return f"{n / 1e6:5.2f}M"


def hdr(t: str) -> None:
    print(f"\n{'=' * 88}\n{t}\n{'=' * 88}")


def main() -> None:
    cov = np.load(COCO + "covariates.npz")
    size = cov["mean_obj_size"] / 256.0
    n_inst = cov["n_inst"]
    qedges = np.quantile(size, [0.2, 0.4, 0.6, 0.8])
    qi = np.searchsorted(qedges, size)

    hdr("1. COCO, 3 training seeds per arm: mean ± sd over training seeds")
    print(f"{'arm':12s} {'params':>7s}  {'mBO_i':>12s} {'mBO_c':>12s} {'FG-ARI':>12s}")
    for arm, runs in ARMS.items():
        cells = []
        for m in M:
            v = [agg(COCO, r, m) for r in runs]
            cells.append(f"{np.mean(v):5.2f} ± {np.std(v, ddof=1):4.2f}")
        print(f"{arm:12s} {params(runs[0]):>7s}  " + " ".join(f"{c:>12s}" for c in cells))
        print(f"{'':12s} {'':>7s}  per seed: " + "; ".join(
            " / ".join(f"{agg(COCO, r, m):.2f}" for r in runs) for m in M))
    ev = [100 * np.nanmean(load(COCO, r, "fg_ari"), axis=1).std()
          for rs in ARMS.values() for r in rs]
    print(f"eval-seed sd of FG-ARI (within one trained model): median {np.median(ev):.2f}")

    hdr("2. Ingredient deltas: mean over the 9 training-seed pairings [min, max]")
    for name, a, b in [("add block, T=1       (baseline_pm -> trm_t1)", "baseline_pm", "trm_t1"),
                       ("recursion T=1 -> T=8 (trm_t1 -> trm)       ", "trm_t1", "trm"),
                       ("add non-recursive block (baseline_pm -> mlp_block)", "baseline_pm",
                        "mlp_block")]:
        cells = []
        for m in M:
            d = [agg(COCO, y, m) - agg(COCO, x, m)
                 for x, y in itertools.product(ARMS[a], ARMS[b])]
            cells.append(f"{np.mean(d):+5.2f} [{min(d):+.2f},{max(d):+.2f}]")
        print(f"{name}\n    mBO_i {cells[0]}   mBO_c {cells[1]}   FG-ARI {cells[2]}")

    hdr("3. T=1 -> T=8 by object-size quintile (mean object size, 16x16 patches)")
    q = np.quantile(size, [0, .2, .4, .6, .8, 1])
    print("quintile edges:", " | ".join(f"{q[k]:.1f}-{q[k+1]:.1f}" for k in range(5)))
    print("mean GT instances per image:",
          "  ".join(f"Q{k+1} {n_inst[qi == k].mean():.1f}" for k in range(5)))
    nulls = list(itertools.combinations(ARMS["trm_t1"], 2)) + \
        list(itertools.combinations(ARMS["trm"], 2))
    for m in M:
        prof = np.array([[np.nanmean(100 * (img(COCO, y, m) - img(COCO, x, m))[qi == k])
                          for k in range(5)]
                         for x, y in itertools.product(ARMS["trm_t1"], ARMS["trm"])])
        nul = np.array([[np.nanmean(100 * (img(COCO, y, m) - img(COCO, x, m))[qi == k])
                         for k in range(5)] for x, y in nulls])
        rhos = [spearmanr(size, img(COCO, y, m) - img(COCO, x, m), nan_policy="omit")[0]
                for x, y in itertools.product(ARMS["trm_t1"], ARMS["trm"])]
        nrho = [spearmanr(size, img(COCO, y, m) - img(COCO, x, m), nan_policy="omit")[0]
                for x, y in nulls]
        print(f"\n{m}:")
        print("  T1->T8 mean   " + "  ".join(f"Q{k+1} {prof[:, k].mean():+5.2f}"
                                             for k in range(5)))
        print("  T1->T8 range  " + "  ".join(f"[{prof[:, k].min():+.2f},{prof[:, k].max():+.2f}]"
                                             for k in range(5)))
        print("  null max|Δ|   " + "  ".join(f"Q{k+1} {np.abs(nul[:, k]).max():5.2f}"
                                             for k in range(5)))
        print(f"  size-rho per pairing [{min(rhos):+.3f}, {max(rhos):+.3f}]   "
              f"null (same config, different seed) [{min(nrho):+.3f}, {max(nrho):+.3f}]")
    same_sign = sum(
        np.sign(np.nanmean(100 * (img(COCO, y, "fg_ari") - img(COCO, x, "fg_ari"))[qi == k]))
        == (1 if k < 2 else -1)
        for x, y in itertools.product(ARMS["trm_t1"], ARMS["trm"]) for k in (0, 1, 3, 4))
    print(f"\nFG-ARI: pairings x bands (Q1,Q2 positive; Q4,Q5 negative) with expected sign: "
          f"{same_sign}/36")

    hdr("4. Dose response (T=2, T=4 single training seed) vs T=1 averaged over 3 seeds")
    t1 = np.mean([img(COCO, r, "fg_ari") for r in ARMS["trm_t1"]], 0)
    for r in ["trm_t2", "trm_t4"] + [ARMS["trm"]]:
        v = np.mean([img(COCO, x, "fg_ari") for x in r], 0) if isinstance(r, list) \
            else img(COCO, r, "fg_ari")
        d = 100 * (v - t1)
        name = "trm (T=8, 3 seeds)" if isinstance(r, list) else r
        print(f"  {name:20s} " + "  ".join(f"Q{k+1} {d[qi == k].mean():+5.2f}"
                                          for k in range(5))
              + f"   all {d.mean():+.2f}   rho {spearmanr(size, d)[0]:+.3f}")

    hdr("5. Is it object SIZE or object COUNT? (T=1 -> T=8, seed-averaged)")
    d = np.mean([img(COCO, r, "fg_ari") for r in ARMS["trm"]], 0) - t1
    print(f"  rho(size, n_inst) = {spearmanr(size, n_inst)[0]:+.3f}")
    for name, strat, other in [("within size terciles, trend vs count", size, n_inst),
                               ("within count terciles, trend vs size", n_inst, size)]:
        b = np.searchsorted(np.quantile(strat, [1 / 3, 2 / 3]), strat)
        out = []
        for k in range(3):
            r, p = spearmanr(other[b == k], d[b == k])
            out.append(f"{r:+.3f} (p={p:.2g})")
        print(f"  {name}: " + ", ".join(out))

    hdr("5b. Upper bound for per-image depth choice (oracle over T=1 / T=8, FG-ARI)")
    print("  select the better model on eval seed 0, score on eval seeds 1-2 (cross-validated);")
    print("  'naive' selects and scores on the same draws, which inflates it with eval noise")
    for a, b in zip(ARMS["trm_t1"], ARMS["trm"]):
        A, B = load(COCO, a, "fg_ari"), load(COCO, b, "fg_ari")
        cv = np.where(B[0] > A[0], B[1:].mean(0), A[1:].mean(0)).mean()
        naive = np.maximum(A.mean(0), B.mean(0)).mean()
        print(f"  {b:10s} T=1 {100*A[1:].mean():.2f}  T=8 {100*B[1:].mean():.2f}  "
              f"oracle cross-validated {100*cv:.2f}  naive {100*naive:.2f}")

    hdr("6. Single-training-seed runs (mean over 3 eval seeds)")
    print(f"{'run':20s} {'params':>7s} {'mBO_i':>7s} {'mBO_c':>7s} {'FG-ARI':>7s}")
    for r in SINGLE:
        if not os.path.exists(f"{COCO}{r}.npz"):
            print(f"{r:20s}  (no per-image file)")
            continue
        print(f"{r:20s} {params(r):>7s} " + " ".join(f"{agg(COCO, r, m):7.2f}" for m in M))

    hdr("7. VOC (same recipe as COCO, 1 training seed each) + matched absolute size bands")
    vcov = np.load(VOC + "covariates.npz")
    vsize = vcov["mean_obj_size"] / 256.0
    for r in ["baseline_pm_m", "trm_t1_m", "trm_m"]:
        print(f"{r:20s} " + " ".join(f"{m} {agg(VOC, r, m):6.2f}" for m in M))
    edges = [0, 8, 16, 32, 64, np.inf]
    lab = ["<8", "8-16", "16-32", "32-64", "64+"]
    for name, path, sz, a, b in [
            ("COCO trm_t1 -> trm (3x3 seed-avg)", COCO, size, ARMS["trm_t1"], ARMS["trm"]),
            ("VOC  trm_t1_m -> trm_m (1 seed)  ", VOC, vsize, ["trm_t1_m"], ["trm_m"])]:
        for m in ("fg_ari", "mbo_i"):
            dd = 100 * (np.mean([img(path, y, m) for y in b], 0)
                        - np.mean([img(path, x, m) for x in a], 0))
            bi = np.digitize(sz, edges) - 1
            print(f"  {name} {m:6s} " + "  ".join(
                f"{lab[k]} {np.nanmean(dd[bi == k]):+5.2f}(n={(bi == k).sum()})"
                for k in range(5)) + f"  all {np.nanmean(dd):+.2f}  "
                f"rho {spearmanr(sz, dd, nan_policy='omit')[0]:+.3f}")


if __name__ == "__main__":
    main()
