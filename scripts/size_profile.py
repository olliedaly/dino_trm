"""Size-stratified FG-ARI profile for one or more model contrasts.

For any model contrast, prints how its Δ FG-ARI varies with object size: quintile
means with paired bootstrap CIs, plus the Spearman trend.

Object size is the mean visible object area inside the eval crop, reported in 16x16
patch units (a DINOv3-S/16 patch at 336px), because the mechanistic claim is about
size RELATIVE TO THE PATCH GRID, not in pixels.

Quintile edges are computed once on the reference dataset and shared across all
contrasts in a run, so columns are comparable down the page.

Examples:
    # the headline effect, averaged over training seeds ('+' averages runs)
    uv run python scripts/size_profile.py \
        --contrast trm_t1+trm_t1_seed1+trm_t1_seed2:trm+trm_seed1+trm_seed2

    # dose-response: does the slope steepen with T?
    uv run python scripts/size_profile.py \
        --contrast trm_t1:trm_t2 --contrast trm_t1:trm_t4 --contrast trm_t1:trm

    # VOC, matched-recipe runs
    uv run python scripts/size_profile.py --dataset voc --contrast trm_t1_m:trm_m
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy import stats

DIRS = {"coco": "results/coco/per_image", "voc": "results/per_image"}
RNG = np.random.default_rng(0)


def load(per_image: str, run: str, metric: str, n_seeds: int) -> np.ndarray:
    """Per-image metric averaged over eval seeds. Accepts a '+' separated list of
    runs (training seeds) and averages those too."""
    out = []
    for r in run.split("+"):
        d = np.load(os.path.join(per_image, f"{r}.npz"))
        out.append(np.stack([d[f"{metric}_s{s}"] for s in range(n_seeds)]).mean(0))
    return np.stack(out).mean(0)


def boot_ci(d: np.ndarray, n: int = 10000) -> tuple[float, float]:
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["coco", "voc"], default="coco")
    ap.add_argument("--metric", default="fg_ari", choices=["fg_ari", "mbo_i", "mbo_c"])
    ap.add_argument("--seeds", type=int, default=3, help="number of EVAL seeds")
    ap.add_argument("--contrast", action="append", required=True,
                    help="'a:b' meaning b minus a; use 'x+y' to average training seeds")
    ap.add_argument("--quintiles", type=int, default=5)
    args = ap.parse_args()

    pi = DIRS[args.dataset]
    cov = dict(np.load(os.path.join(pi, "covariates.npz")))
    size = cov["mean_obj_size"].astype(float)
    q = args.quintiles
    edges = np.quantile(size, np.linspace(0, 1, q + 1))
    sb = np.clip(np.searchsorted(edges[1:-1], size, side="right"), 0, q - 1)

    print(f"dataset={args.dataset}  metric={args.metric}  n={len(size)}  eval_seeds={args.seeds}")
    print("size bands (mean visible object area, in 16x16 patches):")
    for i in range(q):
        m = sb == i
        print(f"  Q{i+1}: {edges[i]/256:7.1f} - {edges[i+1]/256:7.1f}   n={m.sum():4d}"
              f"   mean n_inst={cov['n_inst'][m].mean():5.1f}")
    print()

    hdr = f"  {'contrast':>26s} " + "".join(f"{f'Q{i+1}':>9s}" for i in range(q))
    print(hdr + f"{'overall':>10s}{'rho':>9s}{'p':>10s}")
    print("  " + "-" * (len(hdr) + 27))
    for c in args.contrast:
        a, b = c.split(":")
        d = load(pi, b, args.metric, args.seeds) - load(pi, a, args.metric, args.seeds)
        keep = ~np.isnan(d)
        r = stats.spearmanr(d[keep], size[keep])
        cells = ""
        for i in range(q):
            m = (sb == i) & keep
            lo, hi = boot_ci(d[m])
            cells += f"{100*d[m].mean():+8.2f}" + ("*" if lo > 0 or hi < 0 else " ")
        lo, hi = boot_ci(d[keep])
        star = "*" if lo > 0 or hi < 0 else " "
        label = f"{a.split('+')[0]}->{b.split('+')[0]}"
        print(f"  {label:>26s} {cells}{100*d[keep].mean():+9.2f}{star}"
              f"{r.statistic:+9.4f}{r.pvalue:10.1e}")
    print("\n  * = 95% paired-bootstrap CI excludes zero")


if __name__ == "__main__":
    main()
