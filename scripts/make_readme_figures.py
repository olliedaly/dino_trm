"""Figures for the README, generated from the stored per-image evaluation vectors.

    uv run python scripts/make_readme_figures.py size       # needs results/*/per_image only
    uv run python scripts/make_readme_figures.py examples   # also needs checkpoints + DINOv3

``size`` -> results/figures/size_tradeoff.png
    Per-image metric deltas, averaged within object-size quintiles (COCO val, 2000
    images). Every contrast is shown as the mean over all 3x3 training-seed pairings
    with a min-max envelope; the grey band is the same statistic for pairs of runs that
    differ ONLY in training seed (the noise floor).

``examples`` -> results/figures/size_examples.png
    Two COCO val images, one from the smallest-object quintile and one from the
    largest, segmented by the T=1 and T=8 models (training seed 0, eval seed 0 — the
    exact predictions stored in results/coco/per_image/*.npz, fg_ari_s0).

    Selection rule (fixed before looking at the images, so the examples are
    representative rather than cherry-picked): within the band, keep images whose
    robust per-image effect (averaged over 9 training-seed pairings x 3 eval seeds) is
    on the typical side of the band mean, AND whose displayed single prediction is
    within 3 FG-ARI points of that robust effect (single draws are noisy, so a picture
    of an unlucky draw would misrepresent the model). Then take the image whose robust
    effect is closest to the band's 75th percentile (Q1) / 25th percentile (Q5): that
    is "clearly in the direction of the average effect, but not an extreme".
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

P = "results/coco/per_image/"
OUT = "results/figures"

# reference palette (light surface) — see README "figures" note
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
               "#4a3aa7", "#e34948"]

T1 = ["trm_t1", "trm_t1_seed1", "trm_t1_seed2"]
T8 = ["trm", "trm_seed1", "trm_seed2"]
BASE = ["baseline_pm", "baseline_pm_seed1", "baseline_pm_seed2"]
MLP = ["mlp_block", "mlp_block_seed1", "mlp_block_seed2"]
METRICS = [("fg_ari", "FG-ARI  (grouping)"), ("mbo_i", "mBO$_i$  (per-object mask IoU)"),
           ("mbo_c", "mBO$_c$  (per-class mask IoU)")]


def per_image(run: str, metric: str) -> np.ndarray:
    d = np.load(P + run + ".npz")
    return np.nanmean(np.stack([d[f"{metric}_s{s}"] for s in range(3)]), axis=0)


def size_quintiles() -> tuple[np.ndarray, np.ndarray]:
    cov = np.load(P + "covariates.npz")
    size = cov["mean_obj_size"] / 256.0            # 16x16-pixel patches per object
    edges = np.quantile(size, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(edges, size), np.quantile(size, [0, .2, .4, .6, .8, 1])


def band_profile(a: str, b: str, metric: str, qi: np.ndarray) -> np.ndarray:
    d = 100 * (per_image(b, metric) - per_image(a, metric))
    return np.array([np.nanmean(d[qi == k]) for k in range(5)])


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_size() -> None:
    qi, q = size_quintiles()
    contrasts = [
        ("Recursion depth  T=1 → T=8  (same module, same params)", T1, T8, BLUE),
        ("Adding a non-recursive 3-block stack  (param-matched baseline → mlp_block)", BASE, MLP,
         ORANGE),
    ]
    nulls = list(itertools.combinations(T1, 2)) + list(itertools.combinations(T8, 2))
    xt = [f"{q[k]:.0f}–{q[k+1]:.0f}" for k in range(5)]
    xt[0], xt[-1] = f"<{q[1]:.0f}", f">{q[4]:.0f}"
    x = np.arange(5)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True, facecolor=SURFACE)
    for ax, (metric, title) in zip(axes, METRICS):
        style_axes(ax)
        nl = np.stack([band_profile(a, b, metric, qi) for a, b in nulls])
        ax.fill_between(x, nl.min(0), nl.max(0), color=MUTED, alpha=0.18, linewidth=0,
                        label="Same config, different training seed (noise floor)")
        ax.axhline(0, color=AXIS, linewidth=1.2, zorder=1)
        for label, A, B, col in contrasts:
            prof = np.stack([band_profile(a, b, metric, qi)
                             for a, b in itertools.product(A, B)])
            ax.fill_between(x, prof.min(0), prof.max(0), color=col, alpha=0.16,
                            linewidth=0)
            ax.plot(x, prof.mean(0), color=col, linewidth=2, marker="o", markersize=6.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
        ax.set_xticks(x, xt)
        ax.set_xlabel("mean object size in image (16×16 patches), quintiles",
                      color=INK2, fontsize=9)
    axes[0].set_ylabel("Δ score (points), mean over images in band", color=INK2,
                       fontsize=9)
    h, lab = axes[0].get_legend_handles_labels()
    order = [1, 2, 0]
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc="upper left",
               bbox_to_anchor=(0.012, 0.905), ncol=3, frameon=False, fontsize=9,
               labelcolor=INK2, handlelength=1.6, columnspacing=1.6)
    fig.suptitle("Recursion shifts FG-ARI from large-object images to small-object "
                 "images; mask IoU does not follow",
                 x=0.012, y=0.985, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.text(0.012, 0.915, "COCO multi-object subset, 2000 val images. Lines: mean of "
             "all 9 training-seed pairings (3×3); shaded: min–max over pairings.",
             ha="left", va="bottom", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.865))
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/size_tradeoff.png", dpi=160, facecolor=SURFACE)
    print(f"wrote {OUT}/size_tradeoff.png")


# --------------------------------------------------------------------------- #
# qualitative examples
# --------------------------------------------------------------------------- #
def pick_examples() -> list[tuple[int, str]]:
    qi, _ = size_quintiles()
    rob = np.mean([per_image(b, "fg_ari") for b in T8], 0) - \
        np.mean([per_image(a, "fg_ari") for a in T1], 0)
    disp = np.load(P + "trm.npz")["fg_ari_s0"] - np.load(P + "trm_t1.npz")["fg_ari_s0"]
    picks = []
    for band, pct, sign, name in [(0, 75, +1, "small"), (4, 25, -1, "large")]:
        m = qi == band
        mean = rob[m].mean()
        ok = m & (sign * (rob - mean) >= 0) & (np.abs(disp - rob) <= 0.03)
        target = np.percentile(rob[m], pct)
        cand = np.where(ok)[0]
        i = int(cand[np.argmin(np.abs(rob[cand] - target))])
        print(f"{name}: band mean {100*mean:+.1f}, p{pct} {100*target:+.1f}; picked idx {i} "
              f"robust {100*rob[i]:+.1f}, displayed {100*disp[i]:+.1f}")
        picks.append((i, name))
    return picks


def run_model(run: str, loader, keep: set[int]):
    import torch

    from dino_trm.eval_protocol import _upsample_pred

    sys.path.insert(0, "scripts")
    from extract_per_image import latest_ckpt, set_seed

    from dino_trm.models.full_model import DinoSlotModel

    c = torch.load(latest_ckpt("checkpoints/coco", run), map_location="cuda",
                   weights_only=False)
    model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).cuda()
    model.load_state_dict(c["model"])
    model.eval()
    g, size = model.backbone.grid_size, model.backbone.image_size
    set_seed(0)                          # identical RNG stream to extract_per_image s0
    preds, idx = {}, 0
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch["pixel_values"].cuda())
            for b in range(batch["pixel_values"].shape[0]):
                if idx in keep:
                    preds[idx] = _upsample_pred(out["masks_list"][-1][b].view(-1, g, g),
                                                size)
                idx += 1
    return preds


def align_colors(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Relabel pred's slots so a slot covering the same region as a ref slot gets the
    same colour (Hungarian on pixel overlap). Slot ids carry no meaning across models;
    this only makes the two panels comparable by eye."""
    from scipy.optimize import linear_sum_assignment

    k = int(max(ref.max(), pred.max())) + 1
    ov = np.zeros((k, k))
    np.add.at(ov, (pred.ravel(), ref.ravel()), 1)
    r, c = linear_sum_assignment(-ov)
    lut = np.arange(k)
    lut[r] = c
    return lut[pred]


def boundaries(inst: np.ndarray) -> np.ndarray:
    b = np.zeros_like(inst, dtype=bool)
    b[:-1] |= inst[:-1] != inst[1:]
    b[:, :-1] |= inst[:, :-1] != inst[:, 1:]
    return b


def fig_examples() -> None:
    from dino_trm.data.coco import build_coco_loader
    from dino_trm.data.pascal_voc import IMAGENET_MEAN, IMAGENET_STD
    from dino_trm.eval_protocol import _best_overlap, _fg_ari

    picks = pick_examples()
    loader = build_coco_loader(split="val", batch_size=16, num_workers=4, shuffle=False,
                               return_masks=True)
    keep = {i for i, _ in picks}
    pred = {r: run_model(r, loader, keep) for r in ("trm_t1", "trm")}
    ds = loader.dataset
    cov = np.load(P + "covariates.npz")

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(len(picks), 3, figsize=(11.5, 4.4 * len(picks)),
                             facecolor=SURFACE, gridspec_kw={"hspace": 0.32})
    slot_rgb = np.array([matplotlib.colors.to_rgb(c) for c in SLOT_COLORS])
    for row, (i, name) in enumerate(picks):
        item = ds[i]
        img = item["pixel_values"].permute(1, 2, 0).numpy()
        img = np.clip(img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1)
        inst = item["inst_full"].numpy()
        fg, edge = inst != 0, boundaries(inst)
        gray = img.mean(-1, keepdims=True).repeat(3, -1)

        p1 = pred["trm_t1"][i]
        p8 = align_colors(p1, pred["trm"][i])
        panels = [("Image  (white = object outlines)", None), ("T=1", p1), ("T=8", p8)]
        ids = [int(v) for v in np.unique(inst) if v]
        for col, (title, p) in enumerate(panels):
            ax = axes[row, col]
            if p is None:
                vis = img.copy()
                vis[edge] = 1.0
                sub = (f"{cov['n_inst'][i]} objects, "
                       f"{cov['mean_obj_size'][i] / 256:.1f} patches each on average")
            else:
                colored = slot_rgb[p % len(slot_rgb)]
                vis = np.where(fg[..., None], 0.35 * img + 0.65 * colored,
                               0.25 * gray + 0.75 * SURFACE_RGB)
                vis[edge] = 1.0
                ari = _fg_ari(p, inst)
                mbo = _best_overlap(p, ids, inst)
                sub = f"FG-ARI {100*ari:.1f}    mBO$_i$ {100*mbo:.1f}"
            ax.imshow(vis)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(title, color=INK, fontsize=11, loc="left")
            ax.set_xlabel(sub, color=INK2, fontsize=10)
        label = ("Smallest-object quintile" if name == "small"
                 else "Largest-object quintile")
        axes[row, 0].set_ylabel(label, color=INK, fontsize=11, fontweight="bold")
    fig.suptitle("Same images, T=1 vs T=8 (slot colours matched between models; "
                 "faded = background, which FG-ARI ignores)",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.05, wspace=0.06)
    fig.savefig(f"{OUT}/size_examples.png", dpi=130, facecolor=SURFACE)
    print(f"wrote {OUT}/size_examples.png")


SURFACE_RGB = np.array(matplotlib.colors.to_rgb(SURFACE))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["size", "examples", "all"])
    a = ap.parse_args()
    if a.which in ("size", "all"):
        fig_size()
    if a.which in ("examples", "all"):
        fig_examples()
