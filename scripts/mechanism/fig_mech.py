"""Figures for results/coco/diagnostics/recursion_mechanism.md."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C, OUT = sys.argv[1], sys.argv[2]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})
SEEDS = ["", "_seed1", "_seed2"]
D = {**dict(np.load(f"{C}/mech.npz")), **dict(np.load(f"{C}/mech2.npz"))}
V = dict(np.load(f"{C}/voc_mech.npz"))
rng = np.random.default_rng(0)


def groups(X):
    return [("no same-class\npair", X["gt|has_sameC"] == 0),
            ("same-class pairs,\nnone far apart", (X["gt|has_sameC"] == 1) & (X["gt|has_farC"] == 0)),
            ("far-apart\nsame-class pair", X["gt|has_farC"] == 1),
            ("all images", np.isfinite(X["gt|n_obj"]))]


def eff(X, a, b, st, m):
    d = 100 * np.mean([X[f"{b.format(s=s)}|{st}"] - X[f"{a.format(s=s)}|{st}"] for s in SEEDS], 0)[m]
    d = d[~np.isnan(d)]
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


# Figure 1: per-image group effects, COCO vs VOC (COCO-trained models)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
for ax, st, title in [(axes[0], "fg_ari", "FG-ARI change, T=1 → T=8"),
                      (axes[1], "ari_all", "ARI counting background, T=1 → T=8")]:
    for k, (lab, X, a, b, col) in enumerate([("COCO images", D, "t1{s}|dec", "t8{s}|step8", BLUE),
                                             ("VOC images", V, "coco_trm_t1{s}", "coco_trm{s}", ORANGE)]):
        G = groups(X)
        for j, (g, m) in enumerate(G):
            mu, lo, hi = eff(X, a, b, st, m)
            x = j + (k - 0.5) * 0.36
            ax.bar(x, mu, width=0.34, color=col, label=lab if j == 0 else None, zorder=2)
            ax.plot([x, x], [lo, hi], color=INK, lw=1.2, zorder=3)
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_xticks(range(4), [g for g, _ in groups(D)], fontsize=8.5)
    ax.set_title(title, fontsize=10.5, loc="left", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_ylabel("points (mean over 3 seed pairs)", color=INK2, fontsize=9)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.text(0.01, 0.01, "Bars: mean per-image change; whiskers: 95% image-bootstrap CI. COCO-trained models; eval seed 0.",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(f"{OUT}/recursion_groups.png", dpi=160)

# Figure 2: dose-response in trained depth (seed 0)
T = [1, 2, 4, 8]
parts = ["t1|dec", "t2|step2", "t4|step4", "t8|step8"]
mse = [np.load(f"{C}/rx_{r}.npz")["mse"][:, -1].mean() for r in ["trm_t1", "trm_t2", "trm_t4", "trm"]]
panels = [("far-apart same-class objects\nsharing a slot", [np.nanmean(D[f"{p}|farC"]) for p in parts], "{:.3f}"),
          ("reconstruction error\n(feature MSE)", mse, "{:.4f}"),
          ("ARI counting background", [100 * np.nanmean(D[f"{p}|ari_all"]) for p in parts], "{:.1f}"),
          ("FG-ARI", [100 * np.nanmean(D[f"{p}|fg_ari"]) for p in parts], "{:.1f}")]
fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
for ax, (title, y, fmt) in zip(axes, panels):
    ax.plot(T, y, color=BLUE, lw=2, marker="o", ms=7, mec=SURF, mew=1.5, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(T, [str(t) for t in T])
    ax.minorticks_off()
    ax.set_xlabel("recursion steps T", color=INK2, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    for t, v in zip([T[0], T[-1]], [y[0], y[-1]]):
        ax.annotate(fmt.format(v), (t, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color=INK)
    lo, hi = min(y), max(y)
    ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.3 * (hi - lo))
fig.text(0.01, 0.01, "One training seed per depth (seed 0); all 2,000 val images; decoder masks after the final step.",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(f"{OUT}/recursion_dose.png", dpi=160)
print("figures written")
