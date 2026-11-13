"""Figure readme_takeover.png for the main README: what training with 8 steps does to slot
attention, with the two controls that separate the number of steps from weight sharing.

Panels a to c follow training on the first 512 COCO validation images (eval seed 0). Lines
are 3-seed means and bands the range of the seeds:
  a  how far the output follows slot attention's slots (the swap test of handover.py)
  b  FG-ARI of slot attention's own attention map
  c  class selectivity of the output at 3-7 tokens
for T=1, T=8, T=8 with one wide shared block (trm_wide) and T=8 with a separate block per
step (trm_untied). Panel d: the wide shared block minus the separate blocks on the four
segmentation scores of the final output, on all 2,000 images, paired per image and averaged
over the three seed pairs, with 95% image-bootstrap intervals.
Reads c/handover_*.npz (handover.py), c/sel_curve*.npz (sel_curve.py), c/curve_sa_sum.npz
(curve_sa_an.py), c/sa_curve_any.npz (sa_curve_any.py) and c/mechc_trm_{wide,untied}*.npz
(mech_ctrl.py).
Usage: fig_takeover.py C OUT_DIR"""
import glob
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C, OUT = sys.argv[1], sys.argv[2]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})
SEEDS = ["", "_seed1", "_seed2"]
# label, run base name, colour, marker
MODELS = [("1 step (T=1)", "trm_t1", "#2a78d6", "o"),
          ("8 steps, one small shared block (T=8)", "trm", "#eb6834", "s"),
          ("8 steps, one wide shared block", "trm_wide", "#1baf7a", "^"),
          ("8 steps, a separate block per step", "trm_untied", "#eda100", "D")]

SEL = {}
for f in glob.glob(f"{C}/sel_curve*.npz"):
    if not f.endswith("_sa.npz"):
        SEL.update(np.load(f))
SA = {**np.load(f"{C}/curve_sa_sum.npz"), **np.load(f"{C}/sa_curve_any.npz")}
HO = {}
for f in glob.glob(f"{C}/handover_*.npz"):
    if not f.endswith("_test.npz"):
        HO.update(np.load(f))


def follows(base, e):
    """Swap test, per seed: 1 = the output follows slot attention's slots, 0 = it ignores them."""
    out = []
    for s in SEEDS:
        k = f"{base}{s}|e{e}"
        if f"{k}|err" not in HO:
            return None
        m, mr, mo = HO[f"{k}|err"][:, -1].mean(), HO[f"{k}|err_roll"][:, -1].mean(), HO[f"{k}|err_other"].mean()
        out.append((mr - m) / (mo - m))
    return out


def sa_map(base, e):
    """FG-ARI of slot attention's own map, per seed."""
    keys = [f"{base}{s}|e{e}|fg_ari" for s in SEEDS]
    return [100 * float(SA[k]) for k in keys] if all(k in SA for k in keys) else None


def selectivity(base, e):
    """Class selectivity of the output at 3-7 tokens, per seed."""
    pooled = f"{base}{{s}}|e{e}"  # T=1 and T=8 are stored with one row per seed
    if pooled in SEL:
        a = SEL[pooled]
    elif all(f"{base}{s}|e{e}" in SEL for s in SEEDS):
        a = np.concatenate([SEL[f"{base}{s}|e{e}"] for s in SEEDS])
    else:
        return None
    return list(a[:, 1, 1] / a[:, 2, 1])


def shared_minus_separate(metric, n_boot=5000):
    """Wide shared block minus separate blocks, final step, in points: mean and 95% interval."""
    d = np.mean([np.load(f"{C}/mechc_trm_wide{s}.npz")[f"trm_wide{s}|step8|{metric}"]
                 - np.load(f"{C}/mechc_trm_untied{s}.npz")[f"trm_untied{s}|step8|{metric}"] for s in SEEDS], 0)
    d = 100 * d[~np.isnan(d)]
    boot = d[np.random.default_rng(0).integers(0, len(d), (n_boot, len(d)))].mean(1)
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


fig, axes = plt.subplots(2, 2, figsize=(11, 7.9), gridspec_kw={"wspace": 0.2, "hspace": 0.42})
EPOCHS = range(30)
PANELS = [(axes[0, 0], "a. How far the output follows slot attention's slots", follows),
          (axes[0, 1], "b. FG-ARI of slot attention's own map", sa_map),
          (axes[1, 0], "c. How strongly the output groups by class", selectivity)]
for ax, title, fn in PANELS:
    ax.set_xscale("function", functions=(np.sqrt, np.square))
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    for label, base, colour, marker in MODELS:
        pts = [(e, fn(base, e)) for e in EPOCHS]
        x = [e for e, v in pts if v is not None]
        v = np.array([v for _, v in pts if v is not None])  # (epochs, seeds)
        ax.fill_between(x, v.min(1), v.max(1), color=colour, alpha=0.16, lw=0, zorder=1)
        ax.plot(x, v.mean(1), color=colour, lw=2, marker=marker, markersize=5, markeredgecolor=SURF,
                markeredgewidth=0.9, label=label, zorder=2)
        print(f"{title[:2]} {label:38s} " + " ".join(f"e{e}:{m:.2f}" for e, m in zip(x, v.mean(1))))
    ax.set_xticks([0, 1, 2, 4, 7, 11, 19, 29])
    ax.set_xlim(0, 30.5)
    ax.set_xlabel("training epoch", color=INK2, fontsize=9)
    ax.set_title(title, fontsize=10.5, loc="left", color=INK)
axes[0, 0].set_ylim(-0.04, 1.0)
axes[0, 0].text(30, 0.97, "1 = follows them fully", fontsize=8.5, color=INK2, ha="right", va="top")
axes[0, 0].text(30, 0.045, "0 = ignores them", fontsize=8.5, color=INK2, ha="right", va="bottom")
axes[1, 0].text(30, axes[1, 0].get_ylim()[0] + 0.02, "1 = no preference for same-class patches", fontsize=8.5,
                color=INK2, ha="right", va="bottom")

ax = axes[1, 1]
ROWS = [("FG-ARI", "fg_ari"), ("mBO_i", "mbo_i"), ("mBO_c", "mbo_c"), ("ARI counting\nbackground", "ari_all")]
ax.axvline(0, color=INK2, lw=1, zorder=1)
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
for i, (label, metric) in enumerate(ROWS):
    mean, lo, hi = shared_minus_separate(metric)
    y = len(ROWS) - 1 - i
    ax.plot([lo, hi], [y, y], color=INK, lw=1.6, zorder=2)
    ax.plot([mean], [y], marker="o", markersize=7, color=INK, markeredgecolor=SURF, markeredgewidth=1, zorder=3)
    ax.text(hi + 0.04, y, f"{mean:+.2f}".replace("-", "−"), fontsize=9, color=INK, va="center")
    print(f"d. {metric:8s} {mean:+.2f} [{lo:+.2f}, {hi:+.2f}]")
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([r[0] for r in ROWS][::-1], color=INK)
ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.set_xlim(-1.05, 0.35)
ax.set_xlabel("one shared block minus a block per step, in points", color=INK2, fontsize=9)
ax.set_title("d. Shared weights against separate blocks, at 8 steps", fontsize=10.5, loc="left", color=INK)
ax.text(-1.02, -0.5, "left of zero: the shared block scores lower", fontsize=8.5, color=INK2, va="bottom")
ax.tick_params(axis="y", length=0)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995))
fig.text(0.012, 0.012, "Panels a to c: first 512 COCO validation images; line = mean of 3 training seeds, band = their range. "
         "Panel d: all 2,000 images, both models at 23.5M parameters,\n95% intervals from resampling images. "
         "The small shared block has 6.40M parameters in total, like T=1.", fontsize=8, color=INK2)
fig.subplots_adjust(left=0.065, right=0.975, top=0.875, bottom=0.115)
fig.savefig(f"{OUT}/readme_takeover.png", dpi=130)
print(f"figure: {OUT}/readme_takeover.png")
