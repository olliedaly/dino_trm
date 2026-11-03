"""Figure recursion_profile.png for results/coco/diagnostics/recursion_mechanism.md:
top, token-pair slot sharing by distance (bgloc.npz); bottom, learning curves
(curve_sum.npz, curve_sa_sum.npz). 3-seed means throughout."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C, OUT = sys.argv[1], sys.argv[2]
SURF, INK, INK2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#efeeea"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})
SEEDS = ["", "_seed1", "_seed2"]
B = np.load(f"{C}/bgloc.npz")
names = list(B["names"])
prof = {n: B["vals"][names.index(n)] for n in ("baseline", "T=1", "T=8")}
SERIES = [("baseline (no module)", "baseline", INK2, "--"), ("T=1", "T=1", BLUE, "-"), ("T=8", "T=8", ORANGE, "-")]
TYPES = ["background–background", "same object", "two same-class objects", "two different-class objects"]
BINS = ["≤3", "3–7", "7–14", ">14"]

fig = plt.figure(figsize=(11.5, 6.4))
gs = fig.add_gridspec(2, 12, hspace=0.62, wspace=1.6)
for k, title in enumerate(TYPES):
    ax = fig.add_subplot(gs[0, 3 * k:3 * k + 3])
    for lab, key, col, ls in SERIES:
        ax.plot(range(4), prof[key][k], color=col, ls=ls, lw=2, marker="o", ms=6, mec=SURF, mew=1.2,
                label=lab, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.004, 1.0)
    ax.set_xticks(range(4), BINS)
    ax.set_xlabel("distance (tokens)", color=INK2, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0, which="major")
    if k == 0:
        ax.set_ylabel("share of token pairs\nin one slot (log)", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    else:
        ax.set_yticklabels([])
    ratio = prof["T=8"][k, 3] / prof["T=1"][k, 3] - 1
    ax.annotate(f"T=8 vs T=1: {100 * ratio:+.0f}%".replace("-", "−"), (3, prof["T=8"][k, 3]), textcoords="offset points",
                xytext=(-4, -16), ha="right", fontsize=8, color=INK)

S1, S2 = np.load(f"{C}/curve_sum.npz"), np.load(f"{C}/curve_sa_sum.npz")


def curve(Z, run, stat):
    eps = sorted({int(k.split("|e")[1].split("|")[0]) for k in Z.files if k.startswith(f"{run}|e")})
    eps = [e for e in eps if all(f"{run}{s}|e{e}|{stat}" in Z.files for s in SEEDS)]
    return np.array(eps), np.array([np.mean([Z[f"{run}{s}|e{e}|{stat}"] for s in SEEDS]) for e in eps])


panels = [("reconstruction error, T=8 ÷ T=1", None),
          ("far-apart same-class objects in one slot", (S1, "farC", 1)),
          ("slot attention's own map, FG-ARI", (S2, "fg_ari", 100))]
for j, (title, spec) in enumerate(panels):
    ax = fig.add_subplot(gs[1, 4 * j:4 * j + 4])
    ax.axvspan(2, 4, color=BAND, zorder=0, lw=0)
    if spec is None:
        e, t1 = curve(S1, "trm_t1", "val_mse")
        _, t8 = curve(S1, "trm", "val_mse")
        ax.plot(e, t8 / t1, color=ORANGE, lw=2, marker="o", ms=5, mec=SURF, mew=1, zorder=3)
        ax.axhline(1, color=INK2, lw=0.8, zorder=1)
        ax.annotate(f"{(t8 / t1)[-1]:.2f}", (e[-1], (t8 / t1)[-1]), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8.5, color=INK)
    else:
        Z, stat, sc = spec
        for lab, run, col in (("T=1", "trm_t1", BLUE), ("T=8", "trm", ORANGE)):
            e, y = curve(Z, run, stat)
            ax.plot(e, sc * y, color=col, lw=2, marker="o", ms=5, mec=SURF, mew=1, label=lab, zorder=3)
        if j == 1:
            ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_xlabel("training epoch", color=INK2, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    if j == 0:
        ax.text(3, ax.get_ylim()[1], "epochs 2–4", ha="center", va="top", fontsize=8, color=INK2)
fig.text(0.01, 0.005, "Top: decoder masks after the final step, all 2,000 val images. Bottom: first 512 val images "
         "(training images give the same error ratio). 3-seed means; eval seed 0.", fontsize=8, color=INK2)
fig.savefig(f"{OUT}/recursion_profile.png", dpi=160, bbox_inches="tight")
print("figure written")
