"""Figure recursion_race.png for results/coco/diagnostics/recursion_mechanism.md: the
output's dependence on slot attention's slots (left), slot attention's own map FG-ARI
(middle) and class selectivity at 3-7 tokens (right) at every saved epoch, for T=1, T=8 and
the 1-step control (3-seed means) and T=4 (seed 0). First 512 val images, eval seed 0.
Reads c/handover_*.npz (handover.py), c/sel_curve.npz, c/sel_curve_early.npz and, for the
1-step control's seeds 1-2, the other c/sel_curve_*.npz (sel_curve.py), c/curve_sa_sum.npz
(curve_sa_an.py) and c/sa_curve_any.npz (sa_curve_any.py).
Usage: fig_race.py C OUT_DIR"""
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
EP = [0, 1, 2, 3, 4, 7, 11, 15, 19, 24, 29]
SEEDS3 = ["", "_seed1", "_seed2"]
SEL = {**np.load(f"{C}/sel_curve.npz"), **np.load(f"{C}/sel_curve_early.npz")}
PER = {}  # the 1-step control's seeds 1-2, scored per seed while they trained (peek, mid-run, final)
for f in glob.glob(f"{C}/sel_curve_*.npz"):
    if not f.endswith("_sa.npz"):
        PER.update({k: v for k, v in np.load(f).items() if k.startswith("trm_1step_seed")})
for e in EP:
    SEL[f"trm_1step{{s}}|e{e}"] = np.concatenate([SEL[f"trm_1step|e{e}"]] + [PER[f"trm_1step{s}|e{e}"] for s in SEEDS3[1:]])
SA = {**np.load(f"{C}/curve_sa_sum.npz"), **np.load(f"{C}/sa_curve_any.npz")}
HO = {}
for f in glob.glob(f"{C}/handover_*.npz"):
    if not f.endswith("_test.npz"):
        HO.update(np.load(f))
# label, selectivity key pattern, slot-attention runs, colour, line style
MODELS = [("T=1", "trm_t1{s}", ["trm_t1" + s for s in SEEDS3], "#2a78d6", "-"),
          ("T=4", "trm_t4", ["trm_t4"], "#1baf7a", "-"),
          ("T=8", "trm{s}", ["trm" + s for s in SEEDS3], "#eb6834", "-"),
          ("1-step control", "trm_1step{s}", ["trm_1step" + s for s in SEEDS3], INK2, "--")]


def sel(pat, e):
    a = SEL[f"{pat}|e{e}"]
    return np.mean(a[:, 1, 1] / a[:, 2, 1])


def sa(runs, e):
    return 100 * np.mean([SA[f"{r}|e{e}|fg_ari"] for r in runs])


def dep(runs, e):
    """Swap each image's slot-attention slots for another image's: the share of the way the
    final error moves towards that image's own output (0 = ignored, 1 = followed)."""
    out = []
    for r in runs:
        m, mr, mo = (HO[f"{r}|e{e}|{s}"] for s in ("err", "err_roll", "err_other"))
        out.append((mr[:, -1].mean() - m[:, -1].mean()) / (mo.mean() - m[:, -1].mean()))
    return np.mean(out)


fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), gridspec_kw={"wspace": 0.24})
END = {(2, "T=8"): 7, (2, "1-step control"): -9, (1, "T=8"): -6, (1, "T=4"): 4,
       (0, "T=8"): -15, (0, "1-step control"): 0, (0, "T=4"): 15}  # label nudges (points)
PANELS = [(axes[0], "a. output's dependence on slot attention's slots", dep),
          (axes[1], "b. slot attention's own map: FG-ARI", lambda pat, runs, e: sa(runs, e)),
          (axes[2], "c. class selectivity, 3–7 tokens", lambda pat, runs, e: sel(pat, e))]
for k, (ax, title, fn) in enumerate(PANELS):
    ax.set_xscale("function", functions=(np.sqrt, np.square))
    ax.axvspan(2, 7, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    for lab, pat, runs, col, ls in MODELS:
        y = [fn(runs, e) if k == 0 else fn(pat, runs, e) for e in EP]
        ax.plot(EP, y, color=col, lw=2, ls=ls, zorder=2)
        ax.scatter(EP, y, s=22, color=col, edgecolor=SURF, linewidth=1.0, zorder=3)
        ax.annotate(lab, (EP[-1], y[-1]), textcoords="offset points", xytext=(8, END.get((k, lab), 0) - 3),
                    fontsize=8.5, color=INK, zorder=4)
    ax.set_xticks(EP)
    ax.set_xlim(0, 33)
    ax.set_xlabel("epoch", color=INK2, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.text(2.1, ax.get_ylim()[1], "epochs 2–7", fontsize=8, color=INK2, va="top")
fig.text(0.01, 0.01, "First 512 COCO val images, eval seed 0. T=1, T=8 and the 1-step control: mean of 3 training seeds; T=4: seed 0 "
         "(T=2, tables in section 4, behaves like T=4 but later).\n"
         "(a) Each image's slot-attention slots are swapped for another image's: 0 = the final output is unchanged, 1 = it becomes "
         "the other image's output.\n"
         "With more steps the module learns to rebuild the slots from the image, so the output stops following slot attention's slots, "
         "the more steps the sooner (a). Slot attention then stops segmenting (b).\n"
         "T=1 and T=4 take up slot attention's grouping (c) while their output still follows it; T=8 and the 1-step control "
         "stop following it first.", fontsize=7.5, color=INK2)
fig.subplots_adjust(left=0.04, right=0.955, top=0.9, bottom=0.25)
fig.savefig(f"{OUT}/recursion_race.png", dpi=130)
for lab, pat, runs, col, ls in MODELS:
    print(f"{lab:15s} sel " + " ".join(f"{sel(pat, e):.2f}" for e in EP) + " | SA " + " ".join(f"{sa(runs, e):.0f}" for e in EP)
          + " | dep " + " ".join(f"{dep(runs, e):.2f}" for e in EP))
