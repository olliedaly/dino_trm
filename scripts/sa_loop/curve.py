"""Score against the number of test-time passes, for each way of training slot attention.

Reads results/sa_loop/curve/{run}.npz, written by
    eval_arms.py --passes 1,2,3,4,6,8,12,16 --readouts map --out-dir results/sa_loop/curve
Prints one markdown table per score (mean of the seeds found, slot attention's map) and
saves results/sa_loop/pass_curve.png: the seed mean as a line and the seed range as a band.
Arms with no scored run are left out.

    uv run python scripts/sa_loop/curve.py [--out results/sa_loop/pass_curve.md]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DIR = "results/sa_loop/curve"
SEEDS = ["", "_seed1", "_seed2"]
# arm -> (run base name, label, colour, passes it was trained with)
ARMS = {
    "A": ("baseline_pm", "A: trained with 3 passes", "#52514e", 3),
    "C": ("sa8_last", "C: 8 passes, last-update gradient", "#eda100", 8),
    "E": ("sa8_full", "E: 8 passes, ordinary training", "#2a78d6", 8),
    "D": ("sa_rec_1step", "D: 8 passes, a loss after each", "#eb6834", 8),
    "F": ("sa8_rand", "F: as D, one sampled pass per image", "#1baf7a", 8),
}
METRICS = [("mbo_i", "mBO_i"), ("fg_ari", "FG-ARI"), ("mbo_c", "mBO_c"), ("ari_bg", "ARI counting background")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--dir", default=DIR)
    a = ap.parse_args()
    data = {}
    for arm, (base, *_rest) in ARMS.items():
        runs = [np.load(f"{a.dir}/{base}{s}.npz") for s in SEEDS if os.path.exists(f"{a.dir}/{base}{s}.npz")]
        if runs:
            data[arm] = runs
    if not data:
        raise SystemExit(f"no files in {a.dir}; run eval_arms.py with --out-dir {a.dir} first")
    passes = sorted(set.intersection(*[set(json.loads(str(z["meta"]))["passes"]) for rs in data.values() for z in rs]))
    L = ["# Scores against the number of test-time passes\n",
         "Slot attention's map, mean of the seeds scored: " + ", ".join(f"{k} {len(v)}/3" for k, v in data.items()) + ".\n"]
    curves = {}
    for m, lab in METRICS:
        L.append(f"## {lab}\n")
        L.append("| arm | " + " | ".join(str(n) for n in passes) + " |\n|---|" + "---|" * len(passes))
        for arm, runs in data.items():
            v = np.array([[100 * np.nanmean(z[f"map{n}|{m}"]) for n in passes] for z in runs])  # (seeds, passes)
            curves[(arm, m)] = v
            L.append(f"| {ARMS[arm][1]} | " + " | ".join(f"{x:.1f}" for x in v.mean(0)) + " |")
        L.append("")
    text = "\n".join(L)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    x = np.arange(len(passes))
    for ax, (m, lab) in zip(axes, METRICS[:2]):
        for arm in data:
            _, label, colour, trained = ARMS[arm]
            v = curves[(arm, m)]
            ax.fill_between(x, v.min(0), v.max(0), color=colour, alpha=0.18, linewidth=0)
            ax.plot(x, v.mean(0), color=colour, linewidth=2, marker="o", markersize=4, label=label)
            if trained in passes:  # ring the pass count the arm was trained with
                i = passes.index(trained)
                ax.plot(x[i], v.mean(0)[i], marker="o", markersize=10, markerfacecolor="none",
                        markeredgecolor=colour, markeredgewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(passes)
        ax.set_xlabel("passes at test time")
        ax.set_title(lab, loc="left", fontsize=10.5)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        # Cut the axis 4 points below the weakest arm's best value: the first passes of
        # some arms sit far below and would flatten the differences that matter.
        ax.set_ylim(bottom=min(curves[(arm, m)].mean(0).max() for arm in data) - 4)
    # The legend sits above the panels: inside them it would cover the late-rising arms.
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5, loc="upper center", ncol=3)
    fig.text(0.01, 0.01, "Line: mean of the seeds. Band: range of the seeds. Ring: the number of passes used in training. "
             "The axis is cut, so the earliest passes of some arms lie below it.",
             fontsize=8.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    out_png = os.path.join(os.path.dirname(a.dir.rstrip("/")) or ".", "pass_curve.png")
    fig.savefig(out_png, dpi=130)
    print(f"figure: {out_png}")


if __name__ == "__main__":
    main()
