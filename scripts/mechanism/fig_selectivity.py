"""Figure recursion_selectivity.png for results/coco/diagnostics/recursion_mechanism.md:
reconstruction error against class selectivity at 3-7 tokens (left) and against long-range
same-object sharing (right), one point per model and training seed (eval seed 0, all val
images). Token profiles from bgloc_dose.npz, bgloc_loc.npz, bgloc_ctrl1s3.npz (1-step control,
3 seeds), bgloc_fin3.npz or bgloc_fin2.npz (loss on the final step only) and, when present,
bgloc_lp{10,40}.npz; error from rx_{run}.npz. The line is fitted to the 15 module models and
seeds it was recorded on (not the baseline, not the held-out points scored after it: the
trained-locality runs, the 1-step control's seeds 1-2 and the final-step-loss runs)."""
import os
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
D, L = np.load(f"{C}/bgloc_dose.npz"), np.load(f"{C}/bgloc_loc.npz")
PRIORS = [(f"prior b = {b}\n(held out)", np.load(f"{C}/bgloc_lp{b}.npz")[f"seeds|lp{b}"], f"trm_t1_lp{b}{{s}}")
          for b in (10, 40) if os.path.exists(f"{C}/bgloc_lp{b}.npz")]  # T=1 trained with a locality prior, not fitted
FIRST = {lab for lab, _, _ in PRIORS}  # held-out points far apart: label the first seed, not the mean
OS = np.load(f"{C}/bgloc_ctrl1s3.npz")["seeds|os"]  # 1-step control, 3 seeds; seed 0 is bgloc_dose's
FIN_LAB = "loss on the final step only (held out)"
FIN = next(np.load(f"{C}/bgloc_{t}.npz")["seeds|fin"] for t in ("fin3", "fin2") if os.path.exists(f"{C}/bgloc_{t}.npz"))


def seeds(prof, role, first=0):
    """(seed suffix, profile (type, bin), role) per seed; role: fit, held (out of sample) or ref."""
    return [(SEEDS[first + i], v, role) for i, v in enumerate(prof)]


# label, colour, rx run pattern, per-seed (suffix, profile, role)
MODELS = [("T=1", "#2a78d6", "trm_t1{s}", seeds(D["seeds|T=1"], "fit")),
          ("T=8", "#eb6834", "trm{s}", seeds(D["seeds|T=8"], "fit")),
          ("mlp_block", "#1baf7a", "mlp_block{s}", seeds(D["seeds|mlp_block"], "fit")),
          ("T=2", "#eda100", "trm_t2{s}", seeds(D["seeds|T2"], "fit")),
          ("T=4", "#eda100", "trm_t4{s}", seeds(D["seeds|T4"], "fit")),
          ("locality imposed", "#e87ba4", "trm_t1{s}_loc10", seeds(L["seeds|loc10"], "fit")),
          ("1-step control", "#6250d6", "trm_1step{s}", seeds(OS[:1], "fit") + seeds(OS[1:], "held", 1)),
          ("baseline (no module)", INK2, "baseline_pm{s}", seeds(D["seeds|baseline"], "ref")),
          (FIN_LAB, "#008300", "trm_final{s}", seeds(FIN, "held"))]
MODELS += [(lab, "#e34948", pat, seeds(prof, "held")) for lab, prof, pat in PRIORS]
pts = []
for lab, col, pat, per_seed in MODELS:
    for s, v, role in per_seed:
        mse = 1000 * np.load(f"{C}/rx_{pat.format(s=s)}.npz")["mse"][:, -1].mean()
        pts.append((lab, col, role, v[2, 1] / v[3, 1], v[1, 3], mse, s))
x_sel, x_far, y = (np.array([p[i] for p in pts]) for i in (3, 4, 5))
fit = np.array([p[2] == "fit" for p in pts])
# label placement (offset points, alignment[, anchor point]) where the default (7, 4, left, at the
# group's mean; first seed for the priors) would collide
OFFSET = {(1, "T=8"): (0, 8, "center", "top"), (1, "baseline (no module)"): (8, -13, "left"),
          (0, "baseline (no module)"): (-2, -17, "center"),
          (0, "prior b = 10\n(held out)"): (-10, -24, "right"), (1, "prior b = 10\n(held out)"): (-10, -24, "right"),
          (0, "prior b = 40\n(held out)"): (0, -34, "center", "mean"), (1, "prior b = 40\n(held out)"): (-10, -6, "right"),
          (0, FIN_LAB): (9, -3, "left", "right"), (1, FIN_LAB): (9, -3, "left", "right")}


def anchor(grp, how):
    g = np.array(grp)
    return {"mean": g.mean(0), "first": g[0], "top": g[g[:, 1].argmax()], "right": g[g[:, 0].argmax()]}[how]


b, a = np.polyfit(x_sel[fit], y[fit], 1)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, x, title, xl in [(axes[0], x_sel, "class selectivity, 3–7 tokens", "same-class ÷ different-class sharing"),
                         (axes[1], x_far, "long-range locality", "same-object token pairs > 14 tokens apart in one slot")]:
    r = np.corrcoef(x[fit], y[fit])[0, 1]
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    if ax is axes[0]:
        x0 = x[fit].min() - 0.03
        xs = np.linspace(x0, x[fit].max() + 0.03, 2)
        ax.plot(xs, a + b * xs, color=INK2, lw=1.2, ls="--", zorder=1)
        if x.min() < x0:  # held-out points below the fitted range: the line extended, dotted
            xs = np.linspace(x.min() - 0.03, x0, 2)
            ax.plot(xs, a + b * xs, color=INK2, lw=1.0, ls=":", zorder=1)
    seen = set()
    for lab, col, role, xs_, xf, yy, _ in pts:
        xv = xs_ if ax is axes[0] else xf
        held = role == "held"
        ax.scatter(xv, yy, s=64 if held else 46, color=SURF if held else col, edgecolor=col if held else SURF,
                   linewidth=2.0 if held else 1.2, zorder=3)
        if lab not in seen:
            seen.add(lab)
            grp = [(p[3] if ax is axes[0] else p[4], p[5]) for p in pts if p[0] == lab]
            dx, dy, ha, *how = OFFSET.get((int(ax is axes[1]), lab), (7, 4, "left"))
            mx, my = anchor(grp, how[0] if how else "first" if lab in FIRST else "mean")
            ax.annotate(lab, (mx, my), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8.5, color=INK, zorder=4)
    ax.set_title(f"{title}   r = {r:+.2f}".replace("-", "−").replace(title.replace("-", "−"), title) + " (module models)",
                 fontsize=10, loc="left", color=INK)
    ax.set_xlabel(xl, color=INK2, fontsize=9)
axes[0].set_ylabel("reconstruction error ×10³", color=INK2, fontsize=9)
fig.text(0.01, 0.008, "One point per model and training seed (eval seed 0, 2,000 COCO val images). Dashed line: least-squares fit to the module "
         f"models (not the baseline), error = {a:.1f} − {abs(b):.1f} × selectivity; dotted: the line extended.\nOpen circles: scored after "
         "the line was recorded, not in the fit: T=1 trained with a locality prior, the 1-step control's seeds 1–2 and the loss on the "
         "final step only.", fontsize=7.5, color=INK2)
fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=0.19)
fig.savefig(f"{OUT}/recursion_selectivity.png", dpi=130)
print(f"fit: error = {a:.2f} {b:+.2f} * sel3-7; fitted points {fit.sum()} of {len(pts)}; "
      f"residual sd {np.std(y[fit] - (a + b * x_sel[fit]), ddof=2):.2f}")
for lab, col, role, xs_, xf, yy, s in pts:
    if role == "held":
        print(f"{lab.splitlines()[0]} {s or '_seed0'}: sel {xs_:.3f} far {xf:.3f} error {yy:.2f}, "
              f"predicted {a + b * xs_:.2f}, miss {yy - a - b * xs_:+.2f}")
