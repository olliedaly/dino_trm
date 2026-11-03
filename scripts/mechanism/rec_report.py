"""sa_rec_1step vs baseline_pm / trm_t1 / trm on all 2000 val images (eval seed 0)."""
import sys

import numpy as np

C = sys.argv[1]
PD = np.load(f"{C}/pd_trm.npy")
FARSC = (PD[:, 1] == 1) & (PD[:, 2] > 112)


def load(run):
    for pre in ("itan", "an"):
        try:
            return dict(np.load(f"{C}/{pre}_{run}.npz"))
        except FileNotFoundError:
            pass


def row(d, part):
    assert len(d[f"{part}|pairs"]) == len(PD)
    return [100 * np.nanmean(d[f"{part}|fg_ari"]), 100 * np.nanmean(d[f"{part}|mbo_i"]),
            100 * np.nanmean(d[f"{part}|mbo_c"]), d[f"{part}|pairs"][FARSC].mean(),
            100 * np.nanmean(d[f"{part}|max_share"])]


SEEDS = {"baseline_pm": ["baseline_pm", "baseline_pm_seed1", "baseline_pm_seed2"],
         "trm_t1": ["trm_t1", "trm_t1_seed1", "trm_t1_seed2"],
         "trm": ["trm", "trm_seed1", "trm_seed2"]}
LINES = [("baseline_pm", "SA map, 3 it (trained)", "sa3"), ("baseline_pm", "decoder, 3 it", "dec3"),
         ("baseline_pm", "SA map, 16 it at test", "sa16"),
         ("trm_t1", "decoder (output)", "t1"), ("trm_t1", "SA map", "sa"),
         ("trm", "decoder (output)", "t8"), ("trm", "SA map", "sa"),
         ("sa_rec_1step", "SA map, 8 it (trained)", "sa8"), ("sa_rec_1step", "decoder, 8 it", "dec8"),
         ("sa_rec_1step", "SA map, 16 it at test", "sa16"), ("sa_rec_1step", "decoder, 16 it", "dec16")]
print(f"{'run':14s}{'readout':24s}{'FG-ARI':>8s}{'mBO_i':>7s}{'mBO_c':>7s}{'farSC':>7s}{'maxsh':>7s}"
      f"   | 3-seed mean [min-max] FG-ARI, mBO_i, mBO_c")
for run, lab, part in LINES:
    r = row(load(run), part)
    s = ""
    if run in SEEDS:
        v = np.array([row(load(x), part) for x in SEEDS[run]])
        s = "  ".join(f"{v[:, j].mean():.1f} [{v[:, j].min():.1f}-{v[:, j].max():.1f}]" for j in range(3))
    print(f"{run:14s}{lab:24s}" + "".join(f"{x:8.1f}" if j < 3 else f"{x:7.3f}" if j == 3 else f"{x:7.0f}"
                                          for j, x in enumerate(r)) + f"   | {s}")

d = load("sa_rec_1step")
print("\nsa_rec_1step per iteration:  t | SA map FG-ARI mBO_i farSC maxsh | decoder FG-ARI mBO_i | ARI(dec,SA) ARI(SA_t,SA_8)")
for t in list(range(1, 9)) + [16]:
    a, b = row(d, f"sa{t}"), row(d, f"dec{t}")
    print(f"{t:3d} | {a[0]:5.1f} {a[1]:5.1f} {a[3]:.3f} {a[4]:3.0f} | {b[0]:5.1f} {b[1]:5.1f} | "
          f"{np.nanmean(d[f'dec{t}|ari_sa{t}']):.2f} {np.nanmean(d[f'sa{t}|ari_sa8']):.2f}")
bp = load("baseline_pm")
print("baseline_pm ARI(dec3, sa3):", round(float(np.nanmean(bp["dec3|ari_sa3"])), 2))

print("\nslot-mates of an object (mean over objects): itself / same-class / other-class / background; best-slot IoU")
for name, runs in [("baseline_pm decoder", ["baseline_pm"]), ("baseline_pm SA map", ["baseline_pm_sa"]),
                   ("trm_t1 decoder", ["trm_t1"]), ("trm decoder", ["trm"]),
                   ("sa_rec_1step decoder", ["sa_rec_1step"]), ("sa_rec_1step SA map", ["sa_rec_1step_sa"]),
                   ("baseline_pm decoder 3s", SEEDS["baseline_pm"]),
                   ("baseline_pm SA map 3s", [x + "_sa" for x in SEEDS["baseline_pm"]]),
                   ("trm_t1 decoder 3s", SEEDS["trm_t1"]), ("trm decoder 3s", SEEDS["trm"])]:
    m = [np.load(f"{C}/mates_{r}.npz") for r in runs]
    mt = np.mean([x["mates"].mean(0) for x in m], 0)
    iou = np.mean([x["iou"].mean() for x in m])
    print(f"{name:24s}" + " ".join(f"{100 * v:5.1f}" for v in mt) + f"   IoU {100 * iou:5.1f}")
