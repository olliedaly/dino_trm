"""Train-set x test-set table for arms A, F (and D where trained): differences per image,
averaged over the matched seeds, [95% image-bootstrap interval], then each seed pair.
Reads results/sa_loop/per_image (same dataset) and results/sa_loop/cross (other dataset,
written by cross_eval.py).

    uv run python scripts/sa_loop/cross_report.py > results/sa_loop/cross_report.txt
"""
import numpy as np
P = "results/sa_loop/per_image"; X = "results/sa_loop/cross"; S = ["", "_seed1", "_seed2"]
def col(d, base, key, suf=""): return np.stack([np.load(f"{d}/{base}{s}{suf}.npz")[key] for s in S])
def boot(d, n=5000, seed=0):
    rng = np.random.default_rng(seed); m = np.nanmean(d, 0); m = m[~np.isnan(m)]
    b = m[rng.integers(0, len(m), (n, len(m)))].mean(1)
    return 100*m.mean(), 100*np.percentile(b, 2.5), 100*np.percentile(b, 97.5)
def show(name, a, b):
    d = a - b; mean, lo, hi = boot(d)
    print(f"   {name:26s} {mean:+.2f} [{lo:+.2f}, {hi:+.2f}]  (" + " / ".join(f"{100*np.nanmean(x):+.2f}" for x in d) + ")")
cells = [("train COCO, test COCO", P, "", dict(A="baseline_pm", F="sa8_rand", D="sa_rec_1step")),
         ("train COCO, test VOC ", X, "__on_voc", dict(A="baseline_pm", F="sa8_rand", D="sa_rec_1step")),
         ("train VOC,  test VOC ", P, "", dict(A="baseline_pm_m", F="sa8_rand_voc")),
         ("train VOC,  test COCO", X, "__on_coco", dict(A="baseline_pm_m", F="sa8_rand_voc"))]
for m in ("mbo_i", "fg_ari", "mbo_c", "ari_bg"):
    print(f"\n##### {m}")
    for name, d, suf, arms in cells:
        g = lambda k, key: col(d, arms[k], key, suf)
        print(f" {name}: map A@3 {100*np.nanmean(g('A', f'map3|{m}')):.2f}  A@8 {100*np.nanmean(g('A', f'map8|{m}')):.2f}  F@3 {100*np.nanmean(g('F', f'map3|{m}')):.2f}  F@8 {100*np.nanmean(g('F', f'map8|{m}')):.2f}"
              f" | dec A@3 {100*np.nanmean(g('A', f'dec3|{m}')):.2f}  A@8 {100*np.nanmean(g('A', f'dec8|{m}')):.2f}  F@8 {100*np.nanmean(g('F', f'dec8|{m}')):.2f}")
        show("map F@8 - A@8", g("F", f"map8|{m}"), g("A", f"map8|{m}"))
        show("map F@8 - A@3", g("F", f"map8|{m}"), g("A", f"map3|{m}"))
        show("dec F@8 - A@3", g("F", f"dec8|{m}"), g("A", f"dec3|{m}"))
        if "D" in arms:
            show("map D@8 - A@8", g("D", f"map8|{m}"), g("A", f"map8|{m}"))
            show("dec D@8 - A@3", g("D", f"dec8|{m}"), g("A", f"dec3|{m}"))
print("\nerr x1e3")
for name, d, suf, arms in cells:
    print(f" {name}: " + "  ".join(f"{k}@{n} {1e3*np.mean(col(d, arms[k], f'err{n}', suf)):.2f}" for k in arms for n in (3, 8)))
