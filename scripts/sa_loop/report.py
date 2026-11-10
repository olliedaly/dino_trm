"""Results table and decision for the slot-attention-loop arms (docs/sa_loop_plan.md).

Reads results/sa_loop/per_image/{run}.npz (written by eval_arms.py). Prints, as markdown:
  1. checks that every run is configured as its arm requires;
  2. each arm's scores (mean of its seeds, then the seeds);
  3. paired differences between arms: per image, averaged over the matched seeds, with a
     95% image-bootstrap interval, and the difference in each seed pair;
  4. the decision, by the rule fixed in the plan before any run was scored.

    uv run python scripts/sa_loop/report.py --stage 1 [--out results/sa_loop/stage1_report.md]
    uv run python scripts/sa_loop/report.py --stage 1 --check-reference   # stage 0
"""
import argparse
import json
import os

import numpy as np

DIR = "results/sa_loop/per_image"
SEEDS = ["", "_seed1", "_seed2"]
METRICS = [("fg_ari", "FG-ARI"), ("mbo_i", "mBO_i"), ("mbo_c", "mBO_c"), ("ari_bg", "ARI counting background")]

# arm -> (run base name, passes used at test time, required settings)
STAGES = {
    1: {
        "A": ("baseline_pm", 3, dict(mode="baseline", trained_passes=3, sa_detach=False, image_size=336)),
        "B": ("baseline_pm", 8, dict(mode="baseline", trained_passes=3, sa_detach=False, image_size=336)),
        "C": ("sa8_last", 8, dict(mode="baseline", trained_passes=8, sa_detach=True, image_size=336)),
        "D": ("sa_rec_1step", 8, dict(mode="sa_rec", trained_passes=8, sa_detach=True, image_size=336,
                                      intermediate_weight=0.25)),
    },
    2: {
        "A": ("sa3_r448", 3, dict(mode="baseline", trained_passes=3, sa_detach=False, image_size=448)),
        "B": ("sa3_r448", 8, dict(mode="baseline", trained_passes=3, sa_detach=False, image_size=448)),
        "C": ("sa8_last_r448", 8, dict(mode="baseline", trained_passes=8, sa_detach=True, image_size=448)),
        "D": ("sa8_each_r448", 8, dict(mode="sa_rec", trained_passes=8, sa_detach=True, image_size=448,
                                       intermediate_weight=0.25)),
    },
}
WHAT = {
    "A": "slot attention trained with 3 passes",
    "B": "A's weights, run for 8 passes at test time",
    "C": "trained with 8 passes, gradient from the last update only",
    "D": "trained with 8 passes, a loss after each pass",
}
CONTRASTS = [("D", "C"), ("D", "B"), ("C", "B"), ("B", "A")]
# Stage-0 reference: what the existing checkpoints scored on 2026-11-09 (eval seed 0).
# (arm, readout, metric) -> (value, tolerance); "A"/"B" are 3-seed means, "D0" is seed 0 of D.
REFERENCE = {
    ("A", "map", "fg_ari"): (47.99, 0.2), ("A", "map", "mbo_i"): (26.06, 0.15), ("A", "map", "mbo_c"): (32.83, 0.15),
    ("A", "map", "ari_bg"): (24.98, 0.2), ("A", "dec", "fg_ari"): (43.89, 0.2), ("A", "dec", "mbo_i"): (23.39, 0.15),
    ("A", "err", ""): (33.57, 0.1),
    ("B", "map", "fg_ari"): (49.27, 0.2), ("B", "map", "mbo_i"): (26.40, 0.15), ("B", "map", "mbo_c"): (33.13, 0.15),
    ("D0", "map", "fg_ari"): (49.18, 0.2), ("D0", "map", "mbo_i"): (27.27, 0.15), ("D0", "map", "mbo_c"): (34.20, 0.15),
    ("D0", "map", "ari_bg"): (25.85, 0.2), ("D0", "dec", "fg_ari"): (44.54, 0.2), ("D0", "err", ""): (30.97, 0.1),
}
MIN_GAIN = 0.5  # mBO_i points; fixed before the runs
PARAMS_M = {1: 6.3967, 2: 6.4845}  # trainable parameters at 336 px and at 448 px (more position codes)


def load(run, d=DIR):
    p = f"{d}/{run}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


def col(d, readout, n, metric):
    return 1e3 * d[f"err{n}"] if readout == "err" else 100 * d[f"{readout}{n}|{metric}"]


def ci(d, rng):
    d = d[~np.isnan(d)]
    bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=sorted(STAGES))
    ap.add_argument("--out")
    ap.add_argument("--dir", default=DIR, help="where the per-image files are (for testing this script)")
    ap.add_argument("--check-reference", action="store_true",
                    help="stage 0: compare the existing checkpoints' scores with the reference values")
    a = ap.parse_args()
    arms = STAGES[a.stage]
    L = []
    data, problems = {}, []
    for arm, (base, n, need) in arms.items():
        data[arm] = []
        for s in SEEDS:
            d = load(base + s, a.dir)
            if d is None:
                continue
            m = d["meta"]
            for k, v in need.items():
                if m.get(k) != v:
                    problems.append(f"{base + s}: {k} is {m.get(k)!r}, arm {arm} needs {v!r}")
            if m["seed"] != SEEDS.index(s):
                problems.append(f"{base + s}: trained with seed {m['seed']}")
            if abs(m["params_M"] - PARAMS_M[a.stage]) > 0.001:
                problems.append(f"{base + s}: {m['params_M']}M parameters, expected {PARAMS_M[a.stage]}M")
            if f"map{n}|fg_ari" not in d:
                problems.append(f"{base + s}: not scored at {n} passes; rerun eval_arms.py with --passes 3,8 --force")
                continue
            if len(d[f"err{n}"]) != 2000:
                problems.append(f"{base + s}: {len(d[f'err{n}'])} images, expected 2000")
            data[arm].append(d)

    L.append(f"# Slot-attention loop, stage {a.stage}\n")
    L.append("## Checks\n")
    L += [f"- PROBLEM: {p}" for p in problems] or [f"- Every scored run has its arm's settings, 2,000 images and {PARAMS_M[a.stage]}M parameters."]
    L.append("- Seeds scored: " + ", ".join(f"{arm} {len(v)}/3" for arm, v in data.items()) + "\n")

    if a.check_reference:
        assert a.stage == 1
        L.append("## Stage 0: existing checkpoints against the reference values\n")
        L.append("| arm | readout | score | reference | now | within tolerance |\n|---|---|---|---|---|---|")
        ok_all = True
        for (arm, readout, metric), (ref, tol) in REFERENCE.items():
            runs = data["D"][:1] if arm == "D0" else data[arm]
            n = arms["D" if arm == "D0" else arm][1]
            want = 1 if arm == "D0" else 3
            if len(runs) < want:
                L.append(f"| {arm} | {readout} | {metric} | {ref} | missing | NO |")
                ok_all = False
                continue
            now = float(np.mean([np.nanmean(col(d, readout, n, metric)) for d in runs]))
            ok = abs(now - ref) <= tol
            ok_all &= ok
            L.append(f"| {arm} | {readout} | {metric or 'error x1e3'} | {ref} | {now:.2f} | {'yes' if ok else 'NO'} |")
        L.append(f"\n**Stage 0 {'PASSED' if ok_all else 'FAILED'}.**"
                 + ("" if ok_all else " Stop and report; do not train anything.") + "\n")

    L.append("## Scores\n")
    L.append("Slot attention's own map at the stated number of passes, mean of the seeds. Seeds in brackets.\n")
    L.append("| arm | what | passes | " + " | ".join(lab for _, lab in METRICS) + " | error x1e3 |\n|---|---|---|" + "---|" * (len(METRICS) + 1))
    for arm, (base, n, _) in arms.items():
        if not data[arm]:
            L.append(f"| {arm} | {WHAT[arm]} | {n} | " + " | ".join("" for _ in METRICS) + " | |")
            continue
        cells = []
        for readout, metric in [("map", m) for m, _ in METRICS] + [("err", "")]:
            v = [float(np.nanmean(col(d, readout, n, metric))) for d in data[arm]]
            cells.append(f"{np.mean(v):.2f} ({' / '.join(f'{x:.1f}' for x in v)})")
        L.append(f"| {arm} | {WHAT[arm]} | {n} | " + " | ".join(cells) + " |")
    L.append("\nThe decoder's masks, same models and passes:\n")
    L.append("| arm | " + " | ".join(lab for _, lab in METRICS) + " |\n|---|" + "---|" * len(METRICS))
    for arm, (base, n, _) in arms.items():
        if data[arm]:
            L.append(f"| {arm} | " + " | ".join(
                f"{np.mean([np.nanmean(col(d, 'dec', n, m)) for d in data[arm]]):.2f}" for m, _ in METRICS) + " |")

    L.append("\n## Differences\n")
    L.append("Slot attention's map. Per image, averaged over the matched seeds; [95% image-bootstrap interval]; "
             "then each seed pair.\n")
    L.append("| contrast | " + " | ".join(lab for _, lab in METRICS) + " |\n|---|" + "---|" * len(METRICS))
    rng = np.random.default_rng(0)
    res = {}
    for x, y in CONTRASTS:
        k = min(len(data[x]), len(data[y]))
        if k == 0:
            continue
        cells = []
        for m, _ in METRICS:
            per = [col(data[x][i], "map", arms[x][1], m) - col(data[y][i], "map", arms[y][1], m) for i in range(k)]
            mean, lo, hi = ci(np.nanmean(np.array(per), 0), rng)
            seeds = [float(np.nanmean(p)) for p in per]
            res[(x, y, m)] = dict(mean=mean, lo=lo, hi=hi, seeds=seeds, k=k)
            cells.append(f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}] ({' / '.join(f'{s:+.2f}' for s in seeds)})")
        L.append(f"| {x} - {y} ({k} seed pair{'s' if k > 1 else ''}) | " + " | ".join(cells) + " |")

    L.append("\n## Decision\n")
    if any(len(v) < 3 for v in data.values()) or problems:
        L.append("**INCOMPLETE.** All four arms need 3 correctly configured seeds before the rule applies.")
    else:
        def gain(x, y):
            r = res[(x, y, "mbo_i")]
            return r["mean"] >= MIN_GAIN and r["lo"] > 0 and all(s > 0 for s in r["seeds"])

        def no_loss(x, y):
            return all(res[(x, y, m)]["hi"] >= 0 for m in ("fg_ari", "mbo_c", "ari_bg"))
        r1, r2 = gain("D", "C"), gain("D", "B")
        r3 = no_loss("D", "C") and no_loss("D", "B")
        L.append(f"- Rule 1, D beats C on mBO_i (mean >= +{MIN_GAIN}, interval above 0, every seed pair above 0): "
                 f"**{'yes' if r1 else 'no'}**")
        L.append(f"- Rule 2, D beats B on mBO_i (same test): **{'yes' if r2 else 'no'}**")
        L.append(f"- Rule 3, D is not below C or B on FG-ARI, mBO_c or ARI counting background "
                 f"(no interval entirely below 0): **{'yes' if r3 else 'no'}**\n")
        if r1 and r2 and r3:
            L.append("**PASS.** A loss after each pass adds mask quality beyond both the free test-time gain and "
                     "the published training method. Stop and report; stage 2 needs the user's go-ahead.")
        elif r2 and r3:
            L.append("**STOP: no gain over the published method.** D beats the free test-time gain but not C, so the "
                     "per-pass loss adds nothing beyond training through the last update only.")
        elif not r2:
            L.append("**STOP: the one-seed result did not replicate.** D does not beat the baseline run for 8 passes.")
        else:
            L.append("**STOP: the gain comes with a loss on another score.** See the differences table.")
    text = "\n".join(L) + "\n"
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
