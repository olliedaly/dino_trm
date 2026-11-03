import sys
import numpy as np

C = sys.argv[1].rstrip("/") + "/"
cov = np.load("results/coco/per_image/covariates.npz")
size = cov["mean_obj_size"] / 256
qi = np.searchsorted(np.quantile(size, [.2, .4, .6, .8]), size)
PAIRS = [("trm_t1", "trm"), ("trm_t1_seed1", "trm_seed1"), ("trm_t1_seed2", "trm_seed2")]
SIGS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
AS = (256, 512, 1024, 2048, 4096, 10 ** 9)
prof = lambda v: np.array([np.nanmean(v[qi == q]) for q in range(5)] + [np.nanmean(v)]) * 100


def load(r):
    d = dict(np.load(C + f"d_{r}.npz"))
    try:
        d.update(np.load(C + f"a_{r}.npz"))
    except FileNotFoundError:
        pass
    return d


def matched(d, keys, target, m="fg_ari", x="pieces"):
    """Interpolate the FG-ARI profile along a sweep to the target value of x."""
    xs = np.array([d[f"{x}_{k}"].mean() for k in keys])
    ps = np.array([prof(d[f"{m}_{k}"]) for k in keys])
    o = np.argsort(xs)
    if not xs.min() <= target <= xs.max():
        return None
    return np.array([np.interp(target, xs[o], ps[o, j]) for j in range(6)])


rows = {}
for a, b in PAIRS:
    A, B = load(a), load(b)
    base = prof(A["fg_ari_s0.0_l0.0"])
    tgt = B["pieces_s0.0_l0.0"].mean()
    add = lambda k, v: rows.setdefault(k, []).append(v)
    add("T=8 (real)", prof(B["fg_ari_s0.0_l0.0"]) - base)
    v = matched(A, [f"s{s}_l0.0" for s in SIGS], tgt)
    add("T=1 blurred to T=8 contiguity", v - base if v is not None else np.full(6, np.nan))
    if "fg_ari_next256" in A:
        for nm, lab in (("next", "T=1 strays -> its 2nd-choice slot"),
                        ("nbr", "T=1 strays -> surrounding slot")):
            v = matched(A, [f"{nm}{x}" for x in AS], tgt)
            add(lab + ", matched", v - base if v is not None else np.full(6, np.nan))
            add(lab + ", all", prof(A[f"fg_ari_{nm}{AS[-1]}"]) - base)
    add("T=1 split into pieces", prof(A["fg_ari_split"]) - base)
    add("both absorbed: T=8 - T=1", prof(B["fg_ari_absorb"]) - prof(A["fg_ari_absorb"]))
    add("both split:    T=8 - T=1", prof(B["fg_ari_split"]) - prof(A["fg_ari_split"]))
    if "fg_ari_next256" in B:
        add("both 2nd-choice: T=8 - T=1",
            prof(B[f"fg_ari_next{AS[-1]}"]) - prof(A[f"fg_ari_next{AS[-1]}"]))
    print(f"{a}: pieces T1 {A['pieces_s0.0_l0.0'].mean():.3f} T8 {tgt:.3f}; "
          f"blur sweep pieces {[round(float(A[f'pieces_s{s}_l0.0'].mean()), 3) for s in SIGS]}")

print(f"\n{'ΔFG-ARI vs T=1, small -> large objects':42s}  Q1     Q2     Q3     Q4     Q5  |  all")
for k, v in rows.items():
    v = np.array(v)
    mn = np.nanmean(v, 0)
    print(f"{k:42s} " + " ".join("%+6.2f" % x for x in mn[:5]) + " | %+5.2f" % mn[5]
          + "   all per seed: " + " ".join("%+5.2f" % x for x in v[:, 5]))
