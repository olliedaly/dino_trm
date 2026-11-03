"""Qualitative figure recursion_examples.png: image, GT instances, and the final slot maps
of the baseline, T=1 and T=8 (training seed 0, eval seed 0, decoder masks).
Images are drawn at random (rng seed 0) by rule, not picked by eye:
  3 of the 150 images where T=1 puts far-apart same-class objects in one slot (far
  same-class sharing >= 0.5), and 1 image with no two objects of the same class.
Each panel is titled with that image's FG-ARI and far same-class sharing."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402

from dino_trm.data.coco import COCOSubset  # noqa: E402
from dino_trm.utils.viz import unnormalize  # noqa: E402

C, OUT = sys.argv[1], sys.argv[2]
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
# Reference categorical order (dataviz palette), one hue per slot.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"font.size": 9, "text.color": INK, "figure.facecolor": SURF})
S = 336
M = dict(np.load(f"{C}/mech.npz"))
GT = np.load(f"{C}/gt.npz")
rng = np.random.default_rng(0)
far = np.flatnonzero(M["t1|dec|farC"] >= 0.5)
none = np.flatnonzero(M["gt|has_sameC"] == 0)
picks = [(i, "T=1 merges far same-class objects") for i in rng.choice(far, 3, replace=False)] + \
        [(rng.choice(none), "no two objects of one class")]
MODELS = [("baseline (no module)", "rx_baseline_pm.npz", "base|dec"), ("T=1", "rx_trm_t1.npz", "t1|dec"),
          ("T=8", "rx_trm.npz", "t8|step8")]
MASKS = {f: np.load(f"{C}/{f}")["masks"][:, -1] for _, f, _ in MODELS}
ds = COCOSubset(split="val", image_size=S, augment=False, return_masks=False)
hex2rgb = lambda h: np.array([int(h[k:k + 2], 16) for k in (1, 3, 5)]) / 255  # noqa: E731


def paint(img, lab, colors):
    out = img.astype(float) / 255 * 0.45 + 0.55 * np.array([hex2rgb(colors[k % len(colors)]) for k in lab.ravel()]).reshape(*lab.shape, 3)
    edge = np.zeros(lab.shape, bool)
    edge[:-1] |= lab[:-1] != lab[1:]
    edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    out[binary_dilation(edge)] = 1.0
    return out


fig, axes = plt.subplots(len(picks), 5, figsize=(11.5, 2.95 * len(picks)), gridspec_kw={"hspace": 0.42, "wspace": 0.04})
for r, (i, why) in enumerate(picks):
    img = unnormalize(ds[int(i)]["pixel_values"])
    inst = GT["inst"][i].astype(int)
    ids = np.unique(inst)
    gt_lab = np.searchsorted(ids, inst)  # 0 = background
    gt_paint = paint(img, gt_lab, ["#fcfcfb"] + CAT)
    gt_paint[inst == 0] = img[inst == 0] / 255 * 0.45 + 0.55
    axes[r, 0].imshow(img)
    axes[r, 0].set_title(f"image {i}\n{why}", fontsize=8.5, loc="left", color=INK)
    axes[r, 1].imshow(gt_paint)
    axes[r, 1].set_title("\nground-truth objects", fontsize=8.5, loc="left", color=INK)
    for c, (name, f, key) in enumerate(MODELS):
        t = torch.from_numpy(MASKS[f][i].astype(np.float32))[None]
        lab = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)[0].argmax(0).numpy()
        axes[r, 2 + c].imshow(paint(img, lab, CAT))
        fs = M[f"{key}|farC"][i]
        axes[r, 2 + c].set_title(f"{name}\nFG-ARI {100 * M[f'{key}|fg_ari'][i]:.0f}"
                                 + (f", far same-class {fs:.2f}" if np.isfinite(fs) else ""),
                                 fontsize=8.5, loc="left", color=INK)
    for ax in axes[r]:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
fig.text(0.01, 0.003, "Slot maps: argmax of the decoder masks after the final step (seed 0); colours mark slots and "
         "are not matched across models; white lines are slot boundaries. \"far same-class\": share of pixel pairs "
         "from far-apart same-class objects that share a slot.", fontsize=7.5, color=INK2, wrap=True)
fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.04)
fig.savefig(f"{OUT}/recursion_examples.png", dpi=130)
print("picked", [int(i) for i, _ in picks])
