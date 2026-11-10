"""The main README's examples figure (results/figures/readme_examples.png): two COCO val
images, each as the image and the slots of T=1 and T=8 (training seed 0, eval seed 0,
decoder masks). Only the annotated objects are coloured, by the slot each pixel falls in,
so one colour on two objects means they share a slot and two colours on one object mean it
is cut. The background is greyed and the white lines are slot boundaries.

The two images are chosen for clarity, not at random (recursion_examples.png is the random
draw): 1086 has two large objects of one class that T=1 puts in one slot and T=8 in two;
612 has one large object that T=1 keeps in one slot and T=8 cuts across several.

    uv run python scripts/mechanism/fig_readme.py C results/figures
"""
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
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
RGB = np.array([[int(h[k:k + 2], 16) for k in (1, 3, 5)] for h in CAT]) / 255
plt.rcParams.update({"font.size": 10, "text.color": INK, "figure.facecolor": SURF})
S = 336
ROWS = [
    (1086, "two objects of one class", "both sandwiches share one slot", "one slot each"),
    (612, "one large object", "the man is one slot", "the man is cut across slots"),
]
M = np.load(f"{C}/mech.npz")
INST = np.load(f"{C}/gt.npz")["inst"]
MODELS = [("1 step (T=1)", np.load(f"{C}/rx_trm_t1.npz")["masks"][:, -1], "t1|dec"),
          ("8 steps (T=8)", np.load(f"{C}/rx_trm.npz")["masks"][:, -1], "t8|step8")]
ds = COCOSubset(split="val", image_size=S, augment=False, return_masks=False)


def paint(img, lab, fg):
    out = np.repeat(img.astype(float).mean(-1, keepdims=True) / 255 * 0.5 + 0.45, 3, -1)
    out[fg] = 0.3 * img[fg] / 255 + 0.7 * RGB[lab[fg]]
    edge = np.zeros(lab.shape, bool)
    edge[:-1] |= lab[:-1] != lab[1:]
    edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    out[binary_dilation(edge)] = 1.0
    return out


fig, axes = plt.subplots(len(ROWS), 3, figsize=(10.2, 3.75 * len(ROWS)), gridspec_kw={"hspace": 0.2, "wspace": 0.04})
for r, (i, what, *says) in enumerate(ROWS):
    img = unnormalize(ds[i]["pixel_values"])
    axes[r, 0].imshow(img)
    axes[r, 0].set_title(f"image: {what}", fontsize=10, loc="left", color=INK)
    for c, ((name, masks, key), say) in enumerate(zip(MODELS, says)):
        t = torch.from_numpy(masks[i].astype(np.float32))[None]
        lab = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)[0].argmax(0).numpy()
        axes[r, 1 + c].imshow(paint(img, lab, INST[i] > 0))
        axes[r, 1 + c].set_title(f"{name}: {say}\nFG-ARI {100 * M[f'{key}|fg_ari'][i]:.0f}",
                                 fontsize=10, loc="left", color=INK)
    for ax in axes[r]:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
fig.text(0.01, 0.012, "Annotated objects are coloured by the slot each pixel falls in; the rest is greyed. "
         "White lines are slot boundaries. Seed-0 models, decoder masks.", fontsize=8.5, color=INK2)
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.05)
fig.savefig(f"{OUT}/readme_examples.png", dpi=120)
print("FG-ARI", {i: [round(100 * float(M[f"{k}|fg_ari"][i])) for _, _, k in MODELS] for i, *_ in ROWS})
