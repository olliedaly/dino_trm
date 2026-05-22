"""Model comparison with GT-matched slot colours + foreground overlay.

Produces two figures over a few fixed val images:

  results/model_comparison_matched.png
    input | ground truth | baseline | trm | coupled
    Each model's slots are Hungarian-matched to the GT objects by IoU, then coloured
    with that GT class's colour (VOC palette). So the slot covering, say, the
    aeroplane is the SAME colour in GT and in all three models; slots that match no GT
    object are greyed. This makes per-object differences directly comparable.

  results/model_comparison_overlay.png
    Object-vs-background view: the union of each model's GT-matched foreground slots,
    overlaid on the (dimmed) input image.

Run:  uv run python scripts/compare_matched.py [n_images]
"""

from __future__ import annotations

import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

from dino_trm.data.pascal_voc import VOID_LABEL, PascalVOC, voc_colormap
from dino_trm.models.full_model import DinoSlotModel
from dino_trm.utils.viz import unnormalize

MODES = ["baseline", "trm", "coupled"]
CKPT_DIR = "checkpoints"
DEVICE = "cuda"
CMAP = voc_colormap() / 255.0  # (256, 3) float, GT class -> colour


def latest_ckpt(mode: str) -> str | None:
    paths = glob.glob(os.path.join(CKPT_DIR, f"{mode}_epoch*.pt"))
    return max(paths, key=lambda p: int(re.search(r"epoch(\d+)", p).group(1))) if paths else None


def load_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = DinoSlotModel(mode=ck["cfg"]["mode"], **ck["cfg"]["model"]).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def up(arr: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(arr.astype(np.int32), "I").resize((size, size), Image.NEAREST))


def match_slots_to_gt(seg: np.ndarray, gt: np.ndarray):
    """Hungarian-match predicted slot ids to GT foreground classes by IoU.

    Returns {slot_id: gt_class} for matched pairs with positive overlap.
    """
    gt_classes = [c for c in np.unique(gt) if c not in (0, VOID_LABEL)]
    slots = list(np.unique(seg))
    if not gt_classes:
        return {}
    iou = np.zeros((len(slots), len(gt_classes)))
    for i, s in enumerate(slots):
        sm = seg == s
        for j, c in enumerate(gt_classes):
            cm = gt == c
            inter = np.logical_and(sm, cm).sum()
            union = np.logical_or(sm, cm).sum()
            iou[i, j] = inter / union if union else 0.0
    rows, cols = linear_sum_assignment(-iou)
    return {slots[r]: gt_classes[c] for r, c in zip(rows, cols) if iou[r, c] > 0.05}


def matched_rgb(seg: np.ndarray, mapping: dict, size: int) -> np.ndarray:
    """Colour matched slots with their GT class colour; unmatched slots grey."""
    seg_u = up(seg, size)
    out = np.full((*seg_u.shape, 3), 0.82)  # light grey for unmatched/background slots
    for slot, cls in mapping.items():
        out[seg_u == slot] = CMAP[int(cls)]
    return out


def gt_rgb(gt_grid: np.ndarray, size: int) -> np.ndarray:
    lab = up(gt_grid, size)
    out = np.full((*lab.shape, 3), 0.82)
    out[lab == VOID_LABEL] = 0.0
    for c in np.unique(lab):
        if c not in (0, VOID_LABEL):
            out[lab == c] = CMAP[int(c)]
    return out


def overlay(img: np.ndarray, seg: np.ndarray, fg_slots, size: int) -> np.ndarray:
    """Dim the image and paint matched foreground slots in their object colour."""
    seg_u = up(seg, size)
    base = img.astype(np.float32) / 255.0 * 0.45  # dim background
    for slot, cls in fg_slots.items():
        m = seg_u == slot
        base[m] = 0.35 * base[m] + 0.65 * CMAP[int(cls)]
    return np.clip(base, 0, 1)


def main() -> None:
    # args: [n_images] [seed]   (seed picks a different random set of fg images)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    models = {m: load_model(c) for m in MODES if (c := latest_ckpt(m))}
    any_m = next(iter(models.values()))
    size, grid = any_m.backbone.image_size, any_m.backbone.grid_size

    # Sample n foreground images from across the whole val set (not just the first
    # batch), so different seeds surface different scenes.
    ds = PascalVOC(split="val", image_size=size, grid_size=grid)
    order = np.random.default_rng(seed).permutation(len(ds))
    samples = []
    for idx in order:
        item = ds[int(idx)]
        lab = item["label"].numpy()
        if np.any((lab != 0) & (lab != VOID_LABEL)):
            samples.append(item)
        if len(samples) == n:
            break
    batch = {
        "pixel_values": torch.stack([s["pixel_values"] for s in samples]),
        "label": torch.stack([s["label"] for s in samples]),
    }
    picks = list(range(len(samples)))

    cols = ["input", "ground truth"] + MODES
    fig_m, ax_m = plt.subplots(n, len(cols), figsize=(2.4 * len(cols), 2.4 * n))
    fig_o, ax_o = plt.subplots(n, len(cols), figsize=(2.4 * len(cols), 2.4 * n))
    if n == 1:
        ax_m, ax_o = ax_m[None, :], ax_o[None, :]

    for r, idx in enumerate(picks):
        px = batch["pixel_values"][idx : idx + 1].to(DEVICE)
        img = unnormalize(px[0])
        gt_grid = batch["label"][idx].numpy()

        for ax in (ax_m, ax_o):
            ax[r, 0].imshow(img)
        ax_m[r, 1].imshow(gt_rgb(gt_grid, size))
        ax_o[r, 1].imshow(gt_rgb(gt_grid, size))

        for c, mode in enumerate(MODES):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = models[mode](px)
            seg = out["masks_list"][-1][0].argmax(0).cpu().numpy().reshape(grid, grid)
            mapping = match_slots_to_gt(seg, gt_grid)
            ax_m[r, 2 + c].imshow(matched_rgb(seg, mapping, size))
            ax_o[r, 2 + c].imshow(overlay(img, seg, mapping, size))

        for ax in (ax_m, ax_o):
            for c in range(len(cols)):
                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])
                if r == 0:
                    ax[r, c].set_title(cols[c], fontsize=11)

    for fig, name in ((fig_m, "model_comparison_matched"), (fig_o, "model_comparison_overlay")):
        fig.tight_layout()
        fig.savefig(os.path.join("results", f"{name}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved results/{name}.png")


if __name__ == "__main__":
    main()
