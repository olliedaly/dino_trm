"""Side-by-side segmentation comparison across the three trained models.

For a few fixed val images, renders a grid:
    input | ground truth | baseline | trm | coupled
where each model column is its predicted segmentation = per-patch argmax over the
decoder slot masks (final recursion step), coloured by slot id and upsampled to image
resolution. Lets you eyeball how the models partition the same scenes.

Run:  uv run python scripts/compare_models.py [n_images]
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

from dino_trm.data.pascal_voc import VOID_LABEL, build_loader
from dino_trm.models.full_model import DinoSlotModel
from dino_trm.utils.viz import unnormalize

MODES = ["baseline", "trm", "coupled"]
CKPT_DIR = "checkpoints"
DEVICE = "cuda"


def latest_ckpt(mode: str) -> str | None:
    paths = glob.glob(os.path.join(CKPT_DIR, f"{mode}_epoch*.pt"))
    return max(paths, key=lambda p: int(re.search(r"epoch(\d+)", p).group(1))) if paths else None


def load_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = DinoSlotModel(mode=ck["cfg"]["mode"], **ck["cfg"]["model"]).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def colourise(seg: np.ndarray, k: int) -> np.ndarray:
    """(h, w) int slot ids -> (h, w, 3) RGB using a categorical colormap."""
    cmap = plt.get_cmap("tab10")
    out = np.zeros((*seg.shape, 3))
    for s in range(k):
        out[seg == s] = cmap(s % 10)[:3]
    return out


def upsample_grid(seg_grid: np.ndarray, size: int) -> np.ndarray:
    """(g, g) int -> (size, size) int via nearest-neighbour."""
    img = Image.fromarray(seg_grid.astype(np.int32), mode="I").resize((size, size), Image.NEAREST)
    return np.asarray(img)


def gt_rgb(label_grid: np.ndarray, size: int) -> np.ndarray:
    """GT label grid -> RGB: background grey, void black, classes coloured."""
    lab = upsample_grid(label_grid, size)
    out = np.full((*lab.shape, 3), 0.6)  # background grey
    out[lab == VOID_LABEL] = 0.0          # void black
    cmap = plt.get_cmap("tab20")
    for c in np.unique(lab):
        if c in (0, VOID_LABEL):
            continue
        out[lab == c] = cmap(int(c) % 20)[:3]
    return out


def main() -> None:
    n_images = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    models = {m: load_model(c) for m in MODES if (c := latest_ckpt(m))}
    if not models:
        print("no checkpoints found")
        return
    any_model = next(iter(models.values()))
    size = any_model.backbone.image_size
    grid = any_model.backbone.grid_size

    loader = build_loader(split="val", batch_size=32, num_workers=2, shuffle=False)
    batch = next(iter(loader))

    # Prefer images that actually contain foreground (>=1 non-bg, non-void label).
    labels = batch["label"].view(batch["label"].shape[0], -1).numpy()
    has_fg = [i for i in range(labels.shape[0])
              if np.any((labels[i] != 0) & (labels[i] != VOID_LABEL))]
    picks = (has_fg + list(range(labels.shape[0])))[:n_images]

    cols = ["input", "ground truth"] + MODES
    fig, axes = plt.subplots(n_images, len(cols), figsize=(2.4 * len(cols), 2.4 * n_images))
    if n_images == 1:
        axes = axes[None, :]

    for r, idx in enumerate(picks):
        px = batch["pixel_values"][idx : idx + 1].to(DEVICE)
        axes[r, 0].imshow(unnormalize(px[0]))
        axes[r, 1].imshow(gt_rgb(batch["label"][idx].numpy(), size))
        for c, mode in enumerate(MODES):
            model = models[mode]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(px)
            seg = out["masks_list"][-1][0].argmax(0).cpu().numpy().reshape(grid, grid)
            axes[r, 2 + c].imshow(colourise(upsample_grid(seg, size), model.num_slots))
        for c in range(len(cols)):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=11)

    fig.tight_layout()
    out_path = os.path.join("results", "model_comparison.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
