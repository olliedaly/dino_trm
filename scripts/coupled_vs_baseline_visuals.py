"""Find COCO val images where coupled+xattn most beats / most loses to baseline on the
foreground pair-agreement score (FG-ARI — the measurement where coupled's specific
advantage shows up in the table), and render side-by-side panels:

    image | ground-truth instances | baseline prediction | coupled+xattn prediction

Single-seed forwards (eval is stochastic; same seed for both models so the
comparison is apples-to-apples on the SAME random slot init).

Run:
    uv run python scripts/coupled_vs_baseline_visuals.py            # top 5 each
    uv run python scripts/coupled_vs_baseline_visuals.py --top-k 5 --seed 0
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dino_trm.data.coco import build_coco_loader
from dino_trm.eval_protocol import _best_overlap, _fg_ari, _upsample_pred
from dino_trm.models.full_model import DinoSlotModel
from dino_trm.utils.viz import unnormalize

DEVICE = "cuda"
CKPT_DIR = "checkpoints/coco"
OUT_DIR = "results/coco/diagnostics"
MIN_INSTANCES = 3   # need ≥3 GT instances for the grouping score to be informative


def latest_ckpt(mode: str) -> str:
    paths = glob.glob(os.path.join(CKPT_DIR, f"{mode}_epoch*.pt"))
    return max(paths, key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))


def load_model(mode: str):
    ck = torch.load(latest_ckpt(mode), map_location=DEVICE, weights_only=False)
    m = DinoSlotModel(mode=ck["cfg"]["mode"], **ck["cfg"]["model"]).to(DEVICE)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def per_image(pred: np.ndarray, sem: np.ndarray, inst: np.ndarray) -> dict:
    cls_ids = [c for c in np.unique(sem) if c != 0]
    inst_ids = [i for i in np.unique(inst) if i != 0]
    return {
        "mbo_i": _best_overlap(pred, inst_ids, inst),
        "mbo_c": _best_overlap(pred, cls_ids, sem),
        "fg_ari": _fg_ari(pred, inst),
        "n_inst": len(inst_ids),
    }


def colorize(labels: np.ndarray, cmap, bg_id: int | None = 0) -> np.ndarray:
    """Map integer label map to an RGB image; ``bg_id`` (if not None) renders as white."""
    out = np.ones((*labels.shape, 3))
    ids = [i for i in np.unique(labels) if i != bg_id]
    for k, lid in enumerate(ids):
        out[labels == lid] = cmap(k % cmap.N)[:3]
    return out


def colorize_pred(pred: np.ndarray, gt_bg: np.ndarray, cmap) -> np.ndarray:
    """Whiten the slot that most covers GT background; colour the remaining slots.

    Predictions are a per-pixel argmax over 7 slots so every pixel is "in" some slot —
    without this aid the figure would be fully coloured and hard to compare to the GT
    panel. Cosmetic only; the metrics are unaffected.
    """
    bg_slot = int(np.bincount(pred[gt_bg]).argmax()) if gt_bg.any() else -1
    return colorize(pred, cmap, bg_id=bg_slot)


@torch.no_grad()
def forward_both(baseline, coupled, px: torch.Tensor, seed: int):
    torch.manual_seed(seed); np.random.seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_b = baseline(px)
    torch.manual_seed(seed); np.random.seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_c = coupled(px)
    return out_b["masks_list"][-1], out_c["masks_list"][-1]


def render(records: list, kind: str, chosen: dict, out_path: str) -> None:
    cmap = plt.get_cmap("tab20")
    n = len(records)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.4 * n))
    if n == 1:
        axes = axes[None]
    for r, rec in enumerate(records):
        d = chosen[rec["idx"]]
        gt_bg = d["inst"] == 0
        axes[r, 0].imshow(d["img"]); axes[r, 0].set_title("image")
        axes[r, 1].imshow(colorize(d["inst"], cmap, bg_id=0))
        axes[r, 1].set_title(f"ground-truth instances (n={rec['mb']['n_inst']})")
        axes[r, 2].imshow(colorize_pred(d["pred_b"], gt_bg, cmap))
        axes[r, 2].set_title(
            f"baseline\nFG-ARI={rec['mb']['fg_ari']:.2f}  per-obj IoU={rec['mb']['mbo_i']:.2f}")
        axes[r, 3].imshow(colorize_pred(d["pred_c"], gt_bg, cmap))
        axes[r, 3].set_title(
            f"coupled+xattn\nFG-ARI={rec['mc']['fg_ari']:.2f}  per-obj IoU={rec['mc']['mbo_i']:.2f}")
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(f"Δ FG-ARI = {rec['delta']:+.2f}", fontsize=12)
    fig.suptitle(
        f"COCO val: top {n} {kind} for coupled+xattn vs baseline (ranked by Δ FG-ARI)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    baseline = load_model("baseline")
    coupled = load_model("coupled")
    loader = build_coco_loader(split="val", batch_size=args.batch_size, num_workers=4,
                               shuffle=False, return_masks=True)
    grid = baseline.backbone.grid_size
    size = baseline.backbone.image_size

    # Pass 1: per-image FG-ARI delta over the whole val.
    print(f"scoring {len(loader.dataset)} val images for both models ...")
    scored = []
    for bi, batch in enumerate(loader):
        masks_b, masks_c = forward_both(baseline, coupled,
                                        batch["pixel_values"].to(DEVICE), args.seed)
        for b in range(batch["pixel_values"].shape[0]):
            sem = batch["label_full"][b].numpy().copy(); sem[sem == 255] = 0
            inst = batch["inst_full"][b].numpy()
            pred_b = _upsample_pred(masks_b[b].view(-1, grid, grid), size)
            pred_c = _upsample_pred(masks_c[b].view(-1, grid, grid), size)
            mb = per_image(pred_b, sem, inst)
            mc = per_image(pred_c, sem, inst)
            if mb["fg_ari"] is None or mc["fg_ari"] is None or mb["n_inst"] < MIN_INSTANCES:
                continue
            scored.append({
                "idx": bi * args.batch_size + b,
                "delta": mc["fg_ari"] - mb["fg_ari"],
                "mb": mb, "mc": mc,
            })
    print(f"  scored {len(scored)} images with >= {MIN_INSTANCES} GT instances")

    scored.sort(key=lambda r: r["delta"], reverse=True)
    winners = scored[: args.top_k]
    losers = scored[-args.top_k:][::-1]   # most-negative first
    keep = {r["idx"] for r in winners} | {r["idx"] for r in losers}

    # Pass 2: re-render only the chosen images (skip batches with no chosen indices).
    print(f"re-rendering {len(keep)} chosen images ...")
    chosen: dict = {}
    for bi, batch in enumerate(loader):
        ids_in_batch = [bi * args.batch_size + b for b in range(batch["pixel_values"].shape[0])]
        if not any(i in keep for i in ids_in_batch):
            continue
        masks_b, masks_c = forward_both(baseline, coupled,
                                        batch["pixel_values"].to(DEVICE), args.seed)
        for b in range(batch["pixel_values"].shape[0]):
            gi = bi * args.batch_size + b
            if gi not in keep:
                continue
            sem = batch["label_full"][b].numpy().copy(); sem[sem == 255] = 0
            inst = batch["inst_full"][b].numpy()
            chosen[gi] = {
                "img": unnormalize(batch["pixel_values"][b]),
                "inst": inst,
                "pred_b": _upsample_pred(masks_b[b].view(-1, grid, grid), size),
                "pred_c": _upsample_pred(masks_c[b].view(-1, grid, grid), size),
            }

    render(winners, "winners", chosen, os.path.join(OUT_DIR, "coupled_vs_baseline_winners.png"))
    render(losers, "losers", chosen, os.path.join(OUT_DIR, "coupled_vs_baseline_losers.png"))


if __name__ == "__main__":
    main()
