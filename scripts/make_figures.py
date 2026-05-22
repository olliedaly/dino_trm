"""Phase 4 deliverables: final full-val eval + the two headline figures.

After training, for each available mode this:
  * loads the latest checkpoint, runs a FULL (uncapped) val eval -> final FG-ARI/mBO
    and per-recursion-step FG-ARI;
  * writes results/summary.json;
  * plots FG-ARI vs recursion depth t for all modes (baseline = flat reference);
  * saves slot-mask evolution figures for 4 fixed val images (coupled if present,
    else the deepest-recursion model available).

Run:  uv run python scripts/make_figures.py
"""

from __future__ import annotations

import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dino_trm.data.pascal_voc import build_loader
from dino_trm.eval_protocol import evaluate_protocol
from dino_trm.models.full_model import DinoSlotModel
from dino_trm.utils.viz import recursion_evolution_figure

MODES = ["baseline", "trm", "coupled"]
CKPT_DIR = "checkpoints"
RESULTS = "results"
DEVICE = "cuda"


def latest_ckpt(mode: str) -> str | None:
    paths = glob.glob(os.path.join(CKPT_DIR, f"{mode}_epoch*.pt"))
    if not paths:
        return None
    return max(paths, key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))


def load_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ck["cfg"]
    model = DinoSlotModel(mode=cfg["mode"], **cfg["model"]).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    val_loader = build_loader(split="val", batch_size=16, num_workers=4, shuffle=False, full_mask=True)

    summary = {}
    models = {}
    for mode in MODES:
        ck = latest_ckpt(mode)
        if ck is None:
            print(f"[skip] no checkpoint for {mode}")
            continue
        model, cfg = load_model(ck)
        models[mode] = model
        metrics = evaluate_protocol(model, val_loader, DEVICE)  # full val, published protocol
        summary[mode] = {
            "ckpt": ck,
            "mbo_i": metrics["mbo_i"],
            "mbo_c": metrics["mbo_c"],
            "fg_ari": metrics["fg_ari"],
            "fg_ari_per_step": metrics["fg_ari_per_step"],
            "n_steps": cfg["model"]["n_steps"],
        }
        print(f"{mode:9s} mBO_i={metrics['mbo_i']:.4f} mBO_c={metrics['mbo_c']:.4f} "
              f"FG-ARI={metrics['fg_ari']:.4f}")

    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- Figure 1: FG-ARI vs recursion depth ---
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode, s in summary.items():
        ys = s["fg_ari_per_step"]
        if len(ys) == 1:
            ax.axhline(ys[0], ls="--", label=f"{mode} (no recursion)")
        else:
            ax.plot(range(1, len(ys) + 1), ys, marker="o", label=mode)
    ax.set_xlabel("recursion step t")
    ax.set_ylabel("FG-ARI (val)")
    ax.set_title("FG-ARI vs recursion depth")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fg_ari_vs_depth.png"), dpi=130)
    plt.close(fig)
    print(f"saved {RESULTS}/fg_ari_vs_depth.png")

    # --- Figure 2: slot-mask evolution for 4 fixed val images ---
    viz_mode = "coupled" if "coupled" in models else ("trm" if "trm" in models else None)
    if viz_mode is not None:
        model = models[viz_mode]
        grid = model.backbone.grid_size
        batch = next(iter(val_loader))
        for i in range(min(4, batch["pixel_values"].shape[0])):
            px = batch["pixel_values"][i : i + 1].to(DEVICE)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(px)
            masks = [m[0] for m in out["masks_list"]]
            fig = recursion_evolution_figure(px[0], masks, grid)
            fig.savefig(os.path.join(RESULTS, f"qualitative_{viz_mode}_img{i}.png"),
                        dpi=110, bbox_inches="tight")
            plt.close(fig)
        print(f"saved {RESULTS}/qualitative_{viz_mode}_img*.png ({viz_mode})")


if __name__ == "__main__":
    main()
