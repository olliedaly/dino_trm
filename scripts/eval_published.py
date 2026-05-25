"""Score the trained checkpoints with the published DINOSAUR protocol and print a
comparison table against the reference numbers.

Eval is stochastic (random slot init each forward), so by default this seed-averages
3 runs and reports mean +/- std — the real signal is a gap >> std.

Run:
    uv run python scripts/eval_published.py                    # VOC (checkpoints/)
    uv run python scripts/eval_published.py --dataset coco     # COCO subset (checkpoints/coco)
    uv run python scripts/eval_published.py --dataset coco --seeds 3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

import numpy as np
import torch

from dino_trm.eval_protocol import evaluate_protocol
from dino_trm.models.full_model import DinoSlotModel

# Canonical run set: baseline + recursive variants + the param/recursion controls.
# Each entry is a *checkpoint name* (used to find `{name}_epoch*.pt` and the results
# subdir); the architecture is recovered from the checkpoint's saved cfg.mode, so the
# T=1 ablations live here as their own run-names alongside the canonical T=8 runs.
MODES = ["baseline", "mlp_block", "trm", "coupled", "trm_t1", "coupled_t1"]
DEVICE = "cuda"

# Reference numbers (MLP decoder), from references/DINOSAUR README (percent).
REFERENCE = {
    "voc": {
        "DINOSAUR (reported)":     {"mbo_i": 39.3, "mbo_c": 40.8, "fg_ari": 24.6},
        "DINOSAUR (reproduction)": {"mbo_i": 39.1, "mbo_c": 42.9, "fg_ari": 26.1},
    },
    # NOTE: full-COCO numbers — our COCO run is a curated multi-object SUBSET, so these
    # are a loose reference for the regime, not a like-for-like target.
    "coco": {
        "DINOSAUR (reported, full COCO)":     {"mbo_i": 26.1, "mbo_c": 30.0, "fg_ari": 39.4},
        "DINOSAUR (reproduction, full COCO)": {"mbo_i": 28.0, "mbo_c": 31.7, "fg_ari": 40.2},
    },
}


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def latest_ckpt(ckpt_dir: str, mode: str) -> str | None:
    paths = glob.glob(os.path.join(ckpt_dir, f"{mode}_epoch*.pt"))
    return max(paths, key=lambda p: int(re.search(r"epoch(\d+)", p).group(1))) if paths else None


def build_val_loader(dataset: str, batch_size: int):
    if dataset == "coco":
        from dino_trm.data.coco import build_coco_loader
        return build_coco_loader(split="val", batch_size=batch_size, num_workers=4,
                                 shuffle=False, return_masks=True)
    from dino_trm.data.pascal_voc import build_loader
    return build_loader(split="val", batch_size=batch_size, num_workers=4,
                        shuffle=False, full_mask=True)


def eval_seed_avg(model, loader, seeds: list[int], max_batches: int | None) -> dict:
    runs = []
    for s in seeds:
        set_seed(s)
        runs.append(evaluate_protocol(model, loader, DEVICE, max_batches=max_batches))
    out = {"n_seeds": len(seeds)}
    for k in ("mbo_i", "mbo_c", "fg_ari"):
        v = np.array([r[k] for r in runs])
        out[k], out[k + "_std"] = float(v.mean()), float(v.std())
    out["fg_ari_per_step"] = runs[0]["fg_ari_per_step"]  # single-seed; shape is the signal
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["voc", "coco"], default="voc")
    ap.add_argument("--seeds", type=int, default=3, help="number of seed-averaged eval runs")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--modes", type=str, default=None,
                    help="comma-separated run-names to evaluate; default = full set")
    args = ap.parse_args()
    modes = args.modes.split(",") if args.modes else MODES

    ckpt_dir = "checkpoints" if args.dataset == "voc" else os.path.join("checkpoints", args.dataset)
    results_dir = "results" if args.dataset == "voc" else os.path.join("results", args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    seeds = list(range(args.seeds))

    loader = build_val_loader(args.dataset, args.batch_size)
    print(f"dataset={args.dataset}  ckpt_dir={ckpt_dir}  seeds={seeds}  "
          f"val_images={len(loader.dataset)}")

    results = {}
    for mode in modes:
        ck = latest_ckpt(ckpt_dir, mode)
        if not ck:
            continue
        c = torch.load(ck, map_location=DEVICE, weights_only=False)
        model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).to(DEVICE)
        model.load_state_dict(c["model"])
        m = eval_seed_avg(model, loader, seeds, args.max_batches)
        results[mode] = {"ckpt": ck, **m}
        xattn = c["cfg"]["model"].get("gnn_cross_attn", False)
        n_steps = c["cfg"]["model"].get("n_steps", None)
        suffix = "+xattn" if xattn else ""
        # Append T=N marker for ablations so the table is self-documenting.
        if c["cfg"]["mode"] in {"trm", "coupled"} and n_steps is not None and n_steps != 8:
            suffix += f" T={n_steps}"
        tag = f"{mode}{suffix}"
        print(f"{tag:22s} mBO_i={m['mbo_i']*100:5.1f}±{m['mbo_i_std']*100:.1f}  "
              f"mBO_c={m['mbo_c']*100:5.1f}±{m['mbo_c_std']*100:.1f}  "
              f"FG-ARI={m['fg_ari']*100:5.1f}±{m['fg_ari_std']*100:.1f}")

    print(f"\n--- reference ({args.dataset}, MLP decoder) ---")
    for name, r in REFERENCE[args.dataset].items():
        print(f"{name:34s} mBO_i={r['mbo_i']:5.1f}  mBO_c={r['mbo_c']:5.1f}  FG-ARI={r['fg_ari']:5.1f}")

    with open(os.path.join(results_dir, "summary_protocol.json"), "w") as f:
        json.dump({"dataset": args.dataset, "seeds": seeds,
                   "ours": results, "reference": REFERENCE[args.dataset]}, f, indent=2)
    print(f"\nwrote {results_dir}/summary_protocol.json")


if __name__ == "__main__":
    main()
