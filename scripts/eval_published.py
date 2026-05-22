"""Score the trained checkpoints with the published DINOSAUR protocol and print a
comparison table against the reference VOC numbers.

Run:  uv run python scripts/eval_published.py
"""

from __future__ import annotations

import glob
import json
import os
import re

import torch

from dino_trm.data.pascal_voc import build_loader
from dino_trm.eval_protocol import evaluate_protocol
from dino_trm.models.full_model import DinoSlotModel

MODES = ["baseline", "trm", "coupled"]
DEVICE = "cuda"

# Reference numbers (MLP decoder), from references/DINOSAUR README (percent).
REFERENCE = {
    "DINOSAUR (reported)":      {"mbo_i": 39.3, "mbo_c": 40.8, "fg_ari": 24.6},
    "DINOSAUR (reproduction)":  {"mbo_i": 39.1, "mbo_c": 42.9, "fg_ari": 26.1},
}


def latest_ckpt(mode: str) -> str | None:
    paths = glob.glob(os.path.join("checkpoints", f"{mode}_epoch*.pt"))
    return max(paths, key=lambda p: int(re.search(r"epoch(\d+)", p).group(1))) if paths else None


def main() -> None:
    loader = build_loader(split="val", batch_size=16, num_workers=4, shuffle=False, full_mask=True)
    results = {}
    for mode in MODES:
        ck = latest_ckpt(mode)
        if not ck:
            continue
        c = torch.load(ck, map_location=DEVICE, weights_only=False)
        model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).to(DEVICE)
        model.load_state_dict(c["model"])
        m = evaluate_protocol(model, loader, DEVICE)
        results[mode] = m
        print(f"{mode:9s} mBO_i={m['mbo_i']*100:5.1f}  mBO_c={m['mbo_c']*100:5.1f}  "
              f"FG-ARI={m['fg_ari']*100:5.1f}")

    print("\n--- reference (VOC, MLP decoder) ---")
    for name, r in REFERENCE.items():
        print(f"{name:28s} mBO_i={r['mbo_i']:5.1f}  mBO_c={r['mbo_c']:5.1f}  FG-ARI={r['fg_ari']:5.1f}")

    with open(os.path.join("results", "summary_protocol.json"), "w") as f:
        json.dump({"ours": results, "reference": REFERENCE}, f, indent=2)


if __name__ == "__main__":
    main()
