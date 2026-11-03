"""Recursing with slot attention, two ways (eval seed 0).

iters    a trained model's slot attention run for n iterations at test time (trained with 3).
         Saves attn_{n}, m_{n} (N, K, g, g) f16: slot attention's map and the decoded masks.
         Every n gets the same initial slot noise.
sa_rec   the slot-attention recursion: attn, m (N, 8, K, g, g) for every trained iteration,
         attn_16, m_16 (N, K, g, g) when run to 16 iterations at test time.
coupled  the coupled model (slot attention re-run every step, conditioned on z). Saves
         attn (N, T+1, K, g, g): every slot-attention call (initial, then one per step),
         m (N, T, K, g, g): decoder masks after every step.
Writes c/it_{run}.npz.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from extract_per_image import latest_ckpt, set_seed  # noqa: E402
from dino_trm.data.coco import build_coco_loader  # noqa: E402
from dino_trm.models.full_model import DinoSlotModel  # noqa: E402

OUT, KIND, RUNS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
NS = [1, 2, 3, 5, 8, 16]
loader = build_coco_loader(split="val", batch_size=16, num_workers=4, shuffle=False,
                           return_masks=True)

for name in RUNS:
    c = torch.load(latest_ckpt("checkpoints/coco", name), map_location="cuda", weights_only=False)
    print(name, c["cfg"]["mode"], c["cfg"]["model"], c["cfg"]["loss"], flush=True)
    model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).cuda()
    model.load_state_dict(c["model"])
    model.eval()
    g = model.backbone.grid_size
    calls = []
    if KIND == "coupled":
        model.slot_attention.register_forward_hook(lambda mod, i, o: calls.append(o[1]))
    keep = {}
    set_seed(0)
    with torch.no_grad():
        for batch in loader:
            px = batch["pixel_values"].cuda()
            b = px.shape[0]
            grid = lambda t: t.float().reshape(b, -1, g, g).cpu().numpy().astype(np.float16)  # noqa: E731
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if KIND == "sa_rec":
                    # every trained iteration (8), plus the final map when run to 16
                    rng = torch.cuda.get_rng_state()
                    for n in (8, 16):
                        torch.cuda.set_rng_state(rng)
                        model.slot_attention.n_iters = n
                        out = model(px)
                        if n == 8:
                            keep.setdefault("attn", []).append(np.stack([grid(a) for a in out["attn_list"]], 1))
                            keep.setdefault("m", []).append(np.stack([grid(m) for m in out["masks_list"]], 1))
                        else:
                            keep.setdefault("attn_16", []).append(grid(out["attn"]))
                            keep.setdefault("m_16", []).append(grid(out["masks_list"][-1]))
                    continue
                if KIND == "coupled":
                    calls.clear()
                    out = model(px)
                    keep.setdefault("attn", []).append(np.stack([grid(a) for a in calls], 1))
                    keep.setdefault("m", []).append(np.stack([grid(m) for m in out["masks_list"]], 1))
                    continue
                feats = model.backbone(px).float()
                if model.target_norm:
                    feats = (feats - feats.mean(1, keepdim=True)) / (feats.std(1, keepdim=True) + 1e-6)
                proj = model.input_proj(feats)
                rng = torch.cuda.get_rng_state()
                for n in NS:
                    torch.cuda.set_rng_state(rng)
                    model.slot_attention.n_iters = n
                    slots, attn = model.slot_attention(proj)
                    _, masks = model.decoder(slots)
                    keep.setdefault(f"attn_{n}", []).append(grid(attn))
                    keep.setdefault(f"m_{n}", []).append(grid(masks))
    save = {k: np.concatenate(v) for k, v in keep.items()}
    np.savez(f"{OUT}/it_{name}.npz", **save)
    if KIND == "iters":
        ref = np.load(f"{OUT}/m_{name}.npy", mmap_mode="r")
        print("   n=3 vs cached: max diff", float(np.abs(save["m_3"].astype(np.float32) - ref).max()), flush=True)
    print(name, {k: v.shape for k, v in save.items()}, flush=True)
