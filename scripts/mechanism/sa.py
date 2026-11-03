"""Slot attention's own result vs the module's output, per run (eval seed 0).

Saves c/sa_{run}.npz
  attn  (N, K, g, g) f16   slot attention's final-iteration attention (softmax over slots)
  mx    (N, K, g, g) f16   the run's decoder applied to slot attention's slots directly
  m     (N, T, K, g, g) f16  decoder masks after every module step (baseline: T=1, = mx)
  x     (N, K, D) f16      slot attention's slots (the module's input)
  y     (N, T, K, D) f16   module output after every step
  ca    (N, S, L, K, g*g) f16  head-averaged slot->patch cross-attention, first and last step
  nrm   (N, T, L, 3) f32   |self-attn|, |cross-attn|, |mlp| update / |stream|, mean over slots
  m_noca (N, K, g, g) f16  final masks with every cross-attention update zeroed
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from extract_per_image import latest_ckpt, set_seed  # noqa: E402
from dino_trm.data.coco import build_coco_loader  # noqa: E402
from dino_trm.models import tiny_gnn  # noqa: E402
from dino_trm.models.full_model import DinoSlotModel  # noqa: E402

OUT, RUNS = sys.argv[1], sys.argv[2].split(",")
REC = {"ca": [], "nrm": [], "ablate": False}


def rel(u, s):
    return u.float().norm(dim=-1).mean(1) / s.float().norm(dim=-1).mean(1)


def attn_weights(mha, q, kv):
    """Head-averaged attention weights (B, K, N), computed aside so the forward is unchanged."""
    E, H = mha.embed_dim, mha.num_heads
    W, b = mha.in_proj_weight.float(), mha.in_proj_bias.float()
    Q = (q.float() @ W[:E].T + b[:E]).unflatten(-1, (H, E // H)).transpose(1, 2)
    K = (kv.float() @ W[E:2 * E].T + b[E:2 * E]).unflatten(-1, (H, E // H)).transpose(1, 2)
    return ((Q @ K.transpose(-1, -2)) * (E // H) ** -0.5).softmax(-1).mean(1)


def block_forward(self, x, patches=None):
    h = self.norm1(x)
    a, _ = self.attn(h, h, h, need_weights=False)
    r = [rel(a, x)]
    x = x + a
    if self.cross_attn is not None:
        q = self.norm_cq(x)
        kv = self.norm_ckv(patches)
        c, _ = self.cross_attn(q, kv, kv, need_weights=False)
        REC["ca"].append(attn_weights(self.cross_attn, q, kv))
        r.append(rel(c, x))
        if REC["ablate"]:
            c = torch.zeros_like(c)
        x = x + c
    else:
        r.append(torch.zeros_like(r[0]))
    m = self.mlp(self.norm2(x))
    r.append(rel(m, x))
    REC["nrm"].append(torch.stack(r, 1))
    return x + m


tiny_gnn._Block.forward = block_forward
loader = build_coco_loader(split="val", batch_size=16, num_workers=4, shuffle=False,
                           return_masks=True)

for name in RUNS:
    c = torch.load(latest_ckpt("checkpoints/coco", name), map_location="cuda", weights_only=False)
    model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).cuda()
    model.load_state_dict(c["model"])
    model.eval()
    g = model.backbone.grid_size
    model.slot_attention.register_forward_hook(lambda mod, i, o: REC.__setitem__("x", o[0]))
    keep = {k: [] for k in ("attn", "mx", "m", "x", "y", "ca", "nrm", "m_noca")}
    for ablate in (False, True):
        if ablate and model.reasoner is None:
            break
        REC["ablate"] = ablate
        set_seed(0)
        with torch.no_grad():
            for batch in loader:
                REC["ca"].clear()
                REC["nrm"].clear()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(batch["pixel_values"].cuda())
                    b, k = out["masks_list"][0].shape[:2]
                    grid = lambda t: t.float().view(b, k, g, g).cpu().numpy().astype(np.float16)  # noqa: E731
                    if ablate:
                        keep["m_noca"].append(grid(out["masks_list"][-1]))
                        continue
                    _, mx = model.decoder(REC["x"])
                keep["attn"].append(grid(out["attn"]))
                keep["mx"].append(grid(mx))
                keep["m"].append(np.stack([grid(m) for m in out["masks_list"]], 1))
                keep["x"].append(REC["x"].float().cpu().numpy().astype(np.float16))
                keep["y"].append(torch.stack(out["slots_list"], 1).float().cpu().numpy().astype(np.float16))
                if REC["ca"]:
                    T = len(out["masks_list"])
                    L = len(REC["ca"]) // T
                    ca = torch.stack(REC["ca"], 1).view(b, T, L, k, -1)
                    keep["ca"].append(ca[:, [0, -1]].cpu().numpy().astype(np.float16))
                    keep["nrm"].append(torch.stack(REC["nrm"], 1).view(b, T, L, 3).cpu().numpy())
    save = {k: np.concatenate(v) for k, v in keep.items() if v}
    np.savez(f"{OUT}/sa_{name}.npz", **save)
    print(name, {k: v.shape for k, v in save.items()}, flush=True)
    if os.path.exists(f"{OUT}/m_{name}.npy"):
        ref = np.load(f"{OUT}/m_{name}.npy", mmap_mode="r")
        fin = save["m"][:, -1].astype(np.float32)
        print("   max|m - cached|", float(np.abs(fin - ref).max()),
              "argmax agree", float((fin.argmax(1) == ref.argmax(1)).mean()), flush=True)
