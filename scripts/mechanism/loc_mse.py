"""Does locality itself cost reconstruction? Impose locality on T=1's decoder after training
and measure the feature-reconstruction error.

The decoder reconstructs sum_k softmax_k(alpha_k) f_k. Locality is imposed on the alpha
logits: alpha_k - b d_k^2, with d_k the distance (image widths) from each patch to the
centroid of slot k's largest 8-connected piece in the unmodified 21x21 argmax map (the
token-level, soft form of geo2.py's loc{b}). The f_k are left as trained.
All val images, eval seed 0 (one pass per model, as gpu_batch.py, so b=0 reproduces
rx_trm_t1*.npz). Writes c/rx_{run}_loc{b}.npz (mse (N, 1), masks (N, 1, K, g, g) f16) so
mech_ctrl.py can score them.
Caps this process's GPU memory so a concurrent training run keeps its headroom."""
import sys

import numpy as np
import torch
from scipy.ndimage import label as cc_label

sys.path.insert(0, "scripts")
from extract_per_image import latest_ckpt, set_seed  # noqa: E402
from dino_trm.data.coco import build_coco_loader  # noqa: E402
from dino_trm.models.full_model import DinoSlotModel  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.2)
C, RUNS = sys.argv[1], sys.argv[2].split(",")
BS = [0, 5, 10, 20]
B = 16
EIGHT = np.ones((3, 3), int)


def d2_main_piece(alpha, g):
    """alpha (b, K, N) -> squared distance of every patch to each slot's main-piece centroid."""
    b, K, _ = alpha.shape
    yy, xx = (np.mgrid[:g, :g] + 0.5) / g
    p = alpha.argmax(1).reshape(b, g, g).cpu().numpy()
    d2 = np.zeros((b, K, g, g), np.float32)
    for i in range(b):
        for k in range(K):
            m = p[i] == k
            if not m.any():
                continue
            comp, _ = cc_label(m, structure=EIGHT)
            big = comp == np.bincount(comp.ravel())[1:].argmax() + 1
            d2[i, k] = (yy - yy[big].mean()) ** 2 + (xx - xx[big].mean()) ** 2
    return torch.from_numpy(d2.reshape(b, K, g * g)).to(alpha.device)


loader = build_coco_loader(split="val", batch_size=B, num_workers=4, shuffle=False, return_masks=False)
c = torch.load(latest_ckpt("checkpoints/coco", RUNS[0]), map_location="cuda", weights_only=False)
bb = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).backbone.cuda().eval()
feats = []
with torch.no_grad():
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats.append(bb(batch["pixel_values"].cuda()).cpu())
feats = torch.cat(feats)
del bb
torch.cuda.empty_cache()
print("cached", tuple(feats.shape), flush=True)
for name in RUNS:
    c = torch.load(latest_ckpt("checkpoints/coco", name), map_location="cuda", weights_only=False)
    model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).cuda()
    model.load_state_dict(c["model"])
    model.backbone = torch.nn.Identity()  # inputs are the cached backbone features
    model.eval()
    g = model.grid_size
    dec = model.decoder
    set_seed(0)
    err = {b: [] for b in BS}
    masks = {b: [] for b in BS}
    with torch.no_grad():
        for i in range(0, len(feats), B):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(feats[i:i + B].cuda())
                y = out["slots_list"][-1]
                n, k, _ = y.shape
                x = y.unsqueeze(2).expand(n, k, dec.num_patches, y.shape[-1]) + dec.pos_emb
                rk, al = dec.mlp(x).split([dec.feature_dim, 1], dim=-1)
            t = out["target"].float()
            rk, al = rk.float(), al.float().squeeze(-1)  # (n, K, N, D), (n, K, N)
            d2 = d2_main_piece(al, g)
            for b in BS:
                m = (al - b * d2).softmax(1)
                r = (rk * m.unsqueeze(-1)).sum(1)
                err[b].append((r - t).pow(2).mean(dim=(1, 2)).cpu().numpy())
                masks[b].append(m.reshape(n, 1, k, g, g).cpu().numpy().astype(np.float16))
    for b in BS:
        e = np.concatenate(err[b])[:, None]
        if b:
            np.savez(f"{C}/rx_{name}_loc{b}.npz", mse=e, masks=np.concatenate(masks[b]))
        print(name, f"loc{b}", "mse", e.mean().round(5), flush=True)
    del model
    torch.cuda.empty_cache()
print("loc_mse done", flush=True)
