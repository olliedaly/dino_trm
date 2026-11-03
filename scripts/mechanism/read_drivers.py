"""What drives the module's reads: appearance or position?

For every slot's cross-attention over the 441 patches (c/sa_{run}.npz 'ca', mean of the 4
cross-attention calls in a step; first and last step), regress log attention on two
standardised predictors, per slot:
  appearance  cosine between the patch's DINO feature and the slot's read content
              (attention-weighted mean feature)
  position    minus the squared distance from the patch to the slot's read centroid
Reports the mean standardised coefficients and the share of variance each explains alone
(squared correlation), over all slots of all 2,000 val images, eval seed 0.
Also the same for slot attention's own map (softmax over slots, 'attn') as a reference.
Usage: read_drivers.py C runs   (runs from sa_{run}.npz, e.g. trm_t1,trm,mlp_block)
Caps this process's GPU memory so a concurrent training run keeps its headroom."""
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from extract_per_image import latest_ckpt  # noqa: E402
from dino_trm.data.coco import build_coco_loader  # noqa: E402
from dino_trm.models.full_model import DinoSlotModel  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.2)
C, RUNS = sys.argv[1], sys.argv[2].split(",")
g = 21
yy, xx = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
POS = torch.stack([yy, xx], -1).reshape(-1, 2).float().cuda()
READS = {}
for run in RUNS:
    z = np.load(f"{C}/sa_{run}.npz")
    ca = z["ca"]  # (N, 2, 4, K, P)
    READS[(run, "first")] = ca[:, 0].astype(np.float32).mean(1)
    if run not in ("trm_t1",) and not run.startswith(("trm_t1_", "mlp_block")):
        READS[(run, "last")] = ca[:, -1].astype(np.float32).mean(1)
    READS[(run, "slot attention map")] = z["attn"].astype(np.float32).reshape(len(ca), 7, -1)
acc = {k: np.zeros(4) for k in READS}  # sum of b_app, b_pos, r2_app, r2_pos
cnt = {k: 0 for k in READS}


def zs(v):
    return (v - v.mean(-1, keepdim=True)) / (v.std(-1, keepdim=True) + 1e-6)


c = torch.load(latest_ckpt("checkpoints/coco", "trm"), map_location="cuda", weights_only=False)
bb = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).backbone.cuda().eval()
loader = build_coco_loader(split="val", batch_size=16, num_workers=4, shuffle=False, return_masks=False)
i0 = 0
with torch.no_grad():
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = bb(batch["pixel_values"].cuda()).float()
        f = torch.nn.functional.normalize(f, dim=-1)  # (B, P, D)
        b = f.shape[0]
        for key, arr in READS.items():
            a = torch.from_numpy(arr[i0:i0 + b]).cuda()  # (B, K, P)
            a = a / a.sum(-1, keepdim=True)
            content = torch.nn.functional.normalize(torch.einsum("bkp,bpd->bkd", a, f), dim=-1)
            app = torch.einsum("bkd,bpd->bkp", content, f)
            cen = torch.einsum("bkp,pc->bkc", a, POS)
            pos = -((POS[None, None] - cen[:, :, None]) ** 2).sum(-1)
            y = zs(torch.log(a + 1e-6))
            X1, X2 = zs(app), zs(pos)
            r12 = (X1 * X2).mean(-1)
            ry1, ry2 = (y * X1).mean(-1), (y * X2).mean(-1)
            den = (1 - r12 ** 2).clamp_min(1e-6)
            b1, b2 = (ry1 - r12 * ry2) / den, (ry2 - r12 * ry1) / den
            ok = torch.isfinite(b1) & torch.isfinite(b2)
            acc[key] += np.array([b1[ok].sum().item(), b2[ok].sum().item(), (ry1[ok] ** 2).sum().item(), (ry2[ok] ** 2).sum().item()])
            cnt[key] += int(ok.sum())
        i0 += b
print(f"{i0} images. Per slot: log attention ~ appearance + position (standardised); mean over slots")
print(f"{'run':16s} {'read':20s} {'coef appearance':>16s} {'coef position':>14s} {'R2 appearance':>14s} {'R2 position':>12s}")
for (run, step), v in acc.items():
    m = v / cnt[(run, step)]
    print(f"{run:16s} {step:20s} {m[0]:16.3f} {m[1]:14.3f} {m[2]:14.3f} {m[3]:12.3f}")
np.savez(f"{C}/read_drivers.npz", keys=[f"{r}|{s}" for r, s in acc], vals=np.array([acc[k] / cnt[k] for k in acc]))
