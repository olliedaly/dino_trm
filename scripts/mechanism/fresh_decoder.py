"""Is the locality in the slots or in the decoder? Fresh-decoder probe.

Freezes a trained model (backbone, slot attention, module), replaces its decoder with a
freshly initialised one of the same architecture, and trains only the new decoder to
reconstruct the DINO features from the model's FINAL slots (T=8: after step 8; T=1: after
its one step), with the training recipe's loss and optimiser (MSE, AdamW 4e-4, warmup 300,
cosine, grad clip 1.0, bf16), batch 16, for E epochs on the COCO train split.
Then decodes all 2,000 val images (eval seed 0, as gpu_batch.py) and saves
c/rx_{run}_fd.npz: mse (N, 1), masks (N, 1, K, g, g) f16.
A fresh decoder trained on T=8's slots that groups by appearance would mean T=8's slots
still carry the grouping and its own decoder suppresses it; one that stays local means the
locality is in the slots.
Usage: fresh_decoder.py C run [epochs]
Caps this process's GPU memory so a concurrent training run keeps its headroom."""
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from extract_per_image import latest_ckpt, set_seed  # noqa: E402
from dino_trm.data.coco import build_coco_loader  # noqa: E402
from dino_trm.models.decoder import SpatialBroadcastDecoder  # noqa: E402
from dino_trm.models.full_model import DinoSlotModel  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.2)
C, RUN = sys.argv[1], sys.argv[2]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
c = torch.load(latest_ckpt("checkpoints/coco", RUN), map_location="cuda", weights_only=False)
model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).cuda()
model.load_state_dict(c["model"])
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
old = model.decoder
torch.manual_seed(1234)
dec = SpatialBroadcastDecoder(slot_dim=old.pos_emb.shape[-1], feature_dim=old.feature_dim, num_patches=old.num_patches,
                              hidden_dim=old.mlp[0].out_features, n_layers=len([m for m in old.mlp if isinstance(m, torch.nn.Linear)])).cuda()
g = model.grid_size

train = build_coco_loader(split="train", batch_size=16, num_workers=4, shuffle=True, return_masks=False)
opt = torch.optim.AdamW(dec.parameters(), lr=4e-4, weight_decay=0.0)
total, warm = EPOCHS * len(train), 300
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))


def final_slots(pix):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(pix)
    return out["slots_list"][-1].float(), out["target"].float()


t0 = time.time()
for ep in range(EPOCHS):
    dec.train()
    run_loss = []
    for it, batch in enumerate(train):
        slots, target = final_slots(batch["pixel_values"].cuda())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            recon, _ = dec(slots)
        loss = F.mse_loss(recon.float(), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss.append(loss.item())
    print(f"{RUN} fresh decoder epoch {ep}: train mse {np.mean(run_loss[-200:]):.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)

dec.eval()
val = build_coco_loader(split="val", batch_size=16, num_workers=4, shuffle=False, return_masks=True)
set_seed(0)
err, masks = [], []
with torch.no_grad():
    for batch in val:
        slots, target = final_slots(batch["pixel_values"].cuda())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            recon, m = dec(slots)
        b = target.shape[0]
        err.append((recon.float() - target).pow(2).mean(dim=(1, 2)).cpu().numpy()[:, None])
        masks.append(m.float().reshape(b, -1, g, g).cpu().numpy().astype(np.float16)[:, None])
np.savez(f"{C}/rx_{RUN}_fd.npz", mse=np.concatenate(err), masks=np.concatenate(masks))
print(f"{RUN} fresh decoder: val mse {np.concatenate(err).mean():.4f}", flush=True)
