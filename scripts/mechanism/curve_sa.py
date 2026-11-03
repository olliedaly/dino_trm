"""Slot attention's own map at every saved epoch (first N val images, eval seed 0), to time
when it stops segmenting in T=8 against the rise in T=8's reconstruction error.
Writes c/curve_sa.npz: {run}|e{epoch}|attn (N, K, g, g) f16.
Caps this process's GPU memory so a concurrent training run keeps its headroom."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from extract_per_image import set_seed  # noqa: E402
from dino_trm.data.coco import build_coco_loader  # noqa: E402
from dino_trm.models.full_model import DinoSlotModel  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.2)
C = sys.argv[1]
RUNS = sys.argv[2].split(",")
EPOCHS = [int(e) for e in sys.argv[3].split(",")]
N = 512
B = 16
loader = build_coco_loader(split="val", batch_size=B, num_workers=4, shuffle=False, augment=False, return_masks=False)
c = torch.load(f"checkpoints/coco/{RUNS[0]}_epoch{EPOCHS[-1]}.pt", map_location="cuda", weights_only=False)
bb = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).backbone.cuda().eval()
feats = []
with torch.no_grad():
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats.append(bb(batch["pixel_values"].cuda()).cpu())
        if sum(len(f) for f in feats) >= N:
            break
feats = torch.cat(feats)[:N]
del bb
torch.cuda.empty_cache()
res = {}
for name in RUNS:
    for e in EPOCHS:
        c = torch.load(f"checkpoints/coco/{name}_epoch{e}.pt", map_location="cuda", weights_only=False)
        model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).cuda()
        model.load_state_dict(c["model"])
        model.backbone = torch.nn.Identity()  # inputs are the cached backbone features
        model.eval()
        g = model.grid_size
        set_seed(0)
        attn = []
        with torch.no_grad():
            for i in range(0, N, B):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(feats[i:i + B].cuda())
                a = out["attn"].float()
                attn.append(a.reshape(a.shape[0], -1, g, g).cpu().numpy().astype(np.float16))
        res[f"{name}|e{e}|attn"] = np.concatenate(attn)
        print(name, e, flush=True)
        del model
        torch.cuda.empty_cache()
np.savez(f"{C}/curve_sa{os.environ.get('CURVE_TAG', '')}.npz", **res)
print("curve_sa done", flush=True)
