"""Cache final-step low-res masks (eval seed 0) for a list of runs, plus the GT once."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from extract_per_image import latest_ckpt, set_seed  # noqa: E402
from dino_trm.data.coco import build_coco_loader  # noqa: E402
from dino_trm.models.full_model import DinoSlotModel  # noqa: E402

OUT = sys.argv[1]
RUNS = sys.argv[2].split(",")
loader = build_coco_loader(split="val", batch_size=16, num_workers=4, shuffle=False,
                           return_masks=True)
os.makedirs(OUT, exist_ok=True)
for name in RUNS:
    c = torch.load(latest_ckpt("checkpoints/coco", name), map_location="cuda", weights_only=False)
    model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).cuda()
    model.load_state_dict(c["model"])
    model.eval()
    g = model.backbone.grid_size
    set_seed(0)
    masks, inst, sem = [], [], []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch["pixel_values"].cuda())
            m = out["masks_list"][-1]
            masks.append(m.float().view(m.shape[0], m.shape[1], g, g).cpu().numpy())
            if not os.path.exists(f"{OUT}/gt.npz"):
                inst.append(batch["inst_full"].numpy().astype(np.uint16))
                s = batch["label_full"].numpy().copy()
                s[s == 255] = 0
                sem.append(s.astype(np.uint16))
    np.save(f"{OUT}/m_{name}.npy", np.concatenate(masks))
    if inst:
        np.savez_compressed(f"{OUT}/gt.npz", inst=np.concatenate(inst), sem=np.concatenate(sem))
    print(name, "done", np.concatenate(masks).shape, flush=True)
