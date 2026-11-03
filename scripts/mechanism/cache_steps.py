"""Cache the low-res masks of every recursion step (eval seed 0): m_{run}_steps.npy (N, T, K, g, g)."""
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from extract_per_image import latest_ckpt, set_seed  # noqa: E402
from dino_trm.data.coco import build_coco_loader  # noqa: E402
from dino_trm.models.full_model import DinoSlotModel  # noqa: E402

OUT, RUNS = sys.argv[1], sys.argv[2].split(",")
loader = build_coco_loader(split="val", batch_size=16, num_workers=4, shuffle=False,
                           return_masks=True)
for name in RUNS:
    c = torch.load(latest_ckpt("checkpoints/coco", name), map_location="cuda", weights_only=False)
    model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).cuda()
    model.load_state_dict(c["model"])
    model.eval()
    g = model.backbone.grid_size
    set_seed(0)
    masks = []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch["pixel_values"].cuda())
            m = torch.stack([x.float() for x in out["masks_list"]], 1)   # (B, T, K, N)
            masks.append(m.view(*m.shape[:3], g, g).cpu().numpy().astype(np.float32))
    np.save(f"{OUT}/m_{name}_steps.npy", np.concatenate(masks))
    print(name, "done", flush=True)
