"""How well do the slots explain the image? Feature-reconstruction MSE on all val images
(eval seed 0), per recursion step, plus the variance of the DINO targets for scale."""
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from extract_per_image import latest_ckpt, set_seed  # noqa: E402
from dino_trm.data.coco import build_coco_loader  # noqa: E402
from dino_trm.models.full_model import DinoSlotModel  # noqa: E402

loader = build_coco_loader(split="val", batch_size=16, num_workers=4, shuffle=False, return_masks=True)
for name in sys.argv[1].split(","):
    c = torch.load(latest_ckpt("checkpoints/coco", name), map_location="cuda", weights_only=False)
    model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).cuda()
    model.load_state_dict(c["model"])
    model.eval()
    set_seed(0)
    err, var = [], []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch["pixel_values"].cuda())
            t = out["target"].float()
            err.append(torch.stack([(r.float() - t).pow(2).mean(dim=(1, 2)) for r in out["recons"]], 1).cpu())
            var.append(t.var(dim=1).mean(dim=1).cpu())
    e, v = torch.cat(err).numpy(), torch.cat(var).numpy()
    print(f"{name:20s} dec_hidden {c['cfg']['model']['decoder_hidden']:5d} | MSE per step "
          + " ".join(f"{x:.4f}" for x in e.mean(0)) + f" | target variance {v.mean():.4f}", flush=True)
