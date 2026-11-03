"""Per-patch reconstruction error (final step, eval seed 0) -> c/pe_{run}.npy (N, 441) f16."""
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from extract_per_image import latest_ckpt, set_seed  # noqa: E402
from dino_trm.data.coco import build_coco_loader  # noqa: E402
from dino_trm.models.full_model import DinoSlotModel  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.2)
C, RUNS = sys.argv[1], sys.argv[2].split(",")
loader = build_coco_loader(split="val", batch_size=16, num_workers=4, shuffle=False, return_masks=True)
for name in RUNS:
    c = torch.load(latest_ckpt("checkpoints/coco", name), map_location="cuda", weights_only=False)
    model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).cuda()
    model.load_state_dict(c["model"])
    model.eval()
    set_seed(0)
    err = []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch["pixel_values"].cuda())
            err.append((out["recons"][-1].float() - out["target"].float()).pow(2).mean(-1).cpu().numpy().astype(np.float16))
    np.save(f"{C}/pe_{name}.npy", np.concatenate(err))
    print(name, "done", flush=True)
    del model
    torch.cuda.empty_cache()
