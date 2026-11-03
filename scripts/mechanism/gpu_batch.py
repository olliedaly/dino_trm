"""One GPU pass per model over all val images (eval seed 0, batch 16 as in the cached masks).
Saves c/rx_{run}.npz: per-image reconstruction MSE per step (N, T), the slots after every
step (N, T, K, D) f16, and the decoder masks after every step (N, T, K, g, g) f16.
EVAL_SEED=s (env) changes the slot-init noise and adds _e{s} to the tag. CKPT_DIR (env, default
checkpoints/coco) is where checkpoints are found; RX_PREFIX (env) is put before the run name in the output. With --steps n (trm only) the module is run for n steps at test time -> c/rx_{run}_n{n}.npz.
Caps this process's GPU memory so a concurrent training run keeps its headroom."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from extract_per_image import latest_ckpt, set_seed  # noqa: E402
from dino_trm.data.coco import build_coco_loader  # noqa: E402
from dino_trm.models.full_model import DinoSlotModel  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.2)
C, RUNS = sys.argv[1], sys.argv[2].split(",")
EVAL_SEED = int(os.environ.get("EVAL_SEED", "0"))
CKPT_DIR, PREFIX = os.environ.get("CKPT_DIR", "checkpoints/coco"), os.environ.get("RX_PREFIX", "")
STEPS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [None]
loader = build_coco_loader(split="val", batch_size=16, num_workers=4, shuffle=False, return_masks=True)
for name in RUNS:
    c = torch.load(latest_ckpt(CKPT_DIR, name), map_location="cuda", weights_only=False)
    model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).cuda()
    model.load_state_dict(c["model"])
    model.eval()
    g = model.backbone.grid_size
    for n in STEPS:
        if n is not None:
            model.reasoner.n_steps = n
        set_seed(EVAL_SEED)
        err, slots, masks = [], [], []
        with torch.no_grad():
            for batch in loader:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(batch["pixel_values"].cuda())
                t = out["target"].float()
                b = t.shape[0]
                err.append(torch.stack([(r.float() - t).pow(2).mean(dim=(1, 2)) for r in out["recons"]], 1).cpu().numpy())
                slots.append(torch.stack([s.float() for s in out["slots_list"]], 1).cpu().numpy().astype(np.float16))
                masks.append(torch.stack([m.float().reshape(b, -1, g, g) for m in out["masks_list"]], 1).cpu().numpy().astype(np.float16))
        tag = PREFIX + (name if n is None else f"{name}_n{n}") + (f"_e{EVAL_SEED}" if EVAL_SEED else "")
        np.savez(f"{C}/rx_{tag}.npz", mse=np.concatenate(err), slots=np.concatenate(slots), masks=np.concatenate(masks))
        print(tag, "mse per step", np.concatenate(err).mean(0).round(4), flush=True)
    del model
    torch.cuda.empty_cache()
