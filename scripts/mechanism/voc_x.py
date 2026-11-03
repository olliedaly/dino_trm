"""VOC val (eval seed 0): final decoder masks + per-image reconstruction MSE for COCO-trained
and VOC-trained T=1/T=8 models, and the GT (void -> background, connected-component
instances >= 200 px, as in evaluate_protocol). Writes c/voc_gt.npz, c/voc_{tag}.npz."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from extract_per_image import _sem_inst, latest_ckpt, set_seed  # noqa: E402
from dino_trm.data.pascal_voc import build_loader  # noqa: E402
from dino_trm.models.full_model import DinoSlotModel  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.2)
C = sys.argv[1]
RUNS = [("coco", r) for r in ["trm_t1", "trm_t1_seed1", "trm_t1_seed2", "trm", "trm_seed1", "trm_seed2"]] + \
       [("voc", r) for r in ["trm_t1_m", "trm_m", "baseline_pm_m"]]
loader = build_loader(split="val", batch_size=16, num_workers=4, shuffle=False, full_mask=True)
if not os.path.exists(f"{C}/voc_gt.npz"):
    sems, insts = [], []
    for batch in loader:
        for b in range(batch["pixel_values"].shape[0]):
            sem, inst = _sem_inst(batch, b)
            sems.append(sem.astype(np.uint8))
            insts.append(inst.astype(np.int16))
    np.savez_compressed(f"{C}/voc_gt.npz", sem=np.stack(sems), inst=np.stack(insts))
    print("voc gt", len(sems), flush=True)
for ds, name in RUNS:
    c = torch.load(latest_ckpt("checkpoints/coco" if ds == "coco" else "checkpoints", name),
                   map_location="cuda", weights_only=False)
    model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).cuda()
    model.load_state_dict(c["model"])
    model.eval()
    g = model.backbone.grid_size
    set_seed(0)
    masks, err = [], []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch["pixel_values"].cuda())
            b = out["target"].shape[0]
            masks.append(out["masks_list"][-1].float().reshape(b, -1, g, g).cpu().numpy().astype(np.float16))
            err.append((out["recons"][-1].float() - out["target"].float()).pow(2).mean(dim=(1, 2)).cpu().numpy())
    tag = f"{ds}_{name}"
    np.savez(f"{C}/voc_{tag}.npz", masks=np.concatenate(masks), mse=np.concatenate(err))
    print(tag, c["cfg"]["mode"], "n_steps", c["cfg"]["model"]["n_steps"], "xattn", c["cfg"]["model"]["gnn_cross_attn"],
          "MSE", np.concatenate(err).mean().round(4), flush=True)
    del model
    torch.cuda.empty_cache()
