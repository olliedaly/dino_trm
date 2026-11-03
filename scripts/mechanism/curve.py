"""Learning curves for T=1 vs T=8: is T=8's higher reconstruction error a training-budget
artifact (still falling at epoch 29) and is it a fit (train) or a generalisation (val) gap?

Backbone features are frozen, so they are cached once for the first N val images and the
first N train images (no augmentation), then every saved epoch checkpoint is run on them.
Writes c/curve.npz:
  {run}|e{epoch}|{val,train}|mse   (N, T) per-image MSE after every step
  {run}|e{epoch}|val|masks         (N, K, g, g) f16 final-step decoder masks
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


def cache(split):
    loader = build_coco_loader(split=split, batch_size=B, num_workers=4, shuffle=False,
                               augment=False, return_masks=False)
    c = torch.load(f"checkpoints/coco/{RUNS[0]}_epoch{EPOCHS[-1]}.pt", map_location="cuda", weights_only=False)
    bb = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).backbone.cuda().eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(bb(batch["pixel_values"].cuda()).cpu())
            if sum(len(o) for o in out) >= N:
                break
    del bb
    torch.cuda.empty_cache()
    return torch.cat(out)[:N]


FEATS = {s: cache(s) for s in ("val", "train")}
print("cached", {k: tuple(v.shape) for k, v in FEATS.items()}, flush=True)
res = {}
for name in RUNS:
    for e in EPOCHS:
        c = torch.load(f"checkpoints/coco/{name}_epoch{e}.pt", map_location="cuda", weights_only=False)
        model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).cuda()
        model.load_state_dict(c["model"])
        model.backbone = torch.nn.Identity()  # inputs are the cached backbone features
        model.eval()
        g = model.grid_size
        for split, f in FEATS.items():
            set_seed(0)
            err, masks = [], []
            with torch.no_grad():
                for i in range(0, N, B):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model(f[i:i + B].cuda())
                    t = out["target"].float()
                    b = t.shape[0]
                    err.append(torch.stack([(r.float() - t).pow(2).mean(dim=(1, 2)) for r in out["recons"]], 1).cpu().numpy())
                    if split == "val":
                        masks.append(out["masks_list"][-1].float().reshape(b, -1, g, g).cpu().numpy().astype(np.float16))
            res[f"{name}|e{e}|{split}|mse"] = np.concatenate(err)
            if split == "val":
                res[f"{name}|e{e}|val|masks"] = np.concatenate(masks)
        print(name, e, "val", res[f"{name}|e{e}|val|mse"][:, -1].mean().round(4),
              "train", res[f"{name}|e{e}|train|mse"][:, -1].mean().round(4), flush=True)
        del model
        torch.cuda.empty_cache()
np.savez(f"{C}/curve{os.environ.get('CURVE_TAG', '')}.npz", **res)
print("curve done", flush=True)
