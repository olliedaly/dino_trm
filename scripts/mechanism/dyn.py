"""CPU check of slot-attention dynamics per iteration: does each run converge to a fixed point?
For each checkpoint: relative slot update, share of patches keeping their argmax slot from the
previous iteration (attention map), and the largest slot's share of the image."""
import sys
import numpy as np
import torch

sys.path.insert(0, "scripts")
from extract_per_image import latest_ckpt, set_seed  # noqa: E402
from dino_trm.data.coco import build_coco_loader  # noqa: E402
from dino_trm.models.full_model import DinoSlotModel  # noqa: E402

torch.set_num_threads(4)
loader = build_coco_loader(split="val", batch_size=24, num_workers=0, shuffle=False, return_masks=True)
x = next(iter(loader))["pixel_values"]
RUNS = [("baseline_pm (8 it at test)", latest_ckpt("checkpoints/coco", "baseline_pm"), 8),
        ("sa_rec ep0", "checkpoints/coco/sa_rec_epoch0.pt", 8),
        ("sa_rec ep4", "checkpoints/coco/sa_rec_epoch4.pt", 8),
        ("sa_rec ep6", "checkpoints/coco/sa_rec_epoch6.pt", 8)]
if len(sys.argv) > 1:  # name=checkpoint pairs, 8 iterations each
    RUNS = [(a.split("=")[0], a.split("=")[1], 8) for a in sys.argv[1:]]
for name, path, iters in RUNS:
    c = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(c["cfg"]["model"]); cfg["slot_iters"] = iters
    model = DinoSlotModel(mode=c["cfg"]["mode"], **cfg)
    model.load_state_dict(c["model"]); model.eval()
    set_seed(0)
    with torch.no_grad():
        proj = model.input_proj(model.backbone(x).float())
        S, A, _ = model.slot_attention(proj, return_all=True)
    am = [a.argmax(1) for a in A]                       # (B, N) argmax slot per patch
    upd = [((S[t] - S[t - 1]).norm(dim=-1) / S[t - 1].norm(dim=-1)).mean().item() for t in range(1, len(S))]
    keep = [(am[t] == am[t - 1]).float().mean().item() for t in range(1, len(S))]
    big = [np.mean([torch.bincount(r, minlength=7).max().item() / r.numel() for r in a]) for a in am]
    print(f"{name:28s} update |s_t-s_t-1|/|s_t-1| " + " ".join(f"{u:.2f}" for u in upd))
    print(f"{'':28s} patches keeping slot        " + " ".join(f"{k:.2f}" for k in keep))
    print(f"{'':28s} largest slot share (it1..8) " + " ".join(f"{b:.2f}" for b in big), flush=True)
