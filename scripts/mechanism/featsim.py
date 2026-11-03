"""Does slot sharing follow feature similarity? Ground-truth-free test of appearance grouping.

For every pair of tokens in an image (21x21 grid), the cosine similarity of their DINO
backbone features and their distance. For each model's final decoder masks (argmax),
the share of pairs in one slot, binned by distance and by feature similarity.
All val images, eval seed 0, 3 seeds per model. Writes c/featsim.npz: for each model,
shared and total pair counts (n_dist, n_sim); prints the sharing rate table and, per
distance, the rise in sharing from the least to the most similar pairs.
Caps this process's GPU memory so a concurrent training run keeps its headroom."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from extract_per_image import latest_ckpt  # noqa: E402
from dino_trm.data.coco import build_coco_loader  # noqa: E402
from dino_trm.models.full_model import DinoSlotModel  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.2)
C = sys.argv[1]
SEEDS = ["", "_seed1", "_seed2"]
RUNS = {"baseline": "baseline_pm{s}", "mlp_block": "mlp_block{s}", "T=1": "trm_t1{s}", "T=8": "trm{s}"}
if len(sys.argv) > 2 and sys.argv[2]:  # extra runs as name=pattern, {s} = seed suffix
    RUNS.update(dict(x.split("=") for x in sys.argv[2].split(",")))
SIM = np.array([-1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01])
DIST = np.array([0, 3, 7, 14, 99])
g = 21
LAB = {}
for name, run in RUNS.items():
    for s in SEEDS:
        if not os.path.exists(f"{C}/rx_{run.format(s=s)}.npz"):
            continue
        m = np.load(f"{C}/rx_{run.format(s=s)}.npz")["masks"][:, -1].astype(np.float32)
        LAB[(name, s)] = torch.from_numpy(m.reshape(m.shape[0], m.shape[1], -1).argmax(1)).cuda()  # (N, 441)
yy, xx = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
pos = torch.stack([yy, xx], -1).reshape(-1, 2).float().cuda()
iu = torch.triu_indices(g * g, g * g, 1).cuda()
dist = (pos[iu[0]] - pos[iu[1]]).norm(dim=-1)
dbin = torch.bucketize(dist, torch.tensor(DIST[1:-1], dtype=torch.float32).cuda(), right=False)  # (0,3],(3,7],(7,14],>14
simedges = torch.tensor(SIM[1:-1], dtype=torch.float32).cuda()
nd, ns = len(DIST) - 1, len(SIM) - 1
tot = torch.zeros(nd * ns, dtype=torch.float64).cuda()
shared = {k: torch.zeros(nd * ns, dtype=torch.float64).cuda() for k in LAB}

c = torch.load(latest_ckpt("checkpoints/coco", "trm"), map_location="cuda", weights_only=False)
bb = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).backbone.cuda().eval()
loader = build_coco_loader(split="val", batch_size=16, num_workers=4, shuffle=False, return_masks=False)
i0 = 0
with torch.no_grad():
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = bb(batch["pixel_values"].cuda()).float()
        f = torch.nn.functional.normalize(f, dim=-1)
        for b in range(f.shape[0]):
            sim = (f[b] @ f[b].T)[iu[0], iu[1]]
            cell = dbin * ns + torch.bucketize(sim, simedges, right=False)
            tot += torch.bincount(cell, minlength=nd * ns).double()
            for k, lab in LAB.items():
                L = lab[i0 + b]
                same = L[iu[0]] == L[iu[1]]
                shared[k] += torch.bincount(cell[same], minlength=nd * ns).double()
        i0 += f.shape[0]
tot = tot.cpu().numpy().reshape(nd, ns)
rate = {k: v.cpu().numpy().reshape(nd, ns) / np.maximum(tot, 1) for k, v in shared.items()}
np.savez(f"{C}/featsim{os.environ.get('FEATSIM_TAG', '')}.npz", tot=tot, sim_edges=SIM, dist_edges=DIST,
         **{f"{n}{s}": rate[(n, s)] for n, s in rate})
print(f"{i0} images. pairs per cell (millions):")
print(np.round(tot / 1e6, 2))
lab_s = [f"{SIM[j]:.1f}-{SIM[j + 1]:.1f}" for j in range(ns)]
for name in RUNS:
    r = np.mean([rate[(name, s)] for s in SEEDS if (name, s) in rate], 0)
    print(f"\n{name}: share of token pairs in one slot (rows: distance <=3 / 3-7 / 7-14 / >14 tokens; cols: feature cosine)")
    print("            " + " ".join(f"{x:>8s}" for x in lab_s))
    for d in range(nd):
        print(f"   dist {d}   " + " ".join(f"{x:8.3f}" for x in r[d]))
print("\nrise in sharing from cosine 0.1-0.2 to 0.6-0.7 pairs, by distance (3-seed mean; per seed in brackets):")
for name in RUNS:
    per = np.array([rate[(name, s)][:, 6] - rate[(name, s)][:, 1] for s in SEEDS if (name, s) in rate])
    print(f"   {name:9s} " + "  ".join(f"{per[:, d].mean():+.3f} ({' '.join(f'{x:+.3f}' for x in per[:, d])})" for d in range(nd)))
