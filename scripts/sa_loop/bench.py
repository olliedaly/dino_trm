"""Speed and memory of one training recipe, on random images (docs/sa_loop_plan.md).

Runs a few real training steps (forward, loss, backward, optimiser step) under the
trainer's bf16 autocast and prints seconds per step and the peak GPU memory. Use it
before launching a recipe that has not been run on this machine.

    source scripts/sa_loop/env.sh && uv run python scripts/sa_loop/bench.py \
        --mode sa_rec --slot-iters 8 --sa-detach --image-size 448 --batch 16
"""
import argparse
import time

import torch
from omegaconf import OmegaConf

from dino_trm.losses import total_training_loss
from dino_trm.models.full_model import DinoSlotModel

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="baseline", choices=["baseline", "sa_rec"])
ap.add_argument("--slot-iters", type=int, default=3)
ap.add_argument("--sa-detach", action="store_true")
ap.add_argument("--sa-sample-pass", action="store_true")
ap.add_argument("--image-size", type=int, default=336)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--decoder-hidden", type=int, default=1469)
ap.add_argument("--steps", type=int, default=25)
a = ap.parse_args()

cfg = OmegaConf.load("configs/base.yaml")
cfg.model.update(slot_iters=a.slot_iters, sa_detach=a.sa_detach, sa_sample_pass=a.sa_sample_pass,
                 image_size=a.image_size,
                 decoder_hidden=a.decoder_hidden)
model = DinoSlotModel(mode=a.mode, **cfg.model).cuda().train()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=cfg.optim.lr)
px = torch.randn(a.batch, 3, a.image_size, a.image_size, device="cuda")
torch.cuda.reset_peak_memory_stats()
times = []
for step in range(a.steps):
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(px)
    loss, _ = total_training_loss(out, intermediate_weight=cfg.loss.intermediate_weight,
                                  final_weight=cfg.loss.final_weight)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    times.append(time.time() - t0)
per = sum(times[5:]) / len(times[5:])
print(f"mode={a.mode} passes={a.slot_iters} detach={a.sa_detach} sample={a.sa_sample_pass} image={a.image_size} "
      f"batch={a.batch} | "
      f"{sum(p.numel() for p in params) / 1e6:.4f}M parameters | {model.num_patches} patches | "
      f"{per:.3f} s/step | peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB "
      f"(the card has 16 GB; keep the peak under 13 GB)")
