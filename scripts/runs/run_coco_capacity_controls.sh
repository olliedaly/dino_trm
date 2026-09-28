#!/usr/bin/env bash
cd "$(dirname "$0")/../.."   # run from the repo root
# COCO multi-object subset — the two decisive capacity controls.
#
# MOTIVATION (in-training eval, 15 val batches, epoch 29, single seed):
#   | run                                   | params | mBO_i | mBO_c | FG-ARI |
#   |---------------------------------------|--------|-------|-------|--------|
#   | baseline (ours, decoder_hidden=1024)  | 3.89M  | 22.29 | 28.41 | 42.83  |
#   | baseline seed=1 (train-noise control) | 3.89M  | 22.30 | 28.14 | 43.26  |
#   | baseline num_slots=9                  | 3.89M  | 23.89 | 28.54 | 42.46  |
#   | baseline num_slots=11                 | 3.89M  | 24.14 | 27.92 | 39.85  |
#   | baseline num_slots=15                 | 3.89M  | 25.31 | 27.83 | 36.95  |
#   | baseline decoder_hidden=2048          | 10.84M | 22.78 | 28.62 | 45.48  |
#   | baseline full-DINOSAUR-fidelity       | 11.25M | 22.84 | 28.84 | 46.98  |
#   | trm + xattn (headline)                | 6.40M  | 24.32 | 30.88 | 46.68  |
#   | coupled + xattn (headline)            | 6.40M  | 24.24 | 30.50 | 47.10  |
#
# Slot count trades mBO_i UP against FG-ARI DOWN and never recovers FG-ARI; decoder
# capacity recovers essentially all of the FG-ARI gain. But the two capacity baselines
# (10.84M / 11.25M) are one-sided (over-provisioned) controls vs the recursive models'
# 6.40M. These two runs close that gap:
#
#   RUN A (baseline_pm) — PARAMETER-MATCHED baseline. mode=baseline with
#     decoder_hidden=1469 (everything else at defaults, 7 slots) gives 6.3967M
#     trainable params, the closest match to the recursive models' 6.3960M
#     (residual +679 params, +0.011%). This is the parameter-matched (rather than
#     one-sided) capacity control: at EQUAL parameter count, does the plain baseline
#     already match trm/coupled on FG-ARI?
#
#   RUN B (trm_dec2048) — RECURSION ON TOP OF FAITHFUL CAPACITY. mode=trm with
#     gnn_cross_attn=true and decoder_hidden=2048 (7 slots): once the decoder has
#     DINOSAUR's official width, does the recursion still add anything on top?
#
#   RUN C (coupled_dec2048) — COUPLED (top-down rebinding) at faithful capacity.
#     Same as Run B but mode=coupled + query_ortho_weight=0.1: does the rebinding
#     feedback add beyond plain recursion once decoder capacity is matched?
#
# MEASURED PARAM COUNTS (336px, 7 slots, requires_grad only; frozen DINOv3 backbone
# params have requires_grad=False and are excluded exactly as train.py does):
#   baseline decoder_hidden=1469  6.3967M   (Run A; best match to 6.3960M)
#   trm + xattn decoder_hidden=2048  13.3469M  (Run B)
#
# VRAM (Run B, one fwd+bwd, batch 32 / 336px, real total_training_loss): measured
# model-own peaks 0.949GB @batch1 and 1.788GB @batch2 (fixed overhead 0.098GB,
# ~0.84GB/sample) extrapolate to ~27GB @batch32 — does NOT fit in ~14.5GB. A
# micro-batch of 16 would land at ~13.5GB, too close to the 16GB card's limit for an
# unattended run, so Run B uses data.batch_size=8 with optim.grad_accum=4 (~6.8GB)
# to preserve an effective batch of 32. steps_per_epoch = len(loader)//grad_accum, so
# the optimizer-step count and LR schedule are identical to every other run.
# Because the micro-batch is 8 rather than 32, eval.max_batches is raised 15 -> 60 so
# the in-training eval still covers the same first 480 val images (val shuffle=False),
# keeping the monitoring numbers comparable across runs.
#
# Same recipe as run_coco.sh (30 epochs, lr 4e-4, COCO 25k subset). Each run uses
# cfg.log.run_name so checkpoints save under baseline_pm_epoch*.pt and
# trm_dec2048_epoch*.pt without overwriting the canonical baseline_epoch*.pt.
set -e
COMMON="data.dataset=coco log.wandb=false optim.epochs=30 optim.warmup_steps=300 \
eval.max_batches=15 log.ckpt_dir=checkpoints/coco"

echo "=== RUN A: COCO BASELINE, param-matched (decoder_hidden=1469, ~6.397M params, 30ep) ==="
uv run python -m dino_trm.train mode=baseline $COMMON model.decoder_hidden=1469 log.run_name=baseline_pm

echo "=== RUN B: COCO TRM + xattn, decoder_hidden=2048 (~13.347M params, 30ep, micro-batch 8 x grad_accum 4) ==="
uv run python -m dino_trm.train mode=trm $COMMON model.gnn_cross_attn=true model.decoder_hidden=2048 data.batch_size=8 optim.grad_accum=4 eval.max_batches=60 log.run_name=trm_dec2048

echo "=== RUN C: COCO COUPLED + xattn, decoder_hidden=2048 (top-down rebinding at faithful capacity, 30ep, micro-batch 8 x grad_accum 4) ==="
uv run python -m dino_trm.train mode=coupled $COMMON model.gnn_cross_attn=true model.decoder_hidden=2048 data.batch_size=8 optim.grad_accum=4 eval.max_batches=60 loss.query_ortho_weight=0.1 log.run_name=coupled_dec2048

echo "=== DONE ==="
