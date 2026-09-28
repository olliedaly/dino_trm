#!/usr/bin/env bash
cd "$(dirname "$0")/../.."   # run from the repo root
# Dose-response in recursion depth.
#
# We have T=1 (trm_t1) and T=8 (trm) at 3 training seeds each, showing recursion
# trades large-object accuracy for small-object accuracy (size-rho = -0.22, 9/9
# cross-seed pairings). A two-point contrast cannot show that RECURSION DEPTH is the
# causal variable — only that the two endpoints differ.
#
# PREDICTION: the size slope steepens MONOTONICALLY with T.
#     T=1 (reference)  ->  T=2  ->  T=4  ->  T=8   with rho going 0 -> ... -> -0.22
# A confound would have to scale with recursion depth to fake a monotone curve, which
# is a much harder thing to do by accident than to produce one endpoint difference.
#
# SEEDS: one seed each. Justified because the training-seed null on the size profile
# is now measured and tiny (size-rho in [-0.045,+0.032] across 9 same-arm
# comparisons, vs the effect's -0.22), so a single seed is enough to place a point on
# the curve. Add seeds only if the curve comes out non-monotone.
#
# Recipe verified against checkpoints/coco/trm_epoch29.pt's saved cfg: mode=trm,
# gnn_cross_attn=true, decoder_hidden=1024, 7 slots, bs32, 30ep, warmup 300, seed 0.
# Identical to run_coco_controls.sh except n_steps.
#
# COST: T=2 ~1.5h, T=4 ~2.5h (recursion cost is ~linear in T: 1.9 min/ep at T=1,
# 9.4 min/ep at T=8).
set -e

COMMON="data.dataset=coco log.wandb=false optim.epochs=30 optim.warmup_steps=300 \
eval.max_batches=15 log.ckpt_dir=checkpoints/coco model.gnn_cross_attn=true"

for T in 2 4; do
  echo "=== COCO TRM + xattn, T=$T (seed 0, 30ep) ==="
  uv run python -m dino_trm.train mode=trm $COMMON \
      model.n_steps=$T log.run_name=trm_t$T
done

echo "=== T-SWEEP DONE ==="
