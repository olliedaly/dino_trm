#!/usr/bin/env bash
cd "$(dirname "$0")/../.."   # run from the repo root
# Closes the last single-seed gap in the COCO analysis.
#
# trm_t1 / trm / mlp_block now have 3 training seeds each, but baseline_pm still has
# only seed 0. Every control CONTRAST is anchored on it:
#     baseline_pm -> mlp_block   (capacity, non-recursive)
#     baseline_pm -> trm_t1      (the TinyGNN block at T=1)
# so their LEVELS inherit a single-seed error bar. The size-rho sign replicated across
# all three mlp_block seeds, so the dissociation conclusion is already safe — but the
# three-way decomposition (block / recursion / coupling) reported at the time is
# quoted as point estimates off this one run, and mlp_block's seed 0 turned out to be
# ~0.9 FG-ARI unlucky (44.65 vs 45.57/45.56), which is exactly the failure mode this
# fixes.
#
# Recipe verified against checkpoints/coco/baseline_pm_epoch29.pt's saved cfg:
# mode=baseline, decoder_hidden=1469 (6.3967M params, param-matched to the recursive
# models' 6.3960M), 7 slots, bs32, 30ep, warmup 300. Note mode=baseline ignores
# gnn_cross_attn, so it is deliberately absent here.
#
# COST: ~1.7h per run, ~3.4h total.
set -e

COMMON="data.dataset=coco log.wandb=false optim.epochs=30 optim.warmup_steps=300 \
eval.max_batches=15 log.ckpt_dir=checkpoints/coco"

for S in 1 2; do
  echo "=== COCO BASELINE param-matched (decoder_hidden=1469), seed=$S ==="
  uv run python -m dino_trm.train mode=baseline $COMMON seed=$S \
      model.decoder_hidden=1469 log.run_name=baseline_pm_seed$S
done

echo "=== BASELINE_PM SEEDS DONE ==="
