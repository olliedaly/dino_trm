#!/usr/bin/env bash
cd "$(dirname "$0")/../.."   # run from the repo root
# Training-seed replication for the CROWDING finding.
#
# CLAIM UNDER TEST: TRM recursion depth (T=1 -> T=8) helps specifically in crowded
# scenes and hurts in sparse ones — measured as a Spearman trend of the per-image
# FG-ARI delta against GT instance count (rho=+0.109, p<1e-4 at one training seed).
#
# WHY THIS IS NEEDED: every config in results/coco/summary_protocol.json has ONE
# training seed. The +-std reported there is over 3 EVAL seeds (random slot init at
# inference), which does not bound training-seed variance at all. The only same-config
# seed pair we have (baseline vs baseline_s7_seed1) differs by 0.24 FG-ARI, but that
# is a df=1 variance estimate and coupled-mode's total slot collapse without
# query_ortho shows this model class is init-sensitive — so that estimate cannot be
# extrapolated.
#
# DESIGN: seeds 1 and 2 for the two arms of the contrast (trm_t1, trm) plus the
# non-recursive control (mlp_block), giving n=3 training seeds per arm and 9
# cross-seed pairings for the trm_t1 -> trm delta. If the crowding trend is real it
# holds across all 9; if it is a property of one lucky init it will not.
#
# Recipes are byte-identical to run_coco.sh / run_coco_controls.sh except for
# `seed` and `log.run_name`. cfg.seed drives model init, data shuffling, augmentation
# and training-time slot init, so it varies the whole training trajectory.
#
# COST (measured from checkpoint mtimes of the seed-0 runs on the RTX 5060 Ti):
#   trm     (T=8)  ~9.4 min/epoch  -> ~4.7 h/run
#   trm_t1  (T=1)  ~1.9 min/epoch  -> ~0.9 h/run
#   mlp_block      ~1.9 min/epoch  -> ~0.9 h/run
#   total 2*(4.7 + 0.9 + 0.9) ~= 13 h
#
# ORDER: the decisive pair (trm_t1 + trm) is completed for seed 1 first, so the main
# contrast has a second full seed after ~5.6 h even if the run is interrupted later.
set -e

COMMON="data.dataset=coco log.wandb=false optim.epochs=30 optim.warmup_steps=300 \
eval.max_batches=15 log.ckpt_dir=checkpoints/coco model.gnn_cross_attn=true"

for S in 1 2; do
  echo "=== TRM T=1, seed=$S ==="
  uv run python -m dino_trm.train mode=trm $COMMON seed=$S \
      model.n_steps=1 log.run_name=trm_t1_seed$S

  echo "=== TRM T=8, seed=$S ==="
  uv run python -m dino_trm.train mode=trm $COMMON seed=$S \
      log.run_name=trm_seed$S
done

for S in 1 2; do
  echo "=== MLP-BLOCK (non-recursive control), seed=$S ==="
  uv run python -m dino_trm.train mode=mlp_block $COMMON seed=$S \
      log.run_name=mlp_block_seed$S
done

echo "=== DONE ==="
