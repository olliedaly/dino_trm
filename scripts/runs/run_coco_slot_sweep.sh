#!/usr/bin/env bash
cd "$(dirname "$0")/../.."   # run from the repo root
# COCO slot-count null-hypothesis control for the recursion claim.
#
# The paper's headline is that the recursive module (mode=trm/coupled) beats the
# DINOSAUR-style baseline (mode=baseline) on COCO FG-ARI (42.5 -> 46.7/47.2). The
# null hypothesis is that the baseline only loses because it has too few slots
# (num_slots=7). Here we retrain the baseline at num_slots = 9 / 11 / 15 on the same
# COCO recipe and check whether FG-ARI recovers toward the recursive numbers.
#
# NOTE on slot count: DINOSAUR's official COCO config already uses n_slots=7, so the
# headline baseline matches the published slot count. Sources:
#   configs/experiment/projects/bridging/dinosaur/coco_feat_rec_dino_base16.yaml
#   configs/experiment/projects/bridging/dinosaur/coco_feat_rec_dino_small16_auto.yaml
# in amazon-science/object-centric-learning-framework, both n_slots: 7 (VOC uses 6).
#
# VRAM check (RTX 5060 Ti 16GB, ~14.5GB budget): batch 32 fits at every slot count
# (max reserved 4.66GB @ 15 slots), so no data.batch_size / optim.grad_accum split is
# needed — the effective batch stays 32 and the LR schedule is untouched.
#
# Same recipe as run_coco.sh (30 epochs, lr 4e-4, batch 32, COCO 25k subset). Each run
# uses cfg.log.run_name to save under baseline_s9/s11/s15_epoch*.pt so the existing
# baseline checkpoints in checkpoints/coco/ are not overwritten.
set -e
COMMON="data.dataset=coco log.wandb=false optim.epochs=30 optim.warmup_steps=300 \
eval.max_batches=15 log.ckpt_dir=checkpoints/coco"

echo "=== COCO BASELINE, num_slots=9 (30ep) ==="
uv run python -m dino_trm.train mode=baseline $COMMON model.num_slots=9 log.run_name=baseline_s9

echo "=== COCO BASELINE, num_slots=11 (30ep) ==="
uv run python -m dino_trm.train mode=baseline $COMMON model.num_slots=11 log.run_name=baseline_s11

echo "=== COCO BASELINE, num_slots=15 (30ep) ==="
uv run python -m dino_trm.train mode=baseline $COMMON model.num_slots=15 log.run_name=baseline_s15

# Training-seed noise yardstick: the existing +/- numbers at the time are EVAL-seed
# only (3 forward passes of one trained model), so they say nothing about run-to-run
# training variance. Without that, a partial FG-ARI recovery at 9/11/15 slots cannot be
# distinguished from noise. This is the canonical 7-slot baseline retrained at seed=1;
# |seed1 - seed0| is the scale a slot-count effect has to beat to mean anything.
echo "=== COCO BASELINE, num_slots=7, seed=1 (train-seed noise control, 30ep) ==="
uv run python -m dino_trm.train mode=baseline $COMMON model.num_slots=7 seed=1 log.run_name=baseline_s7_seed1

echo "=== DONE ==="
