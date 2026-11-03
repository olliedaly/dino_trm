#!/usr/bin/env bash
cd "$(dirname "$0")/../.."   # run from the repo root
# Slot-attention recursion with 1-step gradients (TRM-style deep supervision).
#
# WHY: sa_rec (run_coco_sa_rec.sh) backpropagated through all 8 slot-attention
# iterations, and after epoch 0 slot attention stopped converging. At the last
# iteration the slots still moved 0.3-0.4 of their norm (baseline_pm, run for 8
# iterations at test time: 0.03). Only 26-29% of patches kept their slot from one
# iteration to the next, and one slot took 50-75% of the image. In-training FG-ARI fell
# from 31 (epoch 0) to 10-24, while baseline_pm was at 37-41.
#
# RUN (sa_rec_1step): identical to sa_rec except model.sa_detach=true, which detaches
# the slots between iterations, so each iteration's decoded loss trains only that one
# update. TRM detaches between its deep-supervision steps the same way. Same seed,
# data and parameter count as baseline_pm. NOTE: the trm runs used full backprop.
set -e
COMMON="data.dataset=coco log.wandb=false optim.epochs=30 optim.warmup_steps=300 \
eval.max_batches=15 log.ckpt_dir=checkpoints/coco"

echo "=== COCO SLOT-ATTENTION RECURSION, 8 iterations, 1-step gradients (seed 0, 30ep) ==="
PYTHONUNBUFFERED=1 uv run python -m dino_trm.train mode=sa_rec $COMMON \
    model.decoder_hidden=1469 model.slot_iters=8 model.sa_detach=true \
    log.run_name=sa_rec_1step

echo "=== DONE ==="
