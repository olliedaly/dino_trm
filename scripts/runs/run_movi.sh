#!/usr/bin/env bash
cd "$(dirname "$0")/../.."   # run from the repo root
# MOVi-C: the third domain, after VOC (null) and KITTI (rejected on size range).
#
# QUESTION: does recursion improve SMALL-object segmentation relative to no recursion,
# and does the benefit decline with object size — the COCO result (Q1 +4.27 -> Q5 -1.72,
# rho=-0.309) reproduced in a domain that is not COCO?
#
# WHY MOVi-C CAN ANSWER IT (measured on the 250-image val set):
#   per-object size bands   25.3 / 19.2 / 22.8 / 19.3 / 13.3 %  over a 39x range
#   KITTI, for contrast     59.7 / 16.9 / 15.8 /  4.8 /  2.7 %  <- no large arm, rejected
# Ground truth is complete (renderer instance ids), so the annotation-completeness
# confound that VOC could not rule out does not exist here.
#
# ARCHITECTURE IS BYTE-IDENTICAL TO THE COCO RUNS — only `data.dataset` changes. The
# COMMON string below is the COCO one with dataset=movi and ckpt_dir swapped, so any
# difference in outcome is the data, not the model.
#
# THE SCHEDULE ALSO NEEDS NO ADAPTATION, which is luck worth noting: DINOSAUR's frame
# protocol (9 random frames per clip) over our 3,005-clip subset gives 27,045 training
# images against COCO's 25,000 — within 8%. So epochs=30 / warmup=300 / bs=32 carry
# over unchanged rather than being retuned, removing a confound.
#
# COST (scaled from the measured COCO rates, 845 steps/epoch here vs 781 there):
#   trm_t1  (T=1)  ~2.1 min/epoch -> ~1.0 h
#   trm     (T=8)  ~10.2 min/epoch -> ~5.1 h
#   baseline_pm    ~2.1 min/epoch -> ~1.0 h
#   total ~7.1 h
#
# ORDER: the decisive pair (trm_t1, trm) runs first, so the recursion contrast is
# complete after ~6.1 h even if the run is interrupted before the baseline lands.
set -e

COMMON="data.dataset=movi log.wandb=false optim.epochs=30 optim.warmup_steps=300 \
eval.max_batches=15 log.ckpt_dir=checkpoints/movi model.gnn_cross_attn=true"

echo "=== TRM T=1 (recursion off, block on) ==="
uv run python -m dino_trm.train mode=trm $COMMON seed=0 \
    model.n_steps=1 log.run_name=trm_t1

echo "=== TRM T=8 (recursion on) ==="
uv run python -m dino_trm.train mode=trm $COMMON seed=0 \
    log.run_name=trm

# Param-matched baseline: decoder_hidden=1469 equalises trainable params against the
# TRM arm, so "TRM helps" cannot be read as "TRM has more capacity". Verified against
# checkpoints/coco/baseline_pm_epoch29.pt's saved cfg.
echo "=== BASELINE (param-matched, no block, no recursion) ==="
uv run python -m dino_trm.train mode=baseline $COMMON seed=0 \
    model.decoder_hidden=1469 log.run_name=baseline_pm

echo "=== DONE ==="
