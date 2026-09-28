#!/usr/bin/env bash
cd "$(dirname "$0")/../.."   # run from the repo root
# VOC with the COCO configuration, byte-for-byte. Supersedes ALL earlier VOC work
# for the object-size analysis.
#
# WHY: the pre-existing VOC runs (checkpoints/{baseline,trm,coupled,trm_t1}_epoch*.pt)
# differ from the COCO runs in several ways at once — no gnn_cross_attn (the flag
# postdates them), 40 epochs vs 30, warmup 200 vs 300. A cross-dataset claim cannot be
# made across a recipe that changes in three places, so those runs are retired from
# this analysis. They are NOT deleted: baseline/trm/coupled are the published-DINOSAUR
# replication evidence for the project and stay where they are.
#
# WHAT IS NOW IDENTICAL TO COCO: mode, gnn_cross_attn=true, n_steps, decoder_hidden
# 1024, decoder_layers 4, num_slots 7, slot_dim 256, slot_iters 3, image_size 336,
# batch_size 32, grad_accum 1, lr 4e-4, weight_decay 0, grad_clip 1.0, epochs 30,
# warmup_steps 300, seed 0, deep-supervision weights. Verified against
# checkpoints/coco/{trm,trm_t1,baseline_pm}_epoch29.pt saved cfgs.
#
# THE ONE THING THAT CANNOT MATCH — and must be stated in any writeup:
#   VOC large_train = 15,675 images -> 489 steps/epoch -> 14,670 optimizer steps
#   COCO subset     = 25,000 images -> 781 steps/epoch -> 23,430 optimizer steps
# Same epoch count therefore means ~37% fewer optimizer steps on VOC. Holding epochs
# fixed is the "same recipe" reading; holding steps fixed would need epochs=48 and
# would break the recipe match instead. Both arms (T=1 and T=8) get identical
# treatment either way, so the WITHIN-VOC contrast — which is what the object-size
# claim rests on — is unaffected. Only cross-dataset LEVEL comparisons are exposed to
# it, and those should be avoided in favour of comparing the DELTAS.
#
# Uses data.large_train=true: the standard VOC segmentation train split is ~1.5k
# images, far too small for this recipe. The 15.6k leakage-free image-only set is the
# validated choice from the original replication (scripts/build_train_index.py).
#
# ORDER: the two arms of the size contrast first, so an interruption still leaves the
# decisive comparison complete. baseline_pm last — it only feeds the block term.
#
# COST (scaled from COCO min/epoch by 15.6k/25k): T=1 ~35min, T=8 ~3h, baseline_pm ~1h.
set -e

COMMON="log.wandb=false data.large_train=true optim.epochs=30 optim.warmup_steps=300 \
eval.max_batches=10 model.gnn_cross_attn=true"

echo "=== [1/3] VOC TRM T=1, COCO-matched (30ep, warmup 300, xattn) ==="
uv run python -m dino_trm.train mode=trm $COMMON model.n_steps=1 log.run_name=trm_t1_m

echo "=== [2/3] VOC TRM T=8, COCO-matched ==="
uv run python -m dino_trm.train mode=trm $COMMON log.run_name=trm_m

echo "=== [3/3] VOC baseline param-matched, COCO-matched (decoder_hidden=1469) ==="
uv run python -m dino_trm.train mode=baseline log.wandb=false data.large_train=true \
    optim.epochs=30 optim.warmup_steps=300 eval.max_batches=10 \
    model.decoder_hidden=1469 log.run_name=baseline_pm_m

echo "=== DONE ==="
