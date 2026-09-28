#!/usr/bin/env bash
# COCO multi-object subset — control runs for the recursion claim.
#
# (1) mlp_block + xattn  : non-recursive transformer stack (3 _Blocks w/ slot→patch
#                          cross-attn) in the same architectural slot as TRM. ~11%
#                          MORE trainable params than trm/coupled+xattn. Isolates
#                          "is the gain just from extra params?"
# (2) trm + xattn, T=1   : same module as the headline trm+xattn run, but a single
#                          recursion step. Isolates "is it the recursion specifically,
#                          or just the patch-grounded reasoning module?"
#
# We do NOT separately run coupled @ T=1: at T=1 it is trm @ T=1 plus one extra
# slot-attention pass with the learned z0, so the two are near-duplicates of the same
# data point and add a row to the table that needs a paragraph of caveat to explain.
# trm @ T=1 is the cleanest single recursion control.
#
# Same recipe as run_coco.sh (30 epochs, lr 4e-4, batch 32, COCO 25k subset). T=1
# uses cfg.log.run_name to save under trm_t1_epoch*.pt so the existing T=8
# checkpoints in checkpoints/coco/ are not overwritten.
set -e
COMMON="data.dataset=coco log.wandb=false optim.epochs=30 optim.warmup_steps=300 \
eval.max_batches=15 log.ckpt_dir=checkpoints/coco model.gnn_cross_attn=true"

echo "=== COCO MLP-BLOCK + xattn (3 layers, ~7.1M params, 30ep) ==="
uv run python -m dino_trm.train mode=mlp_block $COMMON

echo "=== COCO TRM + xattn, T=1 (30ep) ==="
uv run python -m dino_trm.train mode=trm $COMMON model.n_steps=1 log.run_name=trm_t1

echo "=== DONE ==="
