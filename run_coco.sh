#!/usr/bin/env bash
set -e
# COCO multi-object subset (25k train / 2k val) — occlusion-rich regime for Step 1
# cross-attn variants. Checkpoints/results are namespaced (checkpoints/coco,
# results/coco/<mode>) so the validated VOC artifacts are left untouched.
COMMON="data.dataset=coco log.wandb=false optim.epochs=30 optim.warmup_steps=300 \
eval.max_batches=15 log.ckpt_dir=checkpoints/coco"

echo "=== COCO BASELINE (30ep) ==="
uv run python -m dino_trm.train mode=baseline $COMMON

echo "=== COCO TRM + cross-attn (30ep) ==="
uv run python -m dino_trm.train mode=trm $COMMON model.gnn_cross_attn=true

echo "=== COCO COUPLED + cross-attn (30ep) ==="
uv run python -m dino_trm.train mode=coupled $COMMON model.gnn_cross_attn=true loss.query_ortho_weight=0.1

echo "=== DONE ==="
