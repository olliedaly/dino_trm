#!/usr/bin/env bash
set -e
COMMON="log.wandb=false"   # image_size/epochs/warmup now from config (336/80/300)
echo "=== BASELINE ==="; uv run python -m dino_trm.train mode=baseline $COMMON eval.max_batches=10
echo "=== TRM ==="; uv run python -m dino_trm.train mode=trm $COMMON eval.max_batches=10
echo "=== COUPLED ==="; uv run python -m dino_trm.train mode=coupled $COMMON eval.max_batches=10 loss.query_ortho_weight=0.1
echo "=== ALL DONE ==="
