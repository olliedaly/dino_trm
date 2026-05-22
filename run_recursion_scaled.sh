#!/usr/bin/env bash
set -e
# gnn_combine=concat is the config default (P3); epochs cut to 40 (baseline plateaus ~ep20)
COMMON="log.wandb=false data.large_train=true optim.epochs=40 optim.warmup_steps=200 eval.max_batches=10"
echo "=== TRM SCALED (concat, 40ep) ==="; uv run python -m dino_trm.train mode=trm $COMMON
echo "=== COUPLED SCALED (concat, 40ep) ==="; uv run python -m dino_trm.train mode=coupled $COMMON loss.query_ortho_weight=0.1
echo "=== DONE ==="
