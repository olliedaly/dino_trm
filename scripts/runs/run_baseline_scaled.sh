#!/usr/bin/env bash
set -e
echo "=== BASELINE SCALED (15.6k imgs, 70 ep) ==="
uv run python -m dino_trm.train mode=baseline log.wandb=false data.large_train=true \
    optim.epochs=70 optim.warmup_steps=300 eval.max_batches=10
echo "=== DONE ==="
