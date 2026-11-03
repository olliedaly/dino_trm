#!/usr/bin/env bash
cd "$(dirname "$0")/../.."   # run from the repo root
# Slot-attention recursion: recurse with slot attention itself instead of a module.
#
# WHY: in trm/coupled the module's cross-attention takes over the grouping and slot
# attention's own map collapses (FG-ARI 17.6 at T=8; ARI 0.20 with the output). Yet
# baseline_pm's slot-attention map is the best readout we have (48.0 FG-ARI) and keeps
# improving when slot attention simply runs longer at test time (49.5 at 16
# iterations), while the decoder masks get worse (43.9 -> 41.2) because the decoder only
# ever saw 3-iteration slots.
#
# RUN (sa_rec): mode=sa_rec runs slot attention for 8 iterations and decodes the slots
# after every iteration, with trm's deep-supervision weights (0.25 per intermediate
# iteration, 1.0 final) and full backprop through all iterations. There is no module,
# so slot attention must do the grouping. Everything else is baseline_pm's recipe
# (decoder_hidden=1469, 7 slots, bs32, 30ep, warmup 300, seed 0, same COCO subset), so
# the trainable parameter count is identical (6.3967M) and the only differences from
# baseline_pm are 8 iterations instead of 3 and the per-iteration supervision.
# Score both readouts afterwards: slot attention's map and the decoder masks.
#
# COST: ~1.56 s per step at batch 32 (8 decodes at decoder width 1469), ~9.1 GB peak.
set -e
COMMON="data.dataset=coco log.wandb=false optim.epochs=30 optim.warmup_steps=300 \
eval.max_batches=15 log.ckpt_dir=checkpoints/coco"

echo "=== COCO SLOT-ATTENTION RECURSION, 8 iterations decoded (seed 0, 30ep) ==="
uv run python -m dino_trm.train mode=sa_rec $COMMON model.decoder_hidden=1469 \
    model.slot_iters=8 log.run_name=sa_rec

echo "=== DONE ==="
