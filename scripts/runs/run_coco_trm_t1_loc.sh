#!/usr/bin/env bash
cd "$(dirname "$0")/../.."   # run from the repo root
# Sufficiency test for what recursion does: T=1 (trm_t1's recipe) trained with a locality
# prior in the decoder (model.loc_bias). Each slot's alpha logit is lowered by
# loc_bias x the squared distance (image widths) from the slot's mask centroid.
#
# T=8 (trm) groups by position where T=1 groups by appearance across distance. Imposing
# that locality on a trained T=1 after training (loc_bias 10 on its logits) matched T=8's
# far same-class sharing and reproduced its background and class-metric costs, but not
# its FG-ARI gain or all of its reconstruction loss. This run asks whether locality
# present during training reproduces T=8 fully.
#
# Usage: run_coco_trm_t1_loc.sh <loc_bias> [seed]   (run name trm_t1_lp<loc_bias>)
# COST: about 1 h (T=1, 1.9 min/epoch).
set -e
B=$1
S=${2:-0}
RUN=trm_t1_lp$B$([ "$S" = 0 ] || echo "_seed$S")
mkdir -p results/coco/$RUN
exec > results/coco/$RUN/train.log 2>&1
echo "=== COCO $RUN (T=1, loc_bias $B, seed $S, 30ep) ==="
export PYTHONUNBUFFERED=1
exec uv run python -m dino_trm.train mode=trm data.dataset=coco log.wandb=false \
    optim.epochs=30 optim.warmup_steps=300 eval.max_batches=15 \
    log.ckpt_dir=checkpoints/coco model.gnn_cross_attn=true model.n_steps=1 \
    model.loc_bias=$B seed=$S log.run_name=$RUN
