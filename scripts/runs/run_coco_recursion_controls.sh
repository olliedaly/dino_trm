#!/usr/bin/env bash
cd "$(dirname "$0")/../.."   # run from the repo root
# Controls for what recursion does (T=8 vs T=1). One run per call:
#
#   trm_1step  T=8 with TRM's 1-step gradients: y and z are detached between steps,
#              so each step's loss trains only that step (x, slot attention's slots,
#              keeps its gradient). TRM detaches between its deep-supervision steps;
#              our trm backpropagates through all 8 steps. Tests whether T=8's
#              locality and higher reconstruction error come from that training method.
#   trm_final  T=8 with the loss on the final step only (no deep supervision), to
#              separate recursion from the losses on the 7 intermediate steps.
#
# Everything else is trm's recipe (run_coco_crowding_seeds.sh): gnn_cross_attn=true,
# decoder_hidden=1024, 7 slots, bs32, 30ep, warmup 300.
# Usage: run_coco_recursion_controls.sh <trm_1step|trm_final> <seed>
# COST: about 4.7 h per run (9.4 min/epoch at T=8).
set -e
NAME=$1
S=$2
case $NAME in
  trm_1step) EXTRA="model.trm_detach=true" ;;
  trm_final) EXTRA="loss.intermediate_weight=0.0" ;;
  *) echo "unknown run: $NAME"; exit 1 ;;
esac
RUN=$NAME$([ "$S" = 0 ] || echo "_seed$S")
mkdir -p results/coco/$RUN
exec > results/coco/$RUN/train.log 2>&1
echo "=== COCO $RUN (T=8, seed $S, 30ep) ==="
export PYTHONUNBUFFERED=1
exec uv run python -m dino_trm.train mode=trm data.dataset=coco log.wandb=false \
    optim.epochs=30 optim.warmup_steps=300 eval.max_batches=15 \
    log.ckpt_dir=checkpoints/coco model.gnn_cross_attn=true seed=$S $EXTRA \
    log.run_name=$RUN
