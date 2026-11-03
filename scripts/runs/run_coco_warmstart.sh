#!/usr/bin/env bash
cd "$(dirname "$0")/../.."   # run from the repo root
# Warm start: continue a trained run's weights with a different number of recursion steps.
#
# T=1 (trm_t1) groups by appearance and T=8 (trm) by position. Does training with 8 steps
# actively remove appearance grouping from a model that has it (trm_t1 -> 8 steps), or does
# T=8 just never find it? And does training with 1 step bring it back (trm -> 1 step)?
# trm_t1 -> 1 step is the control for the disruption of restarting the optimiser.
# Starting from an early epoch tests when the outcome is fixed (before or after T=1 learns
# appearance grouping and T=8's slot attention collapses, epochs 2-7).
# Fresh optimiser and schedule (warmup 300, cosine over the epochs); everything else is
# trm's recipe.
#
# With [freeze] (e.g. slot_attention,input_proj; - for none) those submodules keep the loaded
# weights: from trm epoch 2 with slot attention frozen, can T=8 still lose its grouping?
# With [sched] = orig the run continues the original 30-epoch schedule from from_epoch + 1
# (after a fresh warmup) instead of a fresh 6-epoch cosine, so the learning rate is as high
# as it was at that point of training from scratch.
#
# Usage: run_coco_warmstart.sh <from_run> <n_steps> [epochs] [seed] [from_epoch] [freeze] [sched]
#   (run name ws_<from_run>_T<n_steps>, or ws_<from_run>e<from_epoch>_T<n_steps> when
#    from_epoch is not 29, plus _frozen with [freeze] and _lr30 with [sched] = orig; seed 0
#    starts from <from_run>, seed s from <from_run>_seed<s>)
# COST: about 2 min/epoch at 1 step, 10 min/epoch at 8 steps.
set -e
FROM=$1
N=$2
E=${3:-6}
S=${4:-0}
FE=${5:-29}
FZ=${6:-}
if [ "$FZ" = - ]; then FZ=; fi
SCHED=${7:-}
SRC=$FROM$([ "$S" = 0 ] || echo "_seed$S")
RUN=ws_${FROM}$([ "$FE" = 29 ] || echo "e$FE")_T$N$([ -z "$FZ" ] || echo "_frozen")$([ "$SCHED" != orig ] || echo "_lr30")$([ "$S" = 0 ] || echo "_seed$S")
SCHED_ARGS=
if [ "$SCHED" = orig ]; then SCHED_ARGS="optim.sched_epochs=30 optim.sched_offset_epochs=$((FE + 1))"; fi
mkdir -p results/coco/$RUN
exec > results/coco/$RUN/train.log 2>&1
echo "=== COCO $RUN (from $SRC epoch $FE, $N steps, seed $S, ${E}ep${FZ:+, frozen $FZ}${SCHED_ARGS:+, $SCHED_ARGS}) ==="
export PYTHONUNBUFFERED=1
exec uv run python -m dino_trm.train mode=trm data.dataset=coco log.wandb=false \
    optim.epochs=$E optim.warmup_steps=300 eval.max_batches=15 \
    log.ckpt_dir=checkpoints/coco model.gnn_cross_attn=true model.n_steps=$N \
    init_from=checkpoints/coco/${SRC}_epoch$FE.pt seed=$S log.run_name=$RUN ${FZ:+freeze=[$FZ]} $SCHED_ARGS
