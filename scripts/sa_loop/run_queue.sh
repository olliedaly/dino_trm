#!/usr/bin/env bash
# Train the slot-attention-loop runs one after another (docs/sa_loop_plan.md).
#
#   scripts/sa_loop/run_queue.sh [--dry-run] stage1 | stage2a | stage2b | <run name> ...
#
# Run names decide the recipe:
#   sa8_last[_seedS]         arm C: 8 passes, one loss, gradient from the last update only
#   sa_rec_1step[_seedS]     arm D: 8 passes, a loss after each, gradient cut between passes
#   sa3_r448[_seedS]         arm A at 448 px (3 passes)
#   sa8_last_r448[_seedS]    arm C at 448 px
#   sa8_each_r448[_seedS]    arm D at 448 px
# Everything else is baseline_pm's recipe: decoder_hidden=1469 (6.40M parameters at 336 px),
# 7 slots, 30 epochs, warmup 300, the 25k / 2k COCO subset.
#
# Safe to rerun: a finished run (its epoch-29 checkpoint exists) is skipped, and an
# unfinished one is trained again from scratch. Only one queue can run at a time, and it
# will not start while another training process is alive. It stops at the first failure.
set -u
cd "$(dirname "$0")/../.."   # the repo root
source scripts/sa_loop/env.sh

DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
case "${1:-}" in
  stage1)  RUNS="sa8_last sa8_last_seed1 sa8_last_seed2 sa_rec_1step sa_rec_1step_seed1 sa_rec_1step_seed2" ;;
  stage2a) RUNS="sa3_r448 sa8_last_r448 sa8_each_r448" ;;
  stage2b) RUNS="sa3_r448_seed1 sa3_r448_seed2 sa8_last_r448_seed1 sa8_last_r448_seed2 sa8_each_r448_seed1 sa8_each_r448_seed2" ;;
  "")      echo "usage: $0 [--dry-run] stage1 | stage2a | stage2b | <run name> ..."; exit 2 ;;
  *)       RUNS="$*" ;;
esac

COMMON="data.dataset=coco log.wandb=false optim.epochs=30 optim.warmup_steps=300 log.ckpt_dir=checkpoints/coco model.decoder_hidden=1469"
R336="eval.max_batches=15"
R448="model.image_size=448 data.image_size=448 data.batch_size=16 optim.grad_accum=2 eval.max_batches=30"
LAST="mode=baseline model.slot_iters=8 model.sa_detach=true"
EACH="mode=sa_rec model.slot_iters=8 model.sa_detach=true"

recipe() {  # run name -> training arguments (empty if the name is not one of ours)
  local base=${1%_seed[0-9]}
  case "$base" in
    sa8_last)       echo "$LAST $COMMON $R336" ;;
    sa_rec_1step)   echo "$EACH $COMMON $R336" ;;
    sa3_r448)       echo "mode=baseline $COMMON $R448" ;;
    sa8_last_r448)  echo "$LAST $COMMON $R448" ;;
    sa8_each_r448)  echo "$EACH $COMMON $R448" ;;
  esac
}
seed_of() { case "$1" in *_seed[0-9]) echo "${1##*_seed}" ;; *) echo 0 ;; esac; }

mkdir -p results/sa_loop
LOG=results/sa_loop/queue.log
say() { if [ $DRY = 1 ]; then echo "$*"; else echo "$(date '+%F %T') $*" | tee -a "$LOG"; fi; }

if [ $DRY = 0 ]; then
  exec 9> results/sa_loop/queue.lock
  if ! flock -n 9; then echo "another queue is already running; not starting"; exit 1; fi
  # [d] keeps the pattern from matching a shell whose own command line quotes it
  if pgrep -f "[d]ino_trm[.]train" > /dev/null; then echo "a training process is already running; not starting"; exit 1; fi
fi

for RUN in $RUNS; do
  ARGS=$(recipe "$RUN")
  if [ -z "$ARGS" ]; then say "UNKNOWN run name $RUN, stopping"; exit 2; fi
  S=$(seed_of "$RUN")
  CMD="uv run python -m dino_trm.train $ARGS seed=$S log.run_name=$RUN"
  if [ -f "checkpoints/coco/${RUN}_epoch29.pt" ]; then say "SKIP $RUN (finished)"; continue; fi
  if [ $DRY = 1 ]; then echo "WOULD RUN [$RUN] $CMD"; continue; fi
  rm -f checkpoints/coco/"${RUN}"_epoch*.pt        # leftovers of an unfinished attempt
  mkdir -p "results/coco/$RUN"
  say "START $RUN: $CMD"
  $CMD > "results/coco/$RUN/train.log" 2>&1
  RC=$?
  if [ $RC -ne 0 ] || [ ! -f "checkpoints/coco/${RUN}_epoch29.pt" ]; then
    say "FAILED $RUN (exit $RC); see results/coco/$RUN/train.log. Queue stopped."
    exit 1
  fi
  say "DONE $RUN"
done
say "QUEUE FINISHED: $RUNS"
