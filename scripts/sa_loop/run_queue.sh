#!/usr/bin/env bash
# Train the slot-attention-loop runs one after another.
#
#   scripts/sa_loop/run_queue.sh [--dry-run] stage1 | stage3 | stage4 | stage2a | stage2b | <run name> ...
#
# Run names decide the recipe. On the COCO subset, 336 px:
#   sa8_last[_seedS]         arm C: 8 passes, one loss, gradient from the last update only
#   sa_rec_1step[_seedS]     arm D: 8 passes, a loss after each, gradient cut between passes
#   sa8_full[_seedS]         arm E: 8 passes, one loss, gradient through all passes (ordinary training)
#   sa8_rand[_seedS]         arm F: arm D's loss, with each image decoded at one sampled pass
# On PASCAL VOC, 336 px (checkpoints/ and results/<run>/, as the earlier VOC runs):
#   baseline_pm_m[_seedS]    arm A: 3 passes
#   sa8_full_voc[_seedS]     arm E
#   sa8_rand_voc[_seedS]     arm F
# On the COCO subset, 448 px:
#   sa3_r448[_seedS]  sa8_last_r448[_seedS]  sa8_each_r448[_seedS]      arms A, C, D
# Everything else is baseline_pm's recipe: decoder_hidden=1469 (6.40M parameters at 336 px),
# 7 slots, 30 epochs, warmup 300.
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
  stage3)  RUNS="sa8_full sa8_full_seed1 sa8_full_seed2 sa8_rand sa8_rand_seed1 sa8_rand_seed2" ;;
  stage4)  RUNS="baseline_pm_m baseline_pm_m_seed1 baseline_pm_m_seed2 sa8_full_voc sa8_full_voc_seed1 sa8_full_voc_seed2 sa8_rand_voc sa8_rand_voc_seed1 sa8_rand_voc_seed2" ;;
  stage2a) RUNS="sa3_r448 sa8_last_r448 sa8_each_r448" ;;
  stage2b) RUNS="sa3_r448_seed1 sa3_r448_seed2 sa8_last_r448_seed1 sa8_last_r448_seed2 sa8_each_r448_seed1 sa8_each_r448_seed2" ;;
  "")      echo "usage: $0 [--dry-run] stage1 | stage3 | stage4 | stage2a | stage2b | <run name> ..."; exit 2 ;;
  *)       RUNS="$*" ;;
esac

SHARED="log.wandb=false optim.epochs=30 optim.warmup_steps=300 model.decoder_hidden=1469"
COCO="data.dataset=coco log.ckpt_dir=checkpoints/coco $SHARED"
R336="eval.max_batches=15"
R448="model.image_size=448 data.image_size=448 data.batch_size=16 optim.grad_accum=2 eval.max_batches=30"
VOC="data.large_train=true eval.max_batches=10 $SHARED"   # the VOC dataset and checkpoints/ are the defaults
LAST="mode=baseline model.slot_iters=8 model.sa_detach=true"
EACH="mode=sa_rec model.slot_iters=8 model.sa_detach=true"
FULL="mode=baseline model.slot_iters=8"
RAND="mode=sa_rec model.slot_iters=8 model.sa_detach=true model.sa_sample_pass=true"

recipe() {  # run name -> training arguments (empty if the name is not one of ours)
  local base=${1%_seed[0-9]}
  case "$base" in
    sa8_last)       echo "$LAST $COCO $R336" ;;
    sa_rec_1step)   echo "$EACH $COCO $R336" ;;
    sa8_full)       echo "$FULL $COCO $R336" ;;
    sa8_rand)       echo "$RAND $COCO $R336" ;;
    baseline_pm_m)  echo "mode=baseline $VOC" ;;
    sa8_full_voc)   echo "$FULL $VOC" ;;
    sa8_rand_voc)   echo "$RAND $VOC" ;;
    sa3_r448)       echo "mode=baseline $COCO $R448" ;;
    sa8_last_r448)  echo "$LAST $COCO $R448" ;;
    sa8_each_r448)  echo "$EACH $COCO $R448" ;;
  esac
}
is_voc() { case "${1%_seed[0-9]}" in baseline_pm_m|*_voc) return 0 ;; *) return 1 ;; esac; }
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
  if is_voc "$RUN"; then CK=checkpoints; RES=results/$RUN; else CK=checkpoints/coco; RES=results/coco/$RUN; fi
  S=$(seed_of "$RUN")
  CMD="uv run python -m dino_trm.train $ARGS seed=$S log.run_name=$RUN"
  if [ -f "$CK/${RUN}_epoch29.pt" ]; then say "SKIP $RUN (finished)"; continue; fi
  if [ $DRY = 1 ]; then echo "WOULD RUN [$RUN] $CMD"; continue; fi
  rm -f "$CK/${RUN}"_epoch*.pt        # leftovers of an unfinished attempt
  mkdir -p "$RES"
  say "START $RUN: $CMD"
  $CMD > "$RES/train.log" 2>&1
  RC=$?
  if [ $RC -ne 0 ] || [ ! -f "$CK/${RUN}_epoch29.pt" ]; then
    say "FAILED $RUN (exit $RC); see $RES/train.log. Queue stopped."
    exit 1
  fi
  say "DONE $RUN"
done
say "QUEUE FINISHED: $RUNS"
