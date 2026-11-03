#!/usr/bin/env bash
# Wait for a control run to finish, then extract and score it (eval seed 0).
# Usage: post_control.sh <run>
S="$(cd "$(dirname "$0")" && pwd)"
RUN=$1
cd /run/media/ollie/fedora/home/ollie/Projects/dino_trm
export HF_HOME=/run/media/ollie/fedora/home/ollie/.cache/huggingface HF_HUB_OFFLINE=1
until [ -f checkpoints/coco/${RUN}_epoch29.pt ] && grep -q "\[epoch 29\]" results/coco/$RUN/train.log; do
  grep -qE "Traceback|Error|Killed|out of memory" results/coco/$RUN/train.log 2>/dev/null && { echo "$RUN FAILED"; tail -5 results/coco/$RUN/train.log; exit 1; }
  sleep 60
done
sleep 30
uv run python $S/gpu_batch.py $S/c $RUN 2>&1 | grep -vE "Warning|super\(\)|annotations|^Done|index"
uv run python $S/mech_ctrl.py $S/c $RUN
echo "$RUN processed at $(date +%H:%M)"
