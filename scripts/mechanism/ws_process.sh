#!/usr/bin/env bash
# Score a finished warm-start run (6 epochs): full-val extraction and per-image stats, then
# per-epoch curves (512 val imgs) of output and slot-attention maps, with selectivity.
# Usage: ws_process.sh <run>   (foreground; GPU parts capped at 20% memory)
S="$(cd "$(dirname "$0")" && pwd)"
RUN=$1
cd /run/media/ollie/fedora/home/ollie/Projects/dino_trm
export HF_HOME=/run/media/ollie/fedora/home/ollie/.cache/huggingface HF_HUB_OFFLINE=1
[ -f checkpoints/coco/${RUN}_epoch5.pt ] || { echo "$RUN: no epoch-5 checkpoint"; exit 1; }
uv run python $S/gpu_batch.py $S/c $RUN 2>&1 | grep -vE "Warning|super\(\)|annotations|^Done|index"
uv run python $S/mech_ctrl.py $S/c $RUN 2>&1 | tail -n 1
CURVE_TAG=_$RUN uv run python $S/curve.py $S/c $RUN 0,1,2,3,4,5 2>&1 | grep -E "done|Error"
CURVE_TAG=_$RUN uv run python $S/curve_sa.py $S/c $RUN 0,1,2,3,4,5 2>&1 | grep -E "done|Error"
SEL_TAG=_$RUN uv run python $S/sel_curve.py $S/c "curve_$RUN.npz:$RUN" 2>&1 | tail -n 7
SEL_KEY=attn SEL_TAG=_${RUN}_sa uv run python $S/sel_curve.py $S/c "curve_sa_$RUN.npz:$RUN" 2>&1 | tail -n 7
uv run python $S/sa_curve_any.py $S/c curve_sa_$RUN.npz 2>&1 | tail -n 6
uv run python - "$S/c" "$RUN" <<'EOF'
import sys
import numpy as np
C, run = sys.argv[1:]
Z = np.load(f"{C}/curve_{run}.npz")
for e in range(6):
    m = 1000 * Z[f"{run}|e{e}|val|mse"].mean(0)
    print(f"  e{e} per-step error " + " ".join(f"{v:.1f}" for v in m) + f"   step1-final {m[0] - m[-1]:+.1f}")
EOF
echo "$RUN processed at $(date +%H:%M)"
