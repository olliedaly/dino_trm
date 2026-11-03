#!/usr/bin/env bash
# Look at a running warm start's saved epochs (512 val imgs): output and slot-attention
# selectivity, slot attention's largest slot, and per-step error (handover check).
# Usage: peek.sh <run> <epochs, e.g. 0,1,2>   (writes c/curve_peek_<run>.npz etc.)
S="$(cd "$(dirname "$0")" && pwd)"
R=$1
E=$2
cd /run/media/ollie/fedora/home/ollie/Projects/dino_trm
export HF_HOME=/run/media/ollie/fedora/home/ollie/.cache/huggingface HF_HUB_OFFLINE=1
CURVE_TAG=_peek_$R uv run python $S/curve.py $S/c $R $E 2>&1 | grep -E "Error"
CURVE_TAG=_peek_$R uv run python $S/curve_sa.py $S/c $R $E 2>&1 | grep -E "Error"
echo "output:"; SEL_TAG=_peek_$R uv run python $S/sel_curve.py $S/c "curve_peek_$R.npz:$R" 2>&1 | grep -E "^  e"
echo "slot attention map:"; SEL_KEY=attn SEL_TAG=_peek_${R}_sa uv run python $S/sel_curve.py $S/c "curve_sa_peek_$R.npz:$R" 2>&1 | grep -E "^  e"
uv run python - "$S/c" "$R" "$E" <<'EOF'
import sys
import numpy as np
C, r, eps = sys.argv[1], sys.argv[2], [int(e) for e in sys.argv[3].split(",")]
Z, A = np.load(f"{C}/curve_peek_{r}.npz"), np.load(f"{C}/curve_sa_peek_{r}.npz")
for e in eps:
    m = 1000 * Z[f"{r}|e{e}|val|mse"].mean(0)
    a = A[f"{r}|e{e}|attn"].astype(np.float32)
    tok = a.reshape(a.shape[0], a.shape[1], -1).argmax(1)
    mx = np.mean([np.bincount(t, minlength=a.shape[1]).max() / t.size for t in tok])
    print(f"e{e} per-step error " + " ".join(f"{v:.1f}" for v in m) + f"   step1-final {m[0] - m[-1]:+.1f} | SA largest slot {mx:.2f}")
EOF
