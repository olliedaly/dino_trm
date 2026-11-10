# Environment for the slot-attention-loop scripts on this machine. Use as:
#   source scripts/sa_loop/env.sh && uv run python scripts/sa_loop/eval_arms.py ...
# The gated DINOv3 backbone is cached on the fedora drive; offline mode needs no token.
export HF_HOME=${HF_HOME:-/run/media/ollie/fedora/home/ollie/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTHONUNBUFFERED=1
