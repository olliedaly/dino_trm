#!/usr/bin/env bash
cd "$(dirname "$0")/../.."   # run from the repo root
# COCO multi-object subset — one-sided control that gives the BASELINE the official
# DINOSAUR architectural capacity, to harden the recursion claim against the
# objection "your DINOSAUR baseline is artificially weak".
#
# We audited our COCO baseline against the official DINOSAUR COCO setup (Seitzer et
# al., ICLR 2023, arXiv 2209.14860, Appendix E.1; official configs in
# amazon-science/object-centric-learning-framework) and found three under-provisions,
# all corrected here:
#
#   (1) MLP decoder hidden width. E.1: "The MLP has hidden layer sizes of 1024 for
#       the MOVi datasets, and 2048 for COCO." Official config
#       `coco_feat_rec_dino_base16.yaml` uses build_mlp(features: [2048,2048,2048]).
#       Ours defaulted to decoder_hidden=1024.
#   (2) Slot-attention residual-MLP hidden size. E.1: "The MLP has a hidden size
#       equal to 4*D_slots"; Table 18 lists MLP Hidden Dim = 1024 for COCO
#       (D_slots=256). Ours was hardcoded to 512 (now model.sa_mlp_hidden).
#   (3) Pre-slot-attention feature transform. E.1: "After applying a layer norm, the
#       set of N input features is transformed to slot dimensionality D_slots by a
#       two-layer MLP with hidden size equal to the feature size D_feat, followed by
#       another layer norm." Ours was a single Linear (now model.pre_sa_mlp=true).
#
# Measured trainable params (336px, 7 slots):
#   baseline defaults            ~3.89M
#   baseline decoder_hidden=2048 ~10.84M   (run 1 below)
#   baseline full DINOSAUR       ~11.25M   (run 2 below)
#   trm + cross-attn (ours)      ~6.40M
# So this control makes the baseline LARGER than our recursive models. The recursion
# claim only stands if it survives against this stronger baseline.
#
# Same recipe as run_coco.sh (30 epochs, lr 4e-4, batch 32, COCO 25k subset). Each
# run uses cfg.log.run_name so checkpoints save under baseline_dec2048_epoch*.pt and
# baseline_dinosaur_epoch*.pt without overwriting the canonical baseline_epoch*.pt.
set -e
COMMON="data.dataset=coco log.wandb=false optim.epochs=30 optim.warmup_steps=300 \
eval.max_batches=15 log.ckpt_dir=checkpoints/coco"

echo "=== COCO BASELINE, decoder_hidden=2048 only (~10.84M params, 30ep) ==="
uv run python -m dino_trm.train mode=baseline $COMMON model.decoder_hidden=2048 log.run_name=baseline_dec2048

echo "=== COCO BASELINE, full DINOSAUR fidelity (~11.25M params, 30ep) ==="
uv run python -m dino_trm.train mode=baseline $COMMON model.decoder_hidden=2048 model.sa_mlp_hidden=1024 model.pre_sa_mlp=true log.run_name=baseline_dinosaur

echo "=== DONE ==="
