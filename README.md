# DINO-TRM

A slot-attention object discovery model on top of frozen DINOv3 features. Trained
unsupervised to reconstruct the DINOv3 patch features (the DINOSAUR objective); the
per-pixel argmax of the resulting slot masks is used as a segmentation. The `trm` and
`coupled` modes add a recursive module that iterates over the slot vectors, and in
`coupled` re-runs slot attention each step, testing whether iteration helps separate
touching or overlapping objects that single-pass slot attention merges into one slot.

## What the model does, step by step

For a 336×336 image:

1. **Frozen DINOv3 ViT-S/16 backbone.** Outputs 441 patch features (a 21×21 grid, one
   per 16-px patch), each 384-dim. Runs in bf16 under `no_grad`; no gradients flow into
   it. Source: `models/backbone.py`.
2. **Project to slot dim.** LayerNorm + Linear, 384 → 256.
3. **Slot Attention** (`models/slot_attention.py`). 7 slot vectors are sampled from a
   learned Gaussian. For 3 iterations, slots attend over the 441 patch features with a
   softmax that *competes across slots* (each patch is fought over). Output: 7
   slot vectors (each 256-dim) plus a soft (7, 441) patch-to-slot assignment.
4. **Recursion.** `baseline` skips this; `trm` and `coupled` run an 8-step loop. See
   the three modes below.
5. **Spatial broadcast decoder** (`models/decoder.py`). Each slot vector is broadcast
   over a 21×21 grid, a learned positional code is added, a 4-layer MLP predicts a
   384-dim reconstruction *and* an alpha-mask logit per patch. The output feature map
   is the alpha-weighted sum across slots.
6. **Loss.** Mean squared error between the reconstructed feature map and the original
   DINOv3 patch features. When recursion is active, the loss is applied at every step
   (deep supervision). The slots have to capture object structure to predict the
   feature map DINO produced.

The 7 alpha masks (upsampled to image resolution and argmaxed across slots) give a
per-pixel slot id, which is what we score against ground-truth segmentation.

## The three modes

All share the backbone, slot attention, decoder, and reconstruction loss. They differ
in what sits between slot attention and the decoder:

- **`baseline`**: nothing. Decode directly from the slot-attention output above
  (one invocation of the module, which itself runs the standard 3 internal
  attend-and-GRU iterations). This is the DINOSAUR baseline.
- **`trm`**: an 8-step recursion that refines the 7 slot vectors. Each step is a
  small pre-norm transformer over the 7 slot tokens (self-attention + FFN). With
  `model.gnn_cross_attn=true`, each step also includes a cross-attention layer where
  the slots re-read the 441 patch features (query = slots, key/value = patches), so
  the recursion isn't blind to the image after the initial binding. Full backprop
  through all 8 steps (`tests/test_recursion_grad.py` verifies this). The
  patch-to-slot assignment from the initial slot attention is not changed; only the
  slot vectors themselves move. Sources: `models/tiny_gnn.py`, `models/trm_module.py`.
- **`coupled`**: same as `trm`, plus at every recursion step slot attention is
  *re-run* on the patches, conditioned on the current slot vectors. So patches that
  were assigned to the "wrong" slot in iteration 1 can be re-assigned in iteration 2
  as the slot vectors refine. This is what the COCO experiments below show works on
  touching same-class objects.

Compute cost per training step roughly tracks the recursion depth: `baseline` is the
cheapest, `trm` and `coupled` with T=8 are ≈5× slower (full BPTT through 8 steps).

## Results

**Headline:** moving from clean PASCAL VOC to an occlusion-rich COCO multi-object
subset, and grounding the recursion in patches (slot→patch cross-attention), doubles
the recursion gain over baseline and turns the foreground grouping score (flat on
VOC) into a +4 signal.

### COCO multi-object subset, 3-seed averaged, full 2k val, true instance masks

| Model | per-instance IoU (mBO_i) | per-class IoU (mBO_c) | foreground grouping (FG-ARI) |
|---|:--:|:--:|:--:|
| baseline | 22.9 ±0.1 | 28.6 ±0.1 | 42.5 ±0.2 |
| **trm + cross-attn** | **25.2 ±0.0** | **31.0 ±0.1** | 46.7 ±0.1 |
| **coupled + cross-attn** | 25.1 ±0.0 | 30.9 ±0.0 | **47.2 ±0.1** |
| _DINOSAUR (full COCO, reported / reproduction)_ | _26.1 / 28.0_ | _30.0 / 31.7_ | _39.4 / 40.2_ |

Std ≈ 0 across seeds, so all the gaps are well above noise. Recursion gain ≈ +2.3
mBO vs ~+1 on clean VOC; foreground grouping moves +4.2 to +4.7 where it was flat on
VOC. `coupled` sits above `trm` at every recursion depth on grouping (see
[`results/coco/fg_ari_vs_depth.png`](results/coco/fg_ari_vs_depth.png)); they tie on
mBO. Our subset is curated multi-object so the DINOSAUR reference row is a regime
check, not like-for-like.

### Qualitative: where the coupled feedback loop actually wins

The 5 COCO val images where `coupled + cross-attn` beats baseline most on the
foreground grouping score:

![Top 5 winners: coupled+xattn vs baseline on COCO val](results/coco/diagnostics/coupled_vs_baseline_winners.png)

Pattern: scenes with a few similar / touching objects (3 urinals, 3 zebras, 3
surfers) that baseline collapses into one slot but coupled's per-step re-binding
splits, the regime the design is for. The figure is produced by
[`scripts/coupled_vs_baseline_visuals.py`](scripts/coupled_vs_baseline_visuals.py).

## Environment

PyTorch 2.11 + CUDA 12.8, Python 3.12, bf16 mixed precision throughout. Managed
with `uv`.

```bash
uv sync                                       # install
uv run python scripts/sanity_check.py         # backbone forward, prints shapes
uv run python -m pytest tests/ -q             # unit tests incl. full-BPTT grad check
```

DINOv3 is a gated Hugging Face model. You need access to
`facebook/dinov3-vits16-pretrain-lvd1689m` and a token with gated-repo read
permission. The backbone wrapper also supports `facebook/dinov2-small` as an open
fallback (set `model.model_id`); patch geometry is read from the backbone so the
rest of the pipeline adapts (DINOv2 → 256 patches, DINOv3 → 441 at 336²).

## Training and evaluation

The training-time recipe was developed on Pascal VOC and stayed unchanged when we
moved to COCO (no per-dataset tuning).

```bash
# VOC (15.6k leakage-free image-only train set, validated baseline matches DINOSAUR)
uv run python scripts/build_train_index.py    # one-off: build the leakage-free index
uv run python -m dino_trm.train mode=baseline data.large_train=true optim.epochs=40
uv run python -m dino_trm.train mode=trm      data.large_train=true optim.epochs=40
uv run python -m dino_trm.train mode=coupled  data.large_train=true optim.epochs=40 \
    loss.query_ortho_weight=0.1                # anti-collapse on coupled's queries

# COCO multi-object subset (where the recursion gain shows up; ~14h sequential)
uv run python scripts/build_coco_subset.py    # 25k train + 2k val, downloads only those JPGs
bash run_coco.sh                              # baseline + trm/coupled with cross-attn, 30 ep

# Published-protocol eval + figures
uv run python scripts/eval_published.py --dataset coco        # seed-averaged
uv run python scripts/make_figures.py     --dataset coco      # depth curve + qualitative
```

Config in `configs/base.yaml` (Hydra); `pascal_voc.yaml` inherits and only overrides
`optim.epochs`. COCO artifacts are namespaced under `checkpoints/coco/` and
`results/coco/` so VOC artifacts are untouched.

Eval note: slot init is random per forward, so the published-protocol eval is
stochastic; `eval_published.py` defaults to 3-seed averaging. With T=8 recursion,
gradients from the final step reach the learned latent init `z0` only if BPTT is
intact, which `tests/test_recursion_grad.py` verifies.

## Limitations

- **COCO subset, not full COCO.** Trained on a 25k multi-object subset (eval on 2k
  val) rather than the full ~118k train split, to keep total training time tractable
  on a single 16GB consumer GPU. The DINOSAUR row in the results table is full COCO,
  so it's a regime check, not a direct comparison.
- **Fixed-depth recursion (T=8).** Both `trm` and `coupled` run a hardcoded 8-step
  inner loop (`models/trm_module.py:82`, `configs/base.yaml:17`). The original TRM
  uses ACT-style halting so depth varies per sample at training time. The halting
  head and ACT loss are wired up here (`losses.act_halting_loss`, `loss.act_weight`)
  but kept at 0 in every reported run.
- **Parameter-count confound.** The `trm` and `coupled` variants have ~64% more
  trainable parameters than baseline (6.4M vs 3.9M), so part of the mBO gain over
  baseline may come from the extra capacity rather than the recursion itself, and we
  don't have a parameter-matched, single-pass control to isolate this.

## Future work

- **Variable-depth training with ACT halting.** Enable the existing halting loss
  (`loss.act_weight>0`) so the model learns to stop early on easy scenes and recurse
  deeper on hard ones, matching the original TRM recipe. Plausibly cheaper at
  inference and a better fit for the touching-object cases where the recursion gain
  actually lives.
- **Full COCO** with the same recipe, once a longer training budget is available.
- **Occlusion benchmark** isolating the touching/overlapping-object regime where the
  recursion gain shows up.

## Layout

```
src/dino_trm/
  models/   backbone, slot_attention, decoder, tiny_gnn, trm_module, full_model
  data/     pascal_voc (labelled), voc_imageonly (15.6k unlabelled),
            coco (multi-object subset; returns true instance masks for eval)
  losses.py reconstruction + deep supervision + ACT halting + query orthogonality
  train.py  Hydra training loop with bf16 autocast + grad clip
  eval_protocol.py  full-resolution mBO_i / mBO_c / FG-ARI (uses true instances on COCO)
  utils/    logging, viz
configs/    base.yaml, pascal_voc.yaml
scripts/    build_train_index, build_coco_subset, sanity_check, download_data,
            eval_published, make_figures, coupled_vs_baseline_visuals,
            compare_matched, compare_models
tests/      slot_attention, tiny_gnn, recursion_grad (full-BPTT), losses
```
