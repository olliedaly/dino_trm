# DINO-TRM — Recursive Slot-Graph Reasoning for Occluded Object Recognition

Research prototype combining three ideas:

1. **Object-centric perception** — Slot Attention over a **frozen DINOv3 ViT-S/16**
   backbone, DINOSAUR-style (reconstruct backbone features, not pixels).
2. **Tiny Recursive Model (TRM) reasoning** — a small network applied recursively
   over the graph of slots (K=7, dense all-pairs → self-attention + per-slot FFN).
3. **Top-down feedback** — the recursion's latent `z` conditions the next round of
   slot binding, so perceptual binding and relational reasoning co-evolve.

Downstream hypothesis (not yet tested): for partially occluded objects, neighborhood
reasoning over slots should re-bind ambiguous slots correctly. First milestone is a
working end-to-end pipeline on a clean benchmark (PASCAL VOC 2012).

## State streams (TRM)

- `x` — initial slot embeddings from one pass of slot attention (perceptual evidence)
- `y` — refined slot answer (deeply supervised at every recursion step)
- `z` — per-slot latent reasoning state, carried across steps; conditions re-binding

## Architecture

```
Image 224² → frozen DINOv3 ViT-S/16 (bf16) → 196×384 patch features
           → Linear→256 → Slot Attention (K=7, 3 iters)
           → [TRM loop ×T: TinyGNN(x,y,z) → (y,z); ACT halt Q(z)]   (full BPTT)
           → broadcast MLP decoder → reconstruct DINOv3 features (DINOSAUR loss)
```

Three modes select the phase: `baseline` (Phase 1, slot attention once), `trm`
(Phase 2, recursion refines slots), `coupled` (Phase 3, slot attention re-binds with
`z` each step — the novel contribution).

## Environment

- Fedora, RTX 5060 Ti 16GB (Blackwell sm_120), CUDA 12.8, **PyTorch 2.11+cu128**.
- Python pinned to **3.12** (torch cu128 wheels don't ship for 3.14 yet).
- bf16 mixed precision throughout; frozen backbone runs under `no_grad`.
- Managed with `uv`. The cu128 index is `explicit` and routed only to torch/torchvision
  (see `pyproject.toml`), so everything else resolves from PyPI.

```bash
uv sync                                   # install everything
uv run python scripts/sanity_check.py     # Phase 0: backbone forward, prints shapes
uv run python scripts/download_data.py    # cache PASCAL VOC 2012
uv run python -m pytest tests/ -q         # unit tests incl. full-BPTT gradient check
```

> **DINOv3 is a gated HF model.** You need access to
> `facebook/dinov3-vits16-pretrain-lvd1689m` *and* a token with gated-repo read
> permission. The backbone wrapper also supports `facebook/dinov2-small` as an open
> fallback (set `model.model_id`); patch geometry is read from the backbone so the
> rest of the pipeline adapts automatically (DINOv2 → 256 patches, DINOv3 → 196).

## Training

```bash
uv run python -m dino_trm.train mode=baseline    # Phase 1 DINOSAUR baseline
uv run python -m dino_trm.train mode=trm         # Phase 2 recursion over slots
uv run python -m dino_trm.train mode=coupled loss.entropy_weight=0.01   # Phase 3
uv run python -m dino_trm.train log.wandb=false  # local, no wandb
```

Config in `configs/` (Hydra). Logs FG-ARI/mBO per epoch, per-step reconstruction
losses, and recursion-evolution mask figures to wandb (project `dino-trm`).

## Layout

```
src/dino_trm/
  models/  backbone, slot_attention, decoder, tiny_gnn, trm_module, full_model
  data/    pascal_voc loader (VOC palette → patch-grid labels)
  losses.py  feature-recon + deep-supervision + ACT + entropy reg
  train.py   eval.py   utils/{logging,viz}
configs/   base.yaml, pascal_voc.yaml
scripts/   sanity_check.py, download_data.py
tests/     slot_attention, tiny_gnn, recursion_grad (full-BPTT), losses
```
