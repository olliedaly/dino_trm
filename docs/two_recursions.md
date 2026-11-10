# The two recursions

This project ran two loops. Both repeat one block with the same weights, and both were
trained with a loss after every pass. They differ in where the loop sits:

```
in a module (the main experiment):   patches -> slot attention x3 -> [ module ] x8 -> decoder
inside slot attention:               patches -> [ slot attention ] x8 ---------------> decoder
```

The first did not help. The second gave the best masks in the project. The numbers are
at the end.

## What both share

```
image -> frozen DINOv3 -> 441 patch features -> ... -> 7 slots -> decoder -> rebuilt features
         (not trained)    (21 x 21 grid)                                      |
                                                    loss = error against the real features
```

- **Slots.** 7 vectors, each meant to describe one object.
- **Decoder.** Each slot is copied to all 441 positions with a position code. An MLP turns
  each copy into features and one mask value. At every patch the 7 mask values compete,
  and the output is the mask-weighted sum of the features.
- **Training.** No labels. The only signal is how well the patch features are rebuilt.

## 1. Recursion inside slot attention

```
                          441 patch features
                                  |
                                  v
                 +----------------------------------+
7 random slots   |  slot-attention update           |
(first pass) --> |  a. every patch splits one vote  |
                 |     across the 7 slots           |
       +-------> |  b. each slot averages the       |
       |         |     patches that voted for it    |
       |         |  c. a GRU and an MLP update      |
       |         |     the slot                     |
       |         +----------------------------------+
       |                          |
       |                          v
       +--- next pass -------- 7 slots -> decoder -> loss
            (same weights)
```

- **Baseline:** 3 passes, one loss after the last.
- **Recursion:** 8 passes, a loss after every pass, and the gradient cut between passes,
  so each loss trains one update. This is how TRM trains its steps.
- **Nothing is added.** The model has the baseline's 6.40M parameters.
- **The patches do the choosing.** Each patch has one vote to split, so the slots compete
  for patches and each patch ends up mostly with one slot. That map of votes is itself a
  segmentation, and it can be scored directly.

Code: `src/dino_trm/models/slot_attention.py`; run with `mode=sa_rec model.slot_iters=8
model.sa_detach=true` (`scripts/runs/run_coco_sa_rec_1step.sh`).

## 2. Recursion in a module after slot attention

```
                          441 patch features --------------------+
                                  |                              |
                                  v                              |  the module reads
                      slot attention, 3 passes                   |  the patches itself,
                                  |                              |  at every step
                                  v                              |
                       x = its 7 slots (kept fixed)              |
                                  |                              |
                                  v                              v
                 +----------------------------------------------------+
y starts as x    |  module step                                       |
z starts as  --> |  a. mix x, y and z                                 |
a learned vector |  b. 2 layers: the slots attend to each other, then |
                 |     each slot attends to the patches, then an MLP  |
       +-------> |  c. update the latent z                            |
       |         |  d. run a and b again, then update the slots y     |
       |         +----------------------------------------------------+
       |                          |
       |                          v
       +--- next step ------- y and z -> decoder(y) -> loss
            (same weights)
```

- **T=1:** one step. **T=8:** 8 steps, a loss after every step, and the gradient passed
  through all of them. One control cuts the gradient between steps instead.
- **x, y, z** are TRM's three streams: the question (slot attention's slots), the current
  answer (the slots being refined) and a latent state.
- **The module adds 2.5M parameters.** The baseline's decoder is widened so that both
  have 6.40M.
- **The slots do the choosing.** Each slot spreads its attention over all 441 patches.
  Nothing stops two slots from reading the same patches. They coordinate only by
  attending to each other.
- **Masks can only be read from the decoder,** because the module changes the slots
  after slot attention has made its map.

Code: `src/dino_trm/models/trm_module.py` and `tiny_gnn.py`; run with `mode=trm
model.gnn_cross_attn=true model.n_steps=8`.

## The difference that matters

| | inside slot attention | in a module |
|---|---|---|
| what is repeated | slot attention's own update | a transformer block added after it |
| starts from | random slots | slot attention's finished slots |
| who chooses | patches choose slots, so slots compete | slots choose patches, with no competition |
| routes from the image to the slots | one | two: through slot attention, and through the module's own reads |
| extra parameters | none | 2.5M |
| masks read from | slot attention's map, or the decoder | the decoder only |

The two routes are why the module loop went wrong. With enough steps the module learns
to build the slots from its own reads, stops using slot attention's slots, and slot
attention stops segmenting. What the small shared block then builds is slots divided by
position ([summary](../results/coco/diagnostics/recursion_summary.md), section 2).
A longer slot-attention loop has one route, so there is nothing to displace.

The two blocks also behave differently when simply run for longer than they were
trained for, with no retraining (FG-ARI, 3 seeds):

| passes at test time | 1 | 2 | 3 | 5 | 8 | 16 |
|---|---|---|---|---|---|---|
| baseline slot attention, trained with 3 (its map) | 33.8 | 45.0 | 48.0 | 48.9 | 49.3 | 49.5 |

| steps at test time | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| module, trained with 1 step (T=1) | 45.8 | 32.7 | 21.5 | 15.1 |

Slot attention's update settles and improves. The module's update, repeated, falls apart
unless it was trained that way.

## Results

3 seeds on the 2,000 COCO validation images unless marked.

| model | masks read from | FG-ARI | mBO<sub>i</sub> | mBO<sub>c</sub> | reconstruction error ×10³ |
|---|---|:--:|:--:|:--:|:--:|
| slot attention, 3 passes (baseline) | its map | 48.0 | 26.1 | 32.8 | 33.6 |
| same weights, 8 passes at test time | its map | 49.3 | 26.4 | 33.1 | |
| slot attention trained with 8 passes (1 seed) | its map | 49.2 | 27.3 | 34.2 | 31.0 |
| slot attention, 3 passes (baseline) | decoder | 43.9 | 23.4 | 29.0 | 33.6 |
| slot attention trained with 8 passes (1 seed) | decoder | 44.5 | 24.3 | 30.2 | 31.0 |
| module, T=1 | decoder | 45.8 | 25.7 | 32.9 | 29.7 |
| module, T=8 | decoder | 47.0 | 25.2 | 31.1 | 33.6 |

- **Training slot attention with 8 passes improves every score on both readouts,** and
  lowers the reconstruction error by 8%. Its masks have the highest mBO in the project.
- **Most of its FG-ARI gain is free.** The baseline reaches 49.3 by running 8 passes at
  test time. Against the same seed's baseline run that way, training with 8 passes adds
  0.7 mBO<sub>i</sub> and 0.6 mBO<sub>c</sub>.
- **The gain is on mid-sized objects.** For objects of 16 to 64 patches, the best slot's
  IoU is 50.2, against 46.8 for the baseline's map.
- **It does not move towards dividing the image by position,** which is what the module
  did at T=8.

Limits:

- The 8-pass model is one seed.
- It cuts the gradient between passes. The run script notes a version with the gradient
  passed through all 8 passes as unstable, and that version was not scored.
- Running the baseline for 8 passes at test time helps its map but hurts its decoder's
  masks (FG-ARI 43.9 to 41.1), because the decoder only ever saw 3-pass slots.
- Training slot attention as a loop that settles is not new. Chang, Griffiths and Levine
  (2022) train it with the gradient of the last update only. The version here adds a
  loss on every pass.

## Reference

- Chang, Griffiths and Levine (2022). Object Representations as Fixed Points: Training
  Iterative Refinement Algorithms with Implicit Differentiation. NeurIPS.
