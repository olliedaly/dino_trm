# Slot-attention loop, stage 1

## Checks

- Every scored run has its arm's settings, 2,000 images and 6.3967M parameters.
- Seeds scored: A 3/3, B 3/3, C 0/3, D 1/3

## Stage 0: existing checkpoints against the reference values

| arm | readout | score | reference | now | within tolerance |
|---|---|---|---|---|---|
| A | map | fg_ari | 47.99 | 47.99 | yes |
| A | map | mbo_i | 26.06 | 26.06 | yes |
| A | map | mbo_c | 32.83 | 32.83 | yes |
| A | map | ari_bg | 24.98 | 24.98 | yes |
| A | dec | fg_ari | 43.89 | 43.89 | yes |
| A | dec | mbo_i | 23.39 | 23.39 | yes |
| A | err | error x1e3 | 33.57 | 33.57 | yes |
| B | map | fg_ari | 49.27 | 49.27 | yes |
| B | map | mbo_i | 26.4 | 26.40 | yes |
| B | map | mbo_c | 33.13 | 33.13 | yes |
| D0 | map | fg_ari | 49.18 | 49.18 | yes |
| D0 | map | mbo_i | 27.27 | 27.27 | yes |
| D0 | map | mbo_c | 34.2 | 34.20 | yes |
| D0 | map | ari_bg | 25.85 | 25.85 | yes |
| D0 | dec | fg_ari | 44.54 | 44.54 | yes |
| D0 | err | error x1e3 | 30.97 | 30.97 | yes |

**Stage 0 PASSED.**

## Scores

Slot attention's own map at the stated number of passes, mean of the seeds. Seeds in brackets.

| arm | what | passes | FG-ARI | mBO_i | mBO_c | ARI counting background | error x1e3 |
|---|---|---|---|---|---|---|---|
| A | slot attention trained with 3 passes | 3 | 47.99 (48.1 / 48.3 / 47.6) | 26.06 (26.1 / 26.1 / 26.1) | 32.83 (32.9 / 32.7 / 32.9) | 24.98 (24.9 / 24.8 / 25.1) | 33.57 (33.5 / 33.6 / 33.6) |
| B | A's weights, run for 8 passes at test time | 8 | 49.27 (49.6 / 49.8 / 48.4) | 26.40 (26.6 / 26.4 / 26.2) | 33.13 (33.6 / 32.8 / 33.0) | 25.15 (25.2 / 25.3 / 24.9) | 36.19 (36.1 / 35.5 / 37.0) |
| C | trained with 8 passes, gradient from the last update only | 8 |  |  |  |  | |
| D | trained with 8 passes, a loss after each pass | 8 | 49.18 (49.2) | 27.27 (27.3) | 34.20 (34.2) | 25.85 (25.9) | 30.97 (31.0) |

The decoder's masks, same models and passes:

| arm | FG-ARI | mBO_i | mBO_c | ARI counting background |
|---|---|---|---|---|
| A | 43.89 | 23.39 | 29.01 | 21.56 |
| B | 41.10 | 23.05 | 28.37 | 20.87 |
| D | 44.54 | 24.30 | 30.16 | 22.52 |

## Differences

Slot attention's map. Per image, averaged over the matched seeds; [95% image-bootstrap interval]; then each seed pair.

| contrast | FG-ARI | mBO_i | mBO_c | ARI counting background |
|---|---|---|---|---|
| D - B (1 seed pair) | -0.44 [-1.15, +0.27] (-0.44) | +0.64 [+0.33, +0.95] (+0.64) | +0.58 [+0.22, +0.94] (+0.58) | +0.64 [+0.36, +0.94] (+0.64) |
| B - A (3 seed pairs) | +1.28 [+0.96, +1.61] (+1.56 / +1.45 / +0.84) | +0.35 [+0.22, +0.48] (+0.57 / +0.35 / +0.12) | +0.30 [+0.14, +0.46] (+0.73 / +0.13 / +0.05) | +0.18 [+0.05, +0.31] (+0.27 / +0.46 / -0.19) |

## Decision

**INCOMPLETE.** All four arms need 3 correctly configured seeds before the rule applies.
