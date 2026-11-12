# Slot-attention loop, stage 3

## Checks

- Every scored run has its arm's settings, 2,000 images and 6.3967M parameters.
- Seeds scored: A 3/3, B 3/3, C 3/3, D 3/3, E 3/3, F 3/3

## Scores

Slot attention's own map at the stated number of passes, mean of the seeds. Seeds in brackets.

| arm | what | passes | FG-ARI | mBO_i | mBO_c | ARI counting background | error x1e3 |
|---|---|---|---|---|---|---|---|
| A | slot attention trained with 3 passes | 3 | 47.99 (48.1 / 48.3 / 47.6) | 26.06 (26.1 / 26.1 / 26.1) | 32.83 (32.9 / 32.7 / 32.9) | 24.98 (24.9 / 24.8 / 25.1) | 33.57 (33.5 / 33.6 / 33.6) |
| B | A's weights, run for 8 passes at test time | 8 | 49.27 (49.6 / 49.8 / 48.4) | 26.40 (26.6 / 26.4 / 26.2) | 33.13 (33.6 / 32.8 / 33.0) | 25.15 (25.2 / 25.3 / 24.9) | 36.19 (36.1 / 35.5 / 37.0) |
| C | trained with 8 passes, gradient from the last update only | 8 | 47.94 (47.7 / 47.8 / 48.2) | 26.39 (26.2 / 26.2 / 26.7) | 33.63 (33.7 / 33.4 / 33.8) | 25.11 (24.7 / 24.9 / 25.7) | 31.27 (31.4 / 31.2 / 31.2) |
| D | trained with 8 passes, a loss after each pass | 8 | 49.04 (49.2 / 48.9 / 49.0) | 27.38 (27.3 / 27.5 / 27.4) | 34.37 (34.2 / 34.6 / 34.3) | 25.92 (25.9 / 25.9 / 26.0) | 30.95 (31.0 / 30.9 / 30.9) |
| E | trained with 8 passes the ordinary way (one loss, full gradient) | 8 | 47.38 (47.0 / 47.7 / 47.5) | 26.32 (25.9 / 26.6 / 26.5) | 33.28 (32.9 / 33.6 / 33.4) | 24.87 (24.2 / 25.0 / 25.5) | 37.25 (37.4 / 37.5 / 36.9) |
| F | as D, but each image is decoded at one sampled pass | 8 | 49.05 (48.8 / 49.5 / 48.8) | 27.16 (27.1 / 27.4 / 27.0) | 34.14 (34.1 / 34.4 / 33.9) | 25.67 (25.7 / 25.6 / 25.7) | 31.34 (31.4 / 31.3 / 31.3) |

The decoder's masks, same models and passes:

| arm | FG-ARI | mBO_i | mBO_c | ARI counting background |
|---|---|---|---|---|
| A | 43.89 | 23.39 | 29.01 | 21.56 |
| B | 41.10 | 23.05 | 28.37 | 20.87 |
| C | 43.56 | 23.82 | 29.95 | 21.91 |
| D | 44.55 | 24.44 | 30.34 | 22.58 |
| E | 43.22 | 23.15 | 28.69 | 21.13 |
| F | 44.41 | 24.23 | 30.06 | 22.31 |

## Differences

Slot attention's map. Per image, averaged over the matched seeds; [95% image-bootstrap interval]; then each seed pair.

| contrast | FG-ARI | mBO_i | mBO_c | ARI counting background |
|---|---|---|---|---|
| F - B (3 seed pairs) | -0.23 [-0.66, +0.20] (-0.86 / -0.21 / +0.39) | +0.75 [+0.56, +0.95] (+0.46 / +0.96 / +0.84) | +1.01 [+0.79, +1.24] (+0.44 / +1.59 / +1.00) | +0.52 [+0.33, +0.71] (+0.50 / +0.33 / +0.72) |
| F - C (3 seed pairs) | +1.10 [+0.67, +1.53] (+1.01 / +1.71 / +0.58) | +0.77 [+0.58, +0.96] (+0.85 / +1.15 / +0.30) | +0.51 [+0.28, +0.74] (+0.37 / +0.98 / +0.18) | +0.56 [+0.36, +0.76] (+0.96 / +0.75 / -0.03) |
| F - E (3 seed pairs) | +1.67 [+1.23, +2.11] (+1.79 / +1.85 / +1.36) | +0.84 [+0.65, +1.04] (+1.23 / +0.81 / +0.49) | +0.86 [+0.62, +1.10] (+1.19 / +0.82 / +0.58) | +0.80 [+0.61, +0.99] (+1.53 / +0.67 / +0.19) |
| D - F (3 seed pairs) | -0.00 [-0.38, +0.38] (+0.42 / -0.61 / +0.18) | +0.22 [+0.05, +0.40] (+0.18 / +0.09 / +0.40) | +0.23 [+0.02, +0.44] (+0.14 / +0.16 / +0.39) | +0.25 [+0.08, +0.41] (+0.14 / +0.29 / +0.32) |
| D - E (3 seed pairs) | +1.67 [+1.20, +2.14] (+2.22 / +1.24 / +1.54) | +1.07 [+0.87, +1.27] (+1.41 / +0.89 / +0.89) | +1.09 [+0.84, +1.34] (+1.32 / +0.97 / +0.96) | +1.05 [+0.86, +1.24] (+1.67 / +0.96 / +0.51) |
| E - B (3 seed pairs) | -1.90 [-2.34, -1.45] (-2.66 / -2.06 / -0.97) | -0.09 [-0.27, +0.09] (-0.77 / +0.15 / +0.35) | +0.15 [-0.08, +0.37] (-0.75 / +0.78 / +0.42) | -0.28 [-0.46, -0.10] (-1.04 / -0.34 / +0.53) |

## Decision

- Rule 1, F beats B on mBO_i (mean >= +0.5, interval above 0, every seed pair above 0): **yes**
- Rule 2, F beats C on mBO_i (mean >= +0.5, interval above 0, every seed pair above 0): **yes**
- Rule 3, F beats E on mBO_i (mean >= +0.5, interval above 0, every seed pair above 0): **yes**
- Rule 4, F is not below B, C, E on FG-ARI, mBO_c or ARI counting background (no interval entirely below 0): **yes**

**PASS.** With each image decoded at one sampled pass, at the baseline's training cost, the per-pass loss beats the free test-time gain, last-update training and ordinary 8-pass training. Stop and report.
