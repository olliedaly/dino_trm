# Slot-attention loop, stage 1

## Checks

- Every scored run has its arm's settings, 2,000 images and 6.3967M parameters.
- Seeds scored: A 3/3, B 3/3, C 3/3, D 3/3

## Scores

Slot attention's own map at the stated number of passes, mean of the seeds. Seeds in brackets.

| arm | what | passes | FG-ARI | mBO_i | mBO_c | ARI counting background | error x1e3 |
|---|---|---|---|---|---|---|---|
| A | slot attention trained with 3 passes | 3 | 47.99 (48.1 / 48.3 / 47.6) | 26.06 (26.1 / 26.1 / 26.1) | 32.83 (32.9 / 32.7 / 32.9) | 24.98 (24.9 / 24.8 / 25.1) | 33.57 (33.5 / 33.6 / 33.6) |
| B | A's weights, run for 8 passes at test time | 8 | 49.27 (49.6 / 49.8 / 48.4) | 26.40 (26.6 / 26.4 / 26.2) | 33.13 (33.6 / 32.8 / 33.0) | 25.15 (25.2 / 25.3 / 24.9) | 36.19 (36.1 / 35.5 / 37.0) |
| C | trained with 8 passes, gradient from the last update only | 8 | 47.94 (47.7 / 47.8 / 48.2) | 26.39 (26.2 / 26.2 / 26.7) | 33.63 (33.7 / 33.4 / 33.8) | 25.11 (24.7 / 24.9 / 25.7) | 31.27 (31.4 / 31.2 / 31.2) |
| D | trained with 8 passes, a loss after each pass | 8 | 49.04 (49.2 / 48.9 / 49.0) | 27.38 (27.3 / 27.5 / 27.4) | 34.37 (34.2 / 34.6 / 34.3) | 25.92 (25.9 / 25.9 / 26.0) | 30.95 (31.0 / 30.9 / 30.9) |

The decoder's masks, same models and passes:

| arm | FG-ARI | mBO_i | mBO_c | ARI counting background |
|---|---|---|---|---|
| A | 43.89 | 23.39 | 29.01 | 21.56 |
| B | 41.10 | 23.05 | 28.37 | 20.87 |
| C | 43.56 | 23.82 | 29.95 | 21.91 |
| D | 44.55 | 24.44 | 30.34 | 22.58 |

## Differences

Slot attention's map. Per image, averaged over the matched seeds; [95% image-bootstrap interval]; then each seed pair.

| contrast | FG-ARI | mBO_i | mBO_c | ARI counting background |
|---|---|---|---|---|
| D - C (3 seed pairs) | +1.10 [+0.65, +1.53] (+1.44 / +1.10 / +0.76) | +0.99 [+0.80, +1.19] (+1.03 / +1.24 / +0.70) | +0.74 [+0.51, +0.96] (+0.51 / +1.13 / +0.57) | +0.81 [+0.61, +1.01] (+1.10 / +1.04 / +0.29) |
| D - B (3 seed pairs) | -0.23 [-0.70, +0.23] (-0.44 / -0.82 / +0.57) | +0.98 [+0.78, +1.18] (+0.64 / +1.05 / +1.24) | +1.24 [+0.99, +1.49] (+0.58 / +1.75 / +1.38) | +0.77 [+0.57, +0.96] (+0.64 / +0.62 / +1.04) |
| C - B (3 seed pairs) | -1.33 [-1.81, -0.82] (-1.88 / -1.92 / -0.19) | -0.01 [-0.23, +0.20] (-0.39 / -0.19 / +0.54) | +0.50 [+0.25, +0.75] (+0.07 / +0.62 / +0.81) | -0.04 [-0.27, +0.18] (-0.47 / -0.42 / +0.75) |
| B - A (3 seed pairs) | +1.28 [+0.96, +1.61] (+1.56 / +1.45 / +0.84) | +0.35 [+0.22, +0.48] (+0.57 / +0.35 / +0.12) | +0.30 [+0.15, +0.47] (+0.73 / +0.13 / +0.05) | +0.18 [+0.05, +0.31] (+0.27 / +0.46 / -0.19) |

## Decision

- Rule 1, D beats C on mBO_i (mean >= +0.5, interval above 0, every seed pair above 0): **yes**
- Rule 2, D beats B on mBO_i (same test): **yes**
- Rule 3, D is not below C or B on FG-ARI, mBO_c or ARI counting background (no interval entirely below 0): **yes**

**PASS.** A loss after each pass adds mask quality beyond both the free test-time gain and the published training method. Stop and report; stage 2 needs the user's go-ahead.
