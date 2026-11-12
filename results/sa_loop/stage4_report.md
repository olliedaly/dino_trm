# Slot-attention loop, stage 4

## Checks

- Every scored run has its arm's settings, 1,449 images and 6.3967M parameters.
- Seeds scored: A 3/3, B 3/3, E 3/3, F 3/3

## Scores

Slot attention's own map at the stated number of passes, mean of the seeds. Seeds in brackets.

| arm | what | passes | FG-ARI | mBO_i | mBO_c | ARI counting background | error x1e3 |
|---|---|---|---|---|---|---|---|
| A | slot attention trained with 3 passes | 3 | 23.16 (22.9 / 22.7 / 23.9) | 41.84 (41.4 / 42.2 / 41.9) | 45.86 (45.3 / 46.4 / 45.9) | 19.53 (19.5 / 19.5 / 19.6) | 32.43 (32.5 / 32.3 / 32.5) |
| B | A's weights, run for 8 passes at test time | 8 | 24.85 (24.5 / 24.6 / 25.4) | 42.25 (42.2 / 42.3 / 42.2) | 45.95 (45.8 / 46.0 / 46.0) | 19.25 (19.6 / 19.2 / 19.0) | 34.03 (34.2 / 34.1 / 33.8) |
| E | trained with 8 passes the ordinary way (one loss, full gradient) | 8 | 22.49 (22.8 / 22.2 / 22.4) | 41.81 (41.6 / 42.1 / 41.7) | 46.07 (46.2 / 46.0 / 46.0) | 18.64 (18.4 / 19.1 / 18.4) | 36.39 (36.8 / 36.0 / 36.4) |
| F | as D, but each image is decoded at one sampled pass | 8 | 23.89 (23.8 / 24.0 / 23.9) | 42.10 (41.9 / 42.2 / 42.2) | 45.89 (45.7 / 46.1 / 45.9) | 19.65 (19.6 / 19.5 / 19.8) | 30.50 (30.5 / 30.5 / 30.5) |

The decoder's masks, same models and passes:

| arm | FG-ARI | mBO_i | mBO_c | ARI counting background |
|---|---|---|---|---|
| A | 20.62 | 37.22 | 40.44 | 16.92 |
| B | 19.85 | 36.84 | 39.72 | 16.70 |
| E | 20.06 | 36.69 | 40.02 | 16.01 |
| F | 21.52 | 38.07 | 41.24 | 17.42 |

## Differences

Slot attention's map. Per image, averaged over the matched seeds; [95% image-bootstrap interval]; then each seed pair.

| contrast | FG-ARI | mBO_i | mBO_c | ARI counting background |
|---|---|---|---|---|
| F - B (3 seed pairs) | -0.96 [-1.61, -0.34] (-0.79 / -0.60 / -1.50) | -0.14 [-0.53, +0.24] (-0.30 / -0.13 / +0.00) | -0.05 [-0.45, +0.35] (-0.13 / +0.03 / -0.07) | +0.40 [+0.22, +0.58] (+0.05 / +0.35 / +0.80) |
| F - E (3 seed pairs) | +1.40 [+0.76, +2.04] (+0.94 / +1.76 / +1.50) | +0.29 [-0.11, +0.69] (+0.29 / +0.07 / +0.53) | -0.17 [-0.59, +0.23] (-0.54 / +0.06 / -0.05) | +1.01 [+0.82, +1.20] (+1.29 / +0.37 / +1.37) |
| E - B (3 seed pairs) | -2.36 [-3.05, -1.70] (-1.72 / -2.36 / -3.00) | -0.44 [-0.84, -0.05] (-0.58 / -0.20 / -0.53) | +0.12 [-0.28, +0.51] (+0.41 / -0.03 / -0.02) | -0.61 [-0.79, -0.43] (-1.24 / -0.02 / -0.57) |
| B - A (3 seed pairs) | +1.68 [+1.21, +2.18] (+1.67 / +1.91 / +1.48) | +0.41 [+0.14, +0.68] (+0.84 / +0.10 / +0.30) | +0.08 [-0.19, +0.36] (+0.49 / -0.36 / +0.13) | -0.27 [-0.38, -0.16] (+0.12 / -0.35 / -0.58) |

## Decision

- Rule 1, F beats B on mBO_i (mean >= +0.5, interval above 0, every seed pair above 0): **no**
- Rule 2, F beats E on mBO_i (mean >= +0.5, interval above 0, every seed pair above 0): **no**
- Rule 3, F is not below B, E on FG-ARI, mBO_c or ARI counting background (no interval entirely below 0): **no**

**STOP: not replicated on VOC.** F fails against B, E.
