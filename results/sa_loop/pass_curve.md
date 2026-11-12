# Scores against the number of test-time passes

Slot attention's map, mean of the seeds scored: A 3/3, C 3/3, E 3/3, D 3/3, F 3/3.

## mBO_i

| arm | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| A: trained with 3 passes | 18.1 | 23.8 | 26.1 | 26.1 | 26.3 | 26.4 | 26.5 | 26.5 |
| C: 8 passes, last-update gradient | 14.6 | 16.4 | 21.1 | 23.9 | 25.8 | 26.4 | 26.8 | 27.0 |
| E: 8 passes, ordinary training | 16.2 | 17.0 | 17.5 | 19.1 | 24.5 | 26.3 | 27.1 | 27.3 |
| D: 8 passes, a loss after each | 19.7 | 24.5 | 26.0 | 26.6 | 27.1 | 27.4 | 27.6 | 27.7 |
| F: as D, one sampled pass per image | 19.0 | 24.0 | 25.6 | 26.3 | 26.8 | 27.2 | 27.4 | 27.5 |

## FG-ARI

| arm | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| A: trained with 3 passes | 33.9 | 45.0 | 48.0 | 48.3 | 49.0 | 49.3 | 49.4 | 49.5 |
| C: 8 passes, last-update gradient | 23.2 | 29.2 | 39.8 | 44.5 | 47.2 | 47.9 | 48.4 | 48.7 |
| E: 8 passes, ordinary training | 26.8 | 29.3 | 31.3 | 33.8 | 44.8 | 47.4 | 48.4 | 48.7 |
| D: 8 passes, a loss after each | 38.4 | 45.9 | 47.6 | 48.2 | 48.7 | 49.0 | 49.4 | 49.5 |
| F: as D, one sampled pass per image | 36.3 | 45.3 | 47.1 | 48.0 | 48.7 | 49.0 | 49.3 | 49.5 |

## mBO_c

| arm | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| A: trained with 3 passes | 24.8 | 30.9 | 32.8 | 32.8 | 33.1 | 33.1 | 33.2 | 33.2 |
| C: 8 passes, last-update gradient | 20.6 | 23.2 | 28.4 | 31.2 | 33.1 | 33.6 | 34.1 | 34.3 |
| E: 8 passes, ordinary training | 23.6 | 24.0 | 24.9 | 26.6 | 31.9 | 33.3 | 33.9 | 34.2 |
| D: 8 passes, a loss after each | 26.9 | 31.8 | 33.1 | 33.7 | 34.1 | 34.4 | 34.6 | 34.7 |
| F: as D, one sampled pass per image | 26.2 | 31.4 | 32.7 | 33.4 | 33.9 | 34.1 | 34.4 | 34.5 |

## ARI counting background

| arm | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| A: trained with 3 passes | 18.2 | 23.9 | 25.0 | 25.0 | 25.1 | 25.2 | 25.2 | 25.3 |
| C: 8 passes, last-update gradient | 12.2 | 15.6 | 21.5 | 23.6 | 24.7 | 25.1 | 25.5 | 25.7 |
| E: 8 passes, ordinary training | 16.5 | 18.9 | 18.8 | 20.3 | 24.2 | 24.9 | 25.1 | 25.2 |
| D: 8 passes, a loss after each | 23.1 | 25.5 | 25.7 | 25.8 | 25.8 | 25.9 | 26.1 | 26.1 |
| F: as D, one sampled pass per image | 21.7 | 25.1 | 25.2 | 25.4 | 25.6 | 25.7 | 25.8 | 25.8 |

