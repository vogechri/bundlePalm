# Shared-Only Proposal C1 At I90

The candidate uses K24, local Nesterov, C1, persistent DRS trust, curvature 0.4 recovery/decay, camera metric 75, shared-only camera proximal terms, and proposal damping 0.5 applied only to duplicated cameras through outer iteration 30. C1 is active through iteration 30.

| Family | SSE/base | W/L | Worst | Optimization s | Base s | Time/base | Completed | Recovery exhausted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm | 1.136224 | 1/14 | 1.391386 | 160.688 | 258.562 | 0.621 | 15 | 0 |
| bal | 1.009261 | 4/25 | 1.045295 | 806.758 | 1271.565 | 0.634 | 29 | 0 |

## Gate

Promotion requires one unchanged candidate to improve base DRS on both complete families without recovery exhaustion and without a material worst-scene regression.
