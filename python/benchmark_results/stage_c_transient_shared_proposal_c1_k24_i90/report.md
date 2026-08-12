# Shared-Only Proposal C1 At I90

The candidate uses K24, local Nesterov, C1, persistent DRS trust, curvature 0.4 recovery/decay, camera metric 75, shared-only camera proximal terms, and proposal damping 0.5 applied only to duplicated cameras through outer iteration 30. C1 is active through iteration all.

| Family | SSE/base | W/L | Worst | Optimization s | Base s | Time/base | Completed | Recovery exhausted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm | 1.022809 | 6/9 | 1.384319 | 263.250 | 258.562 | 1.018 | 15 | 0 |
| bal | 1.001478 | 12/17 | 1.023838 | 1262.146 | 1271.565 | 0.993 | 29 | 0 |

## Gate

Promotion requires one unchanged candidate to improve base DRS on both complete families without recovery exhaustion and without a material worst-scene regression.
