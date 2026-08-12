# Shared-Only Proposal C1 At I90

The candidate uses K24, local Nesterov, C1, persistent DRS trust, curvature 0.4 recovery/decay, camera metric 75, shared-only camera proximal terms, and proposal damping 0.5 applied only to duplicated cameras through outer iteration all. C1 is active through iteration all.

| Family | SSE/base | W/L | Worst | Optimization s | Base s | Time/base | Completed | Recovery exhausted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm | 1.017187 | 7/8 | 1.393865 | 283.061 | 258.562 | 1.095 | 14 | 1 |
| bal | 1.001950 | 11/18 | 1.023362 | 1340.435 | 1271.565 | 1.054 | 29 | 0 |

## Gate

Promotion requires one unchanged candidate to improve base DRS on both complete families without recovery exhaustion and without a material worst-scene regression.
