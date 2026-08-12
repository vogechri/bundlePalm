# Base-Backbone C1 Structural Factorial

All arms use K24/I30, local Nesterov, C1, persistent DRS trust, curvature 0.4 recovery/decay, and camera metric 75. The structural factors are legacy all-camera versus shared-only proximal treatment and proposal damping 0.5 versus none. Fixed proposal damping acts only on duplicated cameras in shared-only mode.

| Family | Arm | SSE/base I30 | W/L | Worst | Optimization s | Completed | Recovery exhausted |
|---|---|---:|---:|---:|---:|---:|---:|
| 1dsfm | Legacy all-camera + proposal 0.5 + C1 | 0.946905 | 12/3 | 1.026720 | 90.442 | 15 | 0 |
| 1dsfm | Legacy all-camera + no proposal + C1 | 0.986106 | 8/7 | 1.132180 | 80.250 | 15 | 0 |
| 1dsfm | Shared-only + no proposal + C1 | 1.018965 | 4/11 | 1.162125 | 86.102 | 15 | 0 |
| 1dsfm | Shared-only + proposal 0.5 + C1 | 0.992977 | 8/7 | 1.122462 | 89.645 | 15 | 0 |
| bal | Legacy all-camera + proposal 0.5 + C1 | 1.002873 | 13/16 | 1.068080 | 404.909 | 28 | 1 |
| bal | Legacy all-camera + no proposal + C1 | 0.998630 | 13/16 | 1.038435 | 367.293 | 29 | 0 |
| bal | Shared-only + no proposal + C1 | 0.998843 | 13/16 | 1.038435 | 378.537 | 29 | 0 |
| bal | Shared-only + proposal 0.5 + C1 | 0.997428 | 14/15 | 1.038205 | 412.191 | 29 | 0 |

| Family | Proposal effect, legacy | Shared-only effect, no proposal | Shared-only effect, proposal |
|---|---:|---:|---:|
| 1dsfm | 0.960247 | 1.033321 | 1.048656 |
| bal | 1.004249 | 1.000213 | 0.994570 |

Ratios below one favor the first named factor level.
