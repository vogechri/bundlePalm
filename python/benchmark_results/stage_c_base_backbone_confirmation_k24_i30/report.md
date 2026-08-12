# Base-Backbone C1/C1+C5 All-Scene Confirmation

Both rows use one global K24/I30 shared-only configuration with local Nesterov, persistent DRS trust, curvature 0.4 recovery/decay, camera metric 75, and no proposal damping. C1 is always enabled; only delayed C5 differs.

| Family | Variant | SSE/base I30 | W/L | Worst | Optimization s | Completed | Recovery exhausted |
|---|---|---:|---:|---:|---:|---:|---:|
| 1dsfm | Base backbone + C1 | 1.018965 | 4/11 | 1.162125 | 86.102 | 15 | 0 |
| 1dsfm | Base backbone + C1+C5 | 1.022528 | 5/10 | 1.264580 | 95.898 | 15 | 0 |
| bal | Base backbone + C1 | 0.998843 | 13/16 | 1.038435 | 378.537 | 29 | 0 |
| bal | Base backbone + C1+C5 | 0.998654 | 13/16 | 1.038435 | 431.127 | 29 | 0 |

| Family | C1+C5/C1 SSE | W/L | Worst |
|---|---:|---:|---:|
| 1dsfm | 1.003497 | 5/2 | 1.130368 |
| bal | 0.999812 | 5/1 | 1.000082 |

## Gate

The hybrid is globally promotable only if it improves the preserved base-I30 trajectory on both complete families without recovery exhaustion. C5 remains incremental and must justify its added work.
