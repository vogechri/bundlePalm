# Stage-C Global C5 Start-Threshold Development Sweep

All candidates use one common threshold across scenes and families. Ratios compare C1+C5 with the accepted C1 rung.

| Start | 1DSfM SSE | W/L | Worst | Depth2 | Opt. | BAL SSE | W/L | Worst | Depth2 | Opt. | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.00 | 0.926088978 | 4/2 | 1.034400601 | 0.0539 | 1.408098 | 1.001196780 | 1/1 | 1.006002893 | 0.0097 | 1.320480 | no |
| 3.00 | 0.973256865 | 4/1 | 1.074936759 | 0.0312 | 1.322171 | 0.999999075 | 1/0 | 1.000000000 | 0.0008 | 1.296969 | yes |
| 5.00 | 0.972833938 | 2/3 | 1.064080447 | 0.0285 | 1.431675 | 0.999990756 | 1/0 | 1.000000000 | 0.0008 | 1.350944 | yes |
| 7.00 | 1.013239259 | 1/4 | 1.085565566 | 0.0211 | 1.356676 | 1.000000000 | 0/0 | 1.000000000 | 0.0000 | 1.343481 | no |

## Decision

Development leader: `5.00`. Freeze it before the nine-scene held-out gate; do not tune from held-out results.
