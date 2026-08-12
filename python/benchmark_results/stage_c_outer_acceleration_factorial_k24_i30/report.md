# Matched Outer-Acceleration Sentinel Factorial

All rows use K24/I30, local Nesterov, persistent DRS trust, curvature 0.4 recovery/decay, camera metric 75, shared-only camera proximal terms, proposal damping 0.5 on duplicated cameras, and binary line search `{1,0}`. Only the outer accelerator varies.

| Family | Accelerator | SSE/plain | W/L | Worst | Opt. s | Time/plain | Oracles | Accepts | Fallbacks | Completed | Recovery exhausted |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm | none | 1.000000 | 0/0 | 1.000000 | 16.545 | 1.000 | 60 | 0 | 0 | 2 | 0 |
| 1dsfm | nesterov | 0.779489 | 1/1 | 1.065011 | 27.187 | 1.643 | 106 | 34 | 9 | 2 | 0 |
| 1dsfm | themelis_nesterov | 0.638048 | 2/0 | 0.796664 | 27.424 | 1.658 | 108 | 39 | 6 | 2 | 0 |
| 1dsfm | lbfgs | 0.765693 | 1/1 | 1.039542 | 34.298 | 2.073 | 120 | 34 | 25 | 2 | 0 |
| 1dsfm | anderson | 0.732168 | 2/0 | 0.995884 | 34.132 | 2.063 | 120 | 45 | 8 | 2 | 0 |
| bal | none | 1.000000 | 0/0 | 1.000000 | 16.293 | 1.000 | 60 | 0 | 0 | 2 | 0 |
| bal | nesterov | 0.859211 | 2/0 | 0.914690 | 28.034 | 1.721 | 112 | 51 | 0 | 2 | 0 |
| bal | themelis_nesterov | 0.838088 | 2/0 | 0.879390 | 29.813 | 1.830 | 116 | 54 | 2 | 2 | 0 |
| bal | lbfgs | 0.917322 | 1/1 | 1.057031 | 29.752 | 1.826 | 110 | 44 | 2 | 1 | 1 |
| bal | anderson | 0.941974 | 1/1 | 1.093251 | 33.014 | 2.026 | 118 | 47 | 2 | 1 | 1 |

No mode is selected by scene.
