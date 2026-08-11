# Tuned Stage-C Determinism And Timing Repeats

One warm-up run is excluded. 3 measured repeats use the same global tuned C1+C5 policy, fixed partition cache, and one thread per cluster.

| Family | Scene | SSE | Relative spread | Optimization s | Opt. CV | Overall s | Overall CV | Rejections | Oracles |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1dsfm | roman_forum | 6779649.813810 | 0.000e+00 | 8.925 | 0.0178 | 13.829 | 0.0205 | [6, 6, 6] | [49, 49, 49] |
| 1dsfm | trafalgar | 17306753.670563 | 0.000e+00 | 20.936 | 0.0112 | 31.477 | 0.0208 | [6, 6, 6] | [49, 49, 49] |
| bal | bal52 | 477621.142415 | 0.000e+00 | 4.602 | 0.0235 | 7.992 | 0.0087 | [0, 0, 0] | [58, 58, 58] |
| bal | bal3068 | 3634324.713933 | 0.000e+00 | 21.720 | 0.0186 | 37.246 | 0.0250 | [12, 12, 12] | [43, 43, 43] |

## Decision

Maximum endpoint SSE relative spread is `0.000e+00`. Timing variance is reported rather than used to alter solver settings.
