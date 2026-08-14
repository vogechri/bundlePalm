# DRS Parameter Sensitivity Gate

Fresh K24/I30 runs use the promoted Stage-C C1+C5 stack. Each arm changes exactly one global parameter from the shared control.

| Family | Arm | Geomean SSE/control | Summed SSE/control | W/T/L | Worst | Time/control | Oracles | Accepts | Fallbacks | Step-cap hits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm | control | 1.000000000 | 1.000000000 | 0/6/0 | 1.000000000 | 1.000000 | 302 | 104 | 11 | 0 |
| 1dsfm | block_1em4 | 0.988213811 | 0.955259726 | 3/0/3 | 1.068311493 | 0.974681 | 303 | 105 | 7 | 0 |
| bal | control | 1.000000000 | 1.000000000 | 0/5/0 | 1.000000000 | 1.000000 | 261 | 102 | 4 | 1 |
| bal | block_1em4 | 1.005533949 | 1.004534517 | 2/0/3 | 1.015272495 | 1.042784 | 269 | 109 | 7 | 3 |

This gate is a safety filter, not a promotion cohort. A parameter advances only as one frozen global value to the established six-1DSfM plus five-BAL development gate.

## Decision

Reject block regularization `1e-4` as the common Stage-C setting. It improves
the six-scene 1DSfM geometric/summed SSE to `0.988214x/0.955260x` control, but
loses three scenes and reaches a `1.068311x` Piccadilly tail. More importantly,
the frozen five-scene BAL cohort regresses to `1.005534x` geometric and
`1.004535x` summed SSE, W/T/L `2/0/3`, at `1.042784x` optimization time.

Retain block regularization `5e-5`. Do not interpolate or run held-out/all-29:
the predeclared common-policy development criterion is already falsified.
