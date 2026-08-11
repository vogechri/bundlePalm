# Stage-C Loss Factorials

Date: 2026-08-11

## Question

The fresh all-15/all-29 benchmark loses only Roman Forum, Yorkminster, and
BAL135 with C1+C5. This report separates the C1 main effect, C5 main effect,
and C1-by-C5 interaction under the exact K24/I30 final-benchmark protocol.

C1 is safeguarded Themelis outer acceleration. C5 is per-cluster adaptive local
depth. All other coordinates, metrics, trust policy, inner solver, safeguards,
and budgets are fixed.

## I30 factorial

| Scene | Plain SSE | C1/plain | C5/plain | C1+C5/plain | Interaction ratio | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Roman Forum | 7,700,698.70 | 1.098217 | 1.055628 | 1.047329 | 0.903408 | Both main effects lose, but interaction is beneficial and repairs most of the loss |
| Yorkminster | 6,696,417.58 | 1.030953 | 1.063671 | 1.184323 | 1.079999 | Both main effects lose and interaction amplifies the loss by another 8.0% |
| BAL135 | 741,518.43 | 1.002734 | 1.000000 | 1.002734 | 1.000000 | C5 never activates; the 0.27% loss is entirely C1 |

The interaction ratio is

`SSE(C1+C5) * SSE(plain) / (SSE(C1) * SSE(C5))`.

Values below one are beneficial interaction; values above one are adverse.

## Yorkminster horizon control

The normal I60 run is not a continuation of I30 because the relative safeguard
anneals with the requested total iteration count. A diagnostic
`--safeguard-annealing-iterations 30` run fixes the I30 schedule, clamps it at
its terminal ratio afterward, and reproduces all first 30 trajectory rows
exactly for all four modes.

| Checkpoint | C1/plain | C5/plain | C1+C5/plain |
|---:|---:|---:|---:|
| I30 | 1.030953 | 1.063671 | 1.184323 |
| I40 | 0.943116 | 1.043935 | 1.093068 |
| I50 | 0.918587 | 1.027449 | 1.045546 |
| I60 | 0.908028 | 1.015475 | 1.029215 |

C1's I30 Yorkminster loss is a finite-horizon lag and becomes a 9.20% win by
I60. C5 remains 1.55% worse, and C1+C5 remains 2.92% worse because the adverse
interaction persists. The I30 publication-budget blocker is therefore real,
not a missing fallback event, although its magnitude decreases with more work.

## Why fallback does not imply endpoint dominance

The C1 safeguard compares an accelerated proposal with admissibility thresholds
on the trajectory's current accepted state. It does not run a shadow plain-DRS
trajectory and choose the lower final SSE. Once an accelerated proposal is
accepted, later nominal fallback starts from a different state than plain DRS.
C5 also changes the nominal local resolvent itself, so C1 fallback does not undo
C5.

Consequently, a safeguarded C1+C5 run can remain numerically safe and still end
above a separately run plain control. All factorial runs completed without
recovery exhaustion.

## Decision

Do not reject C1 or C5. Retain both in the final architecture and retain C1+C5
as the intended combined stack. The complete global factorial, not these three
scenes alone, controls tuning and promotion. Yorkminster identifies a current
finite-budget interaction to improve with one common C5 threshold/work policy;
it does not justify per-scene switching or dropping the composition. Report the
isolated and interaction effects explicitly.

## Artifacts

- `stage_c_roman_factorial_k24_i30/`
- `stage_c_yorkminster_factorial_k24_i30/`
- `stage_c_yorkminster_factorial_k24_i60_guard30/`
- `stage_c_bal135_factorial_k24_i30/`
