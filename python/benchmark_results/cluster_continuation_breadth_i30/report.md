# K4/I3 to K24/I27 I30 Breadth Gate

Date: 2026-08-13

## Protocol

Frozen unchanged from the three-scene gate:

- acquisition K4/I3, then fresh-process K24/I27;
- direct equal-budget K24/I30 control;
- local Schur-PCG with block-Jacobi preconditioning and DABA trust;
- direct left-SE3 tangent equations, points-p95 normalization, initial Jacobi
  scaling, shared-only proximal terms, full 9x9 consensus;
- no acceleration, branch selection, scene-specific settings, or K/split
  tuning;
- summed staged time includes K4 and K24 process initialization and partitioning.

All canonical K4 state transfers reproduce the stage endpoint SSE exactly.
Every arm completes its requested outer-iteration budget.

## Results

| Scene | Staged/direct SSE | Staged/direct time |
|---|---:|---:|
| Alamo | 0.984092 | 1.398810 |
| Ellis Island | 1.006733 | 1.151704 |
| Gendarmenmarkt | 0.983141 | 1.228192 |
| Madrid Metropolis | 0.909971 | 1.034057 |
| Montreal Notre Dame | 1.058938 | 1.528336 |
| Notre Dame | 0.941387 | 1.299602 |
| NYC Library | 0.892323 | 1.352584 |
| Piazza del Popolo | 1.021559 | 1.192793 |
| Piccadilly | 1.010127 | 1.437478 |
| Roman Forum | 0.744715 | 1.335969 |
| Tower of London | 0.874925 | 1.300248 |
| Trafalgar | 0.945498 | 1.206178 |
| Union Square | 0.957533 | 1.400006 |
| Vienna Cathedral | 0.931539 | 1.305407 |
| Yorkminster | 0.938928 | 1.379569 |
| BAL1490 | 0.997938 | 1.381323 |
| BAL1778 | 0.998288 | 1.411226 |
| BAL3068 | 1.011140 | 1.392013 |

## Aggregates

| Family | Geometric SSE | Summed SSE | W/T/L | Geometric time | Worst |
|---|---:|---:|---:|---:|---|
| 1DSfM all 15 | 0.943771 | 0.950010 | 11/0/4 | 1.297602 | Montreal 1.058938 |
| Large BAL | 1.002437 | 1.002822 | 2/0/1 | 1.394800 | BAL3068 1.011140 |
| Combined 18 | 0.953305 | 0.955586 | 13/0/5 | 1.313318 | Montreal 1.058938 |

The largest win is Roman Forum at `0.744715x`. The losses are Ellis Island,
Montreal Notre Dame, Piazza del Popolo, Piccadilly, and BAL3068.

## Decision

The breadth result validates cluster-count continuation as a strong I30
research frontier, especially on 1DSfM. It is not a globally safe fixed policy:
four of fifteen 1DSfM scenes regress, Montreal loses `5.89%`, and large-BAL
aggregate SSE is slightly worse than direct K24.

Retain the frozen schedule and resumable runner for diagnostics. Do not promote
it to fast/balanced/quality presets, tune K or the split on these outcomes, or
add scene-dependent routing. A promotion path would require a predeclared,
held-out-safe mechanism that preserves the 1DSfM aggregate gain while declining
harmful continuation without replaying both full branches.

Artifacts were produced by `serverTest/run_cluster_continuation_breadth.sh`.

## Time-Matched Control

Direct K24/I40 is the nearest fixed global wall-time control for the staged
I30 workflow. Across the combined 18-scene cohort, staged/direct-I40 time is
`1.025159x` while geometric/summed SSE is `0.997143x/1.003723x`, W/T/L
`8/0/10`. On all 15 1DSfM scenes, geometric SSE is `0.994912x` but summed SSE
is `1.003110x`, W/T/L `8/0/7`. All three large BAL scenes lose to direct I40:
BAL1490 `1.004745x`, BAL1778 `1.005733x`, BAL3068 `1.014675x`.

The apparent I30 aggregate gain is therefore predominantly a wall-time trade.
Roman Forum retains genuine path/basin evidence (`0.805697x` direct I40), but
that effect is not representative of the breadth cohort.
