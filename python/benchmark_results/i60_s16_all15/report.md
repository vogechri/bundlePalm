# Quality-Budget DRS/Schur Allocation

Date: 2026-08-12

## Selection

The current quality reference, I30+20 Schur, uses `385.234s` summed
optimization work over all 15 1DSfM scenes. A frozen five-scene I60+20 frontier
was run with all solver, damping, tolerance, safeguard, and stopping controls
unchanged. Scaling its measured correction work to all 15 selected I60+16 as
the allocation nearest the existing quality budget. Correction 16 still gave a
`1.0616%` geometric decrease on the five-scene development cohort.

## All-15 Confirmation

| Allocation | DRS s | Schur s | Total s | SSE / base-I200 | SSE / Ceres | Summed SSE / Ceres | W/L vs Ceres |
|---|---:|---:|---:|---:|---:|---:|---:|
| I60 + 16 Schur | 194.300 | 187.221 | 381.521 | 0.834064 | 1.215246 | 1.109252 | 2/13 |
| I30 + 20 Schur | 90.998 | 294.236 | 385.234 | 0.869832 | 1.267361 | 1.160986 | 2/13 |

I60+16 reaches `0.958879x` I30+20 geometrically and `0.955439x` by summed
SSE while using `0.990360x` its optimization time. It wins 10/15 scenes. The
worst regression is Tower of London at `1.147925x`; the strongest gain is Alamo
at `0.819331x`.

All 15 scenes complete without recovery exhaustion. Ellis Island stops after
13 corrections, Piccadilly after 14, and Yorkminster after 10 under the frozen
`1e-3` progress rule. The other 12 scenes reach the correction cap. Every
accepted CG solve converges and every accepted correction decreases independently
evaluated physical SSE. Peak coordinator RSS is `2.815 GiB`; summed case wall
time is `417s`.

## Decision

Promote I60+16 as the quality preset. It strictly improves the aggregate
quality/time point established by I30+20 under one global configuration. Keep
I30+20 as the frozen earlier-handoff frontier reference, not the named quality
preset.
