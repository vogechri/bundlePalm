# Refreshed Factorized C3 Gate

Date: 2026-08-13

## Protocol

Restore the historical matrix-free cross-camera metric
`A - B V^-1 B^T`, preserve startup observability threshold `0.55`,
stabilization `1`, and 32 deterministic pair buckets, then rebuild the metric
on every accepted worker solve instead of freezing the first metric. Compare
Roman Forum and BAL1490 at K24/I3 with one local Schur-PCG step, direct left-SE3
tangent equations, shared-only camera proximal terms, full 9x9 consensus,
Jacobi initial scaling, persistent DABA trust, and no outer acceleration.

A fixed-metric BAL1490 I1 sentinel was run first. It selected C3 with statistic
`0.596984` and reached SSE `177,361,162.268`, versus historical
`177,361,153.074` (relative difference `5.18e-8`).

A C3-disabled raw BAL1490 I1 sentinel reached `275,864,163.330513`, versus
historical `275,864,163.330512`. The restored machinery is behavior-neutral
when disabled.

## Result

| Scene | Selector | Refreshed I1 | Refreshed I2 | Refreshed I3 | Refreshed/fixed I3 | Refreshed/raw I3 |
|---|---:|---:|---:|---:|---:|---:|
| Roman Forum | 0.728273 | 115,019,244.689 | 50,084,737.871 | 32,424,910.572 | 1.013905 | - |
| BAL1490 | 0.596984 | 177,361,162.268 | 21,939,873.813 | 9,196,103.404 | 0.997970 | 1.526003 |

All refreshed candidates were accepted. Historical fixed I3 SSE is
`31,980,249.956` on Roman and `9,214,808.390` on BAL1490. Matched raw BAL1490
I3 SSE is `6,026,153.521`.

## Decision

Reject refreshed C3 as an endpoint-quality mechanism. Refresh strongly changes
neither selected trajectory at I3: Roman regresses `1.39%` versus fixed C3,
and BAL1490 improves only `0.20%` while remaining `52.6%` worse than raw.
The stale-factor hypothesis does not explain the BAL post-I1 crossing.

Keep the restored implementation default-off for reproducibility and future
operator research. Do not retune the global threshold or stabilization from
this two-scene gate. Production quality presets remain unchanged.
