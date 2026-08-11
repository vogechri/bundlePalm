# C4 post-crash all-15 K24 decision

This is the final fixed-policy breadth gate for the maintained C4 comparison.
It applies the same `K24/I30/L1` shared-only direct-tangent DRS configuration
used at K2, K24 Roman/Trafalgar, and six-scene breadth. No inner tolerance,
trust policy, or outer policy was tuned between gates.

## Aggregate result

- Schur-PCG / Nesterov summed SSE: `0.9851x`
- geometric-mean per-scene SSE ratio: `1.0055x`
- SSE wins: `8/15`
- geometric-mean optimization-time ratio: `0.9317x`
- runtime wins: `12/15`
- worst quality regression: NYC Library `1.1953x`
- next-worst quality regression: Piccadilly `1.0913x`

Schur-PCG is a useful speed-oriented C4 option and has bounded, reliable
relative-residual stopping, but it is not a safe universal replacement for the
finite Nesterov trajectory. The aggregate SSE gain is concentrated enough that
the geometric mean and worst-scene behavior regress.

## Decision

Keep Nesterov as the named unchanged baseline/default inner trajectory. Keep
Schur-PCG independently switchable as C4 and report its quality-for-time
tradeoff. Do not tune tolerances from this matrix and do not add scene-dependent
solver selection. The inner-only carry-over gate is closed; proceed to isolated
C1/C5 interaction using the unchanged Nesterov baseline, with PCG tested only
as an explicit C4 factorial level.