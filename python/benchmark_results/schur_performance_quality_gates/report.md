# Schur Performance and Orthogonal Quality Gates

Date: 2026-08-12

## 1. Phase Telemetry

Each final-Schur attempt now reports:

- worker Schur assembly;
- gradient/diagonal aggregation;
- block-Jacobi setup;
- symbolic fingerprint and symbolic BSR construction;
- numeric BSR accumulation;
- CG solve;
- camera-step application plus landmark refinement;
- independent physical-SSE evaluation;
- total attempt time.

A live Roman smoke verifies every field. On the cached all-15 I60+10 run, the
coordinator linear-system totals are `28.755s` numeric BSR accumulation,
`23.106s` CG, `1.264s` preconditioner setup, `0.422s` pattern fingerprinting,
and `0.314s` first symbolic construction. The dominant coordinator targets are
numeric accumulation and CG, not symbolic graph discovery.

## 2. Symbolic BSR Cache

`bsr_low_memory` now fingerprints the ordered worker block pattern, caches the
unique-key scatter maps and SciPy BSR structure, and updates only numeric block
values on a cache hit. A pattern change invalidates and rebuilds the cache.
Synthetic tests cover numeric updates and sparsity changes.

The all-15 I60+10 confirmation is behavior-exact:

- all final endpoints and every accepted candidate SSE are bitwise identical;
- cache hits/misses: `153/15`;
- Schur time: `105.612s -> 102.351s` (`0.969122x`);
- saved time: `3.261s`.

Retain symbolic caching as the default implementation of
`bsr_low_memory`. It is a safe systems improvement, but only a modest one
because numeric accumulation and CG dominate.

The cached scatter maps also identify whether each numeric block source has
unique destinations. Destination-unique sources now use vectorized indexed
addition; duplicate sources retain `np.add.at`. Synthetic tests include an
intentional duplicate source. The all-15 confirmation remains bitwise exact:

- numeric accumulation: `28.755s -> 14.261s` (`0.495966x`);
- total Schur time: `102.351s -> 94.206s` (`0.920428x`);
- saved Schur time: `8.144s`.

Promote both symbolic caching and duplicate-safe vectorized numeric
accumulation as the default `bsr_low_memory` implementation.
The optimized all-15 balanced workflow takes `289.498s` total and reaches
`0.817178x` the nearest equal-time base-I101 endpoint.

## 3. Gauge-Deflated PCG

A default-off symmetric two-level preconditioner uses the seven
projection-preserving similarity gauge modes: three world translations, three
world rotations, and scale. Modes are converted through the exact left-SE3
camera-minus map. Synthetic SPD tests match the Jacobi solution.

Matched I60+10 result:

| Scene | CG iterations, Jacobi / gauge | Schur seconds, Jacobi / gauge | Gauge / Jacobi SSE |
|---|---:|---:|---:|
| Roman | 420 / 406 | 5.125 / 5.193 | 1.007643 |
| Trafalgar | 507 / 238 | 37.749 / 31.211 | 1.019018 |

Gauge deflation materially helps Trafalgar conditioning and time, but changes
the inexact nonlinear path and regresses both endpoints. Keep it as a
diagnostic preconditioner; do not promote or broaden it.

A matched tighter-tolerance (`rtol=1e-8`) gate does not recover
quality-preserving behavior. Relative to matched tight Jacobi, gauge deflation
is `1.034446x` SSE on Trafalgar and terminates after seven accepted corrections;
relative to the promoted default it is `1.044057x`. Roman improves relative to
tight Jacobi but gauge is slower (`5.527s` versus `4.932s`). Tight Jacobi itself
changes the nonlinear path relative to the promoted `1e-6` default. Reject
tolerance tightening as a repair for gauge-deflation trajectory drift.

A final gauge-staged/Jacobi-polished gate uses gauge deflation only to reach
`rtol=1e-3`, then finishes with Jacobi to the promoted `rtol=1e-6`. Final
residuals satisfy the original contract, but finite-Krylov path dependence
remains. Roman improves `1.69%` but Schur time rises `14.5%`; Trafalgar Schur
time falls `7.0%` but SSE regresses `1.25%`. The experimental mode was removed
from production. Matching the final residual is insufficient for trajectory
equivalence, so further preconditioner search is closed until a mechanism can
preserve the accepted nonlinear path by construction.

## 4. Removable Relative-Pose Prior

The repository already contains the globally frozen eight-mode relative
translation/rotation prior with weight `0.1`. It acts only on initialization;
ordinary DRS and Schur optimize and evaluate the standard pixel objective.
The current I60+10 workflow was rerun from all 15 frozen prior-corrected states.

Fixed prior/control endpoint ratios:

- geometric: `0.989513x`;
- summed: `1.001376x`;
- W/T/L: `9/0/6`;
- worst: Trafalgar `1.050620x`.

Tower improves to `0.831921x` control, while Madrid regresses to `1.029319x`.
Fixed integration is therefore unsafe despite aggregate geometric gain.

Reusing the historical predeclared I60 pixel-SSE branch selector before the
Schur10 tail gives `0.981565x` control geometrically and `0.987642x` summed,
but Notre Dame reverses after polishing to `1.005167x`. Thus the old selector
is not automatically safe after composition with the new Schur tail. Retain
the prior as an orthogonal basin proposal; a composition-safe selector must be
validated separately. Do not enable a fixed prior or scene-dependent routing.

Completed traces show that selecting after Schur correction nine is the first
loss-free rule (`0.981570x`, 6 wins/9 ties), while correction ten reaches
`0.980812x`, 9 wins/6 ties. Carrying both branches that far costs approximately
`1.9x` the balanced workflow, so this is a quality oracle rather than a
practical selector. Initial SSE is especially misleading: every prior state
starts with much lower pixel SSE, yet Madrid and Trafalgar finish worse under
fixed use. Pre-Schur pixel SSE, proposal disagreement, and early Schur response
therefore do not provide a validated cheap discriminator for Madrid/Tower-style
basin outcomes.

## Decision

The four gates produce one promotion and two retained diagnostics:

1. promote complete phase telemetry;
2. promote behavior-exact symbolic caching and vectorized numeric BSR
	accumulation;
3. retain gauge deflation default-off because speed gains trade against quality;
4. retain the removable pose prior as a basin proposal, not a fixed policy.

The next systems target is a quality-preserving conditioning method or a
compiled numeric reduction beyond the now-vectorized Python path. The next
quality target is a selector whose decision remains valid after Schur
polishing; existing pre-Schur pixel SSE is insufficient on Notre Dame.

The subsequent early-selector replay closes that quality target for physical
SSE alone. A development-selected I5 rule requiring a 5% prior lead is
aggregate-positive but loses held-out Notre Dame and Yorkminster, while the
first loss-free Schur-prefix rule remains correction nine at `1.900388x`
estimated control time. Do not implement or retune this branch race. See
`benchmark_results/pose_prior_early_selector_replay/report.md`.
