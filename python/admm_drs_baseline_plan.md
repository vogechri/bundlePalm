# CPU ADMM to Variable-Metric DRS Baseline Plan

Updated: 2026-08-10

## Goal

Build one parallel CPU benchmark family on the standard Snavely BAL objective.
Every stage must use the same landmark partition, initialization, Ceres local
residual, final-state evaluator, stopping targets, and resource accounting.
This isolates algorithmic differences from DABA's weighted-ray objective and
from CPU/GPU implementation differences.

The implementation must be a separate coordinator, `client_admm.py`, sharing
only tested infrastructure with `client_acc.py`. Do not put ADMM branches into
the existing DRS acceptance/restart loop.

## Shared Problem Model

- Each landmark and all its observations have one owner.
- Cameras observed by more than one owner are duplicated local variables.
- Local problems run concurrently through the existing ZeroMQ Ceres worker.
- The optimized objective is standard BAL L2 first; Huber is a later matrix.
- E0 independently evaluates every saved global camera/point state.
- Runs report setup, optimization, RSS, messages, bytes, rounds, and local
  Ceres work separately.

## Stage A: Plain Pixel-Objective CPU ADMM

This is an experimental CPU consensus-ADMM baseline for the standard Snavely
pixel objective, not a reproduction of DABA ADMM. It borrows DABA's initial
penalty formula, residual-balancing ratios, and dual over-relaxation, but uses
one penalty over raw nine-parameter Snavely cameras. Official DABA separates
extrinsic and intrinsic consensus/penalty adaptation and uses its weighted-ray
objective and camera representation. Results from this stage must not be
presented as DABA performance.

Status: implemented as a first CPU baseline and smoke-tested on BAL-49. The first
K2/100 run reaches SSE `27,602.67` versus centralized Ceres `26,688.53`.
Ladybug and larger matrices remain pending.

BAL-49 is protocol smoke coverage only. It is too small/easy to select research
mechanisms, tune policies, promote components, or support paper claims.

For cluster `i`, scaled-dual ADMM solves

`(x_i, l_i) <- argmin f_i(x_i, l_i) + rho_i/2 ||x_i - z + y_i||^2`.

Then, for each duplicated camera, DABA first forms the arithmetic reference

`z <- weighted_mean_i(x_i)`

and applies over-relaxation only in the scaled-dual update

`y_i <- y_i + alpha * (x_i - z)`.

Match DABA ADMM defaults where the parameterization permits:

- `rho_0 = 2.5 * observations / cameras` for camera variables;
- residual-balanced penalty increase/decrease factors `1.5 / 0.8`;
- increase when `rho_0 * ||r||^2 > 2.5 * ||s||^2`;
- decrease when `||s||^2 > 10 * rho_0 * ||r||^2`;
- rescale the scaled dual when `rho` changes;
- fixed over-relaxation `alpha = 1.5`;
- up to 20 local LM attempts / 15 accepted steps for the faithful baseline;
- no Nesterov, Anderson, restart, or variable consensus metric.

The worker needs an explicit ADMM-prior request carrying a scalar `rho` and
center `z - y`. It must not infer the penalty from the local Hessian. The
returned local objective and exact landmark snapshot must remain atomically
associated, as in the corrected DRS protocol.

### Stage A Validation

1. `K=1`: ADMM consensus and dual residuals are zero; the local solve agrees
   with standard Ceres subject to its local-iteration budget.
2. Tiny synthetic overlap: consensus and dual updates agree with a NumPy
   reference implementation after every outer iteration.
3. BAL-49: native summed local objective agrees with E0 at consensus cameras.
4. Ladybug-1723: run `K = 4, 8, 16, 30/32` and report objective/time/rounds.

## Stage B: Plain DRS / Themelis Baseline

Use the same local prox operator and scalar Euclidean metric as Stage A. Remove
all current DRS enhancements:

- no inertial step;
- no line-search extrapolation;
- no full 9x9 block metric;
- no coordinate scaling;
- fixed proximal penalty;
- `tau = 1`;
- plain arithmetic consensus in the duplicated-camera space.

This is the clean ADMM-versus-DRS comparison. ADMM and DRS should differ only
in their splitting updates, not local solvers, metrics, or implementation.

## Stage C: Add Innovations One at a Time

Use cumulative and isolated ablations. For every addition, retain the previous
stage as a runnable mode.

### Integration Rule and Current Status

Stage C remains the publication spine. Recent 1DSfM/BAL work on Schur-informed
metrics and basin acquisition extends this plan; it does not replace the C1--C5
contributions. The final implementation must remain one DRS coordinator with
feature flags, not separate experimental forks.

| Stage | Active implementation surface | Status before publication |
|---|---|---|
| B | `client_drs.py`: scalar proximal, arithmetic consensus, no outer acceleration, `tau=1` | Runnable; freeze a named plain-DRS preset and validate against the shared Stage-A prox |
| C1 | `outer_acceleration.py`, line-search fallback, restart, best-state and physical-SSE safeguards | Runnable; Themelis mode is exposed as `themelis_nesterov`; cross-family ablation remains |
| C2 | `camera-scaling=none/jacobi_initial`, scene normalization, direct left-SE3 tangent assembly | Partial; Jacobi is runnable, symmetric Ruiz and block-coordinate transforms remain unimplemented here |
| C3 | scalar/block proximal metrics, arithmetic/diagonal/full consensus, refreshed worker metrics, transient Schur majorizer | Scalable factorized cross-camera metric is runnable and cross-family gated; strong I5 quality, high factor transport and coordinator memory remain |
| C4 | `ceres_pcg`, `schur_pcg`, and `nesterov` local solvers with linear telemetry | Closed all-15 gate: PCG is a faster independently switchable ablation, while Nesterov remains the named default because PCG's per-scene geometric SSE and worst-scene tail regress |
| C5 | fixed `local-steps`, persistent trust state, and residual-driven adaptive local depth | Runnable; complete fixed-depth and adaptive-work matrices remain |
| C6 | safeguarded initial distributed Schur bootstrap, explicit worker trust rebase, and transient observability-selected majorization | Experimental globalization layer; trust rebase is implemented opt-in and needs matched validation; bootstrap is not valid before unchanged long BAL DRS |

The publication candidate should be assembled cumulatively from B through C5,
adding only components that improve the BAL/1DSfM Pareto frontier. C6 may be an
optional initialization/globalization contribution if its long-horizon basin
retention problem is solved. Repeated final Schur corrections are polishing and
must remain outside the claimed DRS core.

The required paper matrix therefore has two views:

1. isolated ablations: B plus exactly one of C1--C6;
2. cumulative ablations: B, B+C1, B+C1+C2, and so on in the accepted order.

Every row must use the same partitions, local model, stopping rule, evaluator,
and resource accounting. A component can stay available in code without being
enabled in the final preset; availability is not evidence for a paper claim.

### C1. Safeguarded Fast DRS

- Themelis fast-DRS/Nesterov search direction on the DRS state.
- Merit-function line search with plain-DRS fallback.
- Adaptive restart and exact best-state restoration.
- Non-finite and metric-ceiling safeguards.

Compare plain DRS, unsafeguarded inertia, fallback only, fallback plus restart,
and the full safeguard.

The full safeguarded C1 breadth gate is positive. At K24/I5 it reaches
`0.873033x` matched block SSE over all 15 1DSfM scenes, W/T/L `7/8/0`, at
`1.019837x` optimization time and `1.138767x` prox-oracle count. BAL
1490/1778/3068 reach `0.892284/0.876836/0.600108x`, geometric mean `0.777231x`,
at `1.137850x` overall time. Promote C1 as the current cumulative Stage-C base.
Do not include fixed C3 in the cross-family default; retain it as an isolated
1DSfM component and interaction ablation.

Coherent C1+C3 execution is now implemented: accelerated worker oracles request
the same frozen factorized metric and coordinator trials use the same exact
factorized projection. In the Roman/BAL1490 sentinel factorial, C1+C3 is
`0.404427x/1.618721x` matched block SSE. Roman shows positive interaction
(`0.732682x` the product of isolated effects), but BAL 1490 is nearly
multiplicative (`0.991261x`) and remains much worse than raw. Retain the
integration and Roman signal, but do not broaden C1+C3 as a cross-family preset.

### C2. Coordinate Equilibration

- none;
- diagonal Jacobi with geometric-mean normalization;
- symmetric Ruiz;
- optional block-Jacobi coordinate transform.

This is a change of coordinates, not yet a change of consensus metric.

### C3. Variable-Metric Consensus

- scalar arithmetic consensus;
- diagonal Hessian metric;
- full 9x9 camera-block metric;
- fixed versus refreshed metric.

Measure conditioning, consensus residual, bytes, and time-to-target. Separate
the benefit of scaling from the benefit of the metric itself.

The full cross-camera Schur extension is staged to preserve backward
compatibility and mathematical consistency:

1. exact sparse coupled consensus projection and quadratic residuals;
2. worker-local sparse Schur proximal metric, including symmetric copy
   multipliers and left-SE3 tangent transforms;
3. packed camera-pair block transport from workers to the coordinator;
4. one opt-in runner mode that enables the same metric in local prox and
   consensus together;
5. dense tiny-system equivalence, then Roman/Trafalgar/Vienna and BAL 52/245
   gates before larger BAL transfer.

Stages 1--4 are implemented behind `COUPLED_SCHUR_PROXIMAL_METRIC=1`. Worker
and coordinator use the same packed sparse camera-pair metric for local prox,
consensus projection, residuals, and DRE accounting. Worker prerequisites also
support arbitrary sparse camera-pair tangent transforms
`T_i^T Q_ij T_j` and symmetric copy scaling
`sqrt(m_i m_j) Q_ij`. A Roman K24/I5 reproduction matched the previous
block-diagonal trajectory bitwise before enabling coupling.

The raw exact Schur metric was too weakly anchored: Roman I1 produced a
catastrophic consensus state despite converged local PCG. The fixed coupled
setting therefore retains every signed off-diagonal block and adds its
Frobenius bound to both incident diagonals. Matched Roman K24/I5 Schur-PCG then
reached `0.554578x` block-diagonal SSE with no rejected iterations, at `4.2949x`
runtime and `14.5142x` received bytes. This is a strong quality-for-cost signal,
not yet a promoted preset.

The camera-pair materialization used for that Roman proof does not scale:
Trafalgar K24/I1 exhausts worker memory even after removing unique-camera
pairs. The scalable implementation now keeps the exact metric factorized as
`A - B V^-1 B^T` in both worker and coordinator. It uses matrix-free local and
global actions, exact-Q block-Jacobi consensus PCG, packed factor transport,
and 32 hash buckets to compute the materialized Frobenius stabilization exactly
without retaining all camera pairs.

Startup observability is measured by a state-preserving, observability-only
worker RPC. At threshold `0.55`, Roman/Trafalgar/Vienna select factorized
coupling before I1; BAL 52/245 select raw and reproduce every I1--I5 SSE value
bitwise. The lightweight selector adds only 982 received bytes over raw I1 on
both tested BAL scenes.

Corrected fixed-factor K24/I5 versus matched refreshed block Schur-PCG:

- Roman `0.551981x`, Trafalgar `0.346666x`, Vienna `0.767573x`;
- three-scene geometric mean `0.527617x`, 3/3 wins, zero rejections;
- geometric-mean runtime `3.533440x` and received bytes `5.458339x`;
- Trafalgar coordinator peak RSS is approximately `1.45 GB`.

The completed all-15 transfer gives:

- geometric-mean SSE `0.745081x` matched block, W/T/L `11/2/2`;
- factorized coupling selected on 13/15 scenes;
- Ellis Island and Montreal Notre Dame select raw and tie bitwise;
- bounded losses on Gendarmenmarkt (`1.150897x`) and Tower (`1.095249x`);
- geometric-mean runtime `2.506342x` and received bytes `4.022384x`;
- maximum candidate coordinator RSS `1.382 GiB` and all rows complete.

Fixed metric reuse transmits factors once and reuses immutable coordinator and
worker state. On Trafalgar I2 it reduces incremental received bytes from about
`254.5 MB` to `9.05 MB`. Exact folding of factors with only one active shared
camera trims the first factor payload further; immutable trial snapshots remove
duplicate coordinator copies. BAL 52/245 remain bitwise raw because freezing is
applied only when startup observability selects factorized coupling.

Factorized consensus now caches its fixed block-Jacobi inverse and warm-starts
strict-tolerance PCG from the previous consensus. This does not change the
metric, projection system, tolerance, or iteration cap. On Trafalgar K24/I5,
the compiled fixed-factor implementation falls from `37.049s` to `17.590s`
overall and from `20.460s` to `9.419s` in consensus projection. The final SSE
variation is within measured same-code worker repeatability, received bytes and
rejections are unchanged, and all 143 maintained source tests pass.

This is a strong C3 quality-for-cost contribution and a valid cross-family
component, not yet a default. The next gates are larger-BAL selector transfer
and small isolated/cumulative compositions with C1 and C5. Freeze C3 exactly as
tested; do not retune it from the two bounded all-15 losses.

The larger-BAL selector transfer is now complete and negative. BAL 1490 and
1778 select fixed C3 at startup statistics `0.596984/0.599898`, improve at I1 to
`0.642929x/0.652024x` raw, then cross above raw at I2 and end I5 at
`1.830125x/1.759679x`. BAL 3068 declines C3 and remains bitwise raw. The
three-scene geometric means are `1.476742x` SSE, `4.149767x` time, and
`5.803458x` received bytes. This rejects fixed C3 as a standalone cross-family
default. Do not tune the selector threshold on this gate; proceed with frozen
C1/C5 compositions to test whether an orthogonal Stage-C mechanism changes the
post-I1 interaction.

### C4. Local Linear Solver

Within identical LM/Gauss-Newton models compare:

- Ceres Schur-PCG;
- accelerated-gradient/Nesterov Schur solve;
- Chebyshev semi-iteration when spectral bounds are reliable;
- optional direct solve on small clusters.

The Nesterov implementation must use the same damping, preconditioned system,
termination target, and accepted nonlinear step as PCG. Report Hessian-vector
products, reductions/synchronizations, achieved linear residual, and nonlinear
model decrease. The contribution is reduced synchronization, not merely a
different iteration counter.

K1 calibration is closed. The portable BAE-style K1 package reaches
`0.993529x` Ceres aggregate SSE over all 15 1DSfM scenes, and tighter PCG does
not reproduce its Roman/Trafalgar trajectory. Do not perform more broad K1
tuning. The post-crash C4 gate now covers K2, K24, six-scene K24, and all-15
K24 with outer acceleration off. At all-15, PCG reaches `0.9851x` summed SSE,
`1.0055x` geometric-mean per-scene SSE, and `0.9317x` geometric-mean
optimization time, with `8/15` quality wins and a `1.1953x` worst regression.
Keep Nesterov as the named baseline and PCG as a separate speed-oriented C4
factorial level. Do not tune tolerances or select solvers by scene. Exact K2
reset/preserve/baseline checkpoint portfolios remain diagnostic upper bounds
and are not the proposed method.

### C5. Inexact Local Solves and Adaptive Work

- 1, 2, 5, and converged accepted local LM steps;
- fixed versus outer-residual-driven inner tolerance;
- local stationarity error versus consensus residual;
- warm-started trust-region radius and linear state.

This determines whether finite local solves preserve the observed DRS gains and
which inexact convergence assumption is defensible.

Factorized C3 now supports C5 interior-defect telemetry without applying the
incompatible block-diagonal proximal-defect formula. In the Roman/BAL1490 I5
sentinel factorial, standalone C5 is inactive and trajectory-neutral. Roman C3
raises nine clusters to depth 2 on I4--I5 and improves from `0.551981x` to
`0.541142x`; BAL 1490 raises none and remains `1.830125x`. Do not broaden C5+C3
at I5. Evaluate isolated C5 at longer horizons where the three-step rolling
window and added local work can amortize.

### C6. Schur-Informed Globalization and Basin Acquisition

- observability-selected transient Schur majorization of the DRS metric;
- one independently safeguarded distributed Schur bootstrap before DRS;
- converged-PCG and physical-SSE acceptance gates;
- consistent reset of all product-space copies after an accepted bootstrap;
- optional short two-branch basin race with independent-SSE checkpoint choice;
- best-checkpoint protection and explicit short- versus long-horizon reporting.

The transient majorizer belongs to DRS because the same metric is used in the
local proximal subproblem and the consensus projection. The bootstrap is a
separate distributed initialization layer. Do not attribute bootstrap or final
correction gains to the DRS fixed-point iteration itself.

Current startup-policy evidence: explicit post-bootstrap trust rebasing is
implemented but matched BAL-49 and Roman tests found the pre-rebase worker
radii already at the initial DABA cap and no trajectory change. A bounded
race-3 Roman test safely restored bootstrap after a worsening exploratory
branch, but did not improve the fixed I5 endpoint. Both controls remain opt-in;
neither is part of the publication preset.

## Additional Controls That Must Not Be Forgotten

1. **Over-relaxation:** ADMM `alpha = 1, 1.5, 1.8`; DRS `tau = 1` and a small
   relaxation sweep. Do not attribute over-relaxation gains to acceleration.
2. **Adaptive penalty/proximal metric:** fixed, residual-balanced scalar, and
   safeguarded block updates. Penalty changes must rescale dual/DRS states
   consistently.
3. **Partitioning:** identical partitions across methods; report camera overlap,
   residual balance, and partition hash.
4. **Consensus stopping:** primal residual, dual/fixed-point residual, and
   objective gap. Never compare at fixed outer iteration only.
5. **Local-solve work:** Ceres attempts, accepted steps, Jacobian evaluations,
   PCG or Hessian-vector iterations, and failed trials.
6. **State provenance:** cost, cameras, and landmarks must be one atomic state.
7. **Gauge and weak intrinsics:** monitor focal/distortion ranges, cheirality,
   and tail reprojection errors; add no priors unless every method receives the
   same documented prior.
8. **Communication:** distinguish algorithm payload from diagnostic snapshots;
   report bytes and synchronization rounds separately.
9. **Parallel execution:** worker tasks are concurrent, but cap total Ceres and
   Eigen threads to physical cores. Report logical partitions separately from
   physical processes/nodes.
10. **Determinism and repeats:** fixed partition seed/hash and at least three
    timed repetitions after warm-up.
11. **Robust loss:** complete the L2 matrix first; then use an identical Huber
    definition and scale in every local solver and evaluator.
12. **Asynchrony/stragglers:** only after synchronous baselines are complete;
    out-of-order replies alone are not asynchronous optimization.

## Experiment Order

The C4 inner-only K1-to-core transfer is closed. The fresh global Stage-C 2x2
selects C1 as the first cumulative rung. Retain C5 and C1+C5 in the final
architecture, but tune C5's single common threshold/work policy because its
current incremental ratio over C1 is `1.005680x` on 1DSfM and `0.999855x` on
BAL. Keep C2 and C4 as separate factorial ablations and do not use per-scene
settings or silently fold PCG into the baseline.

The next tuning sequence is global and one factor at a time: C5 high threshold,
low threshold/hysteresis, rolling window, dwell, then maximum depth. Use C1 as
the cumulative denominator and preserve C5-only as a main-effect row. The
frozen protocol is `benchmark_results/stage_c_global_c5_tuning_plan.md`.

Completed outcome: scalar controls alone did not pass both development
families; delayed activation at I5 did. Promote the common tuned C1+C5 stack
with high/low `0.35/0.20`, window/dwell `3/3`, and depth 2. Preserve plain,
C1, and C5 rows as the final cumulative ablation ladder.

1. Implement and unit-test Stage A on synthetic data; use BAL-49 only as a
   protocol smoke test.
2. Run Stage A on Ladybug-1723 at `K = 4, 8, 16, 30`.
3. Implement Stage B using the same prox request and rerun the matrix.
4. Add C1, then C2, then C3, with an ablation after each.
5. Add C4 PCG-versus-Nesterov only after outer algorithms are stable.
6. Add C5 adaptive local accuracy.
7. Compose C6 only after the C1--C5 matrix is reproducible.
8. Expand to Venice, Final, and verified 1DSfM.
9. Repeat the winning L2 modes with Huber.

The Huber `0.5` sentinel gate is complete. A shared worker/coordinator/evaluator/
Ceres objective is validated, but raw-start Huber and 30-step L2-to-Huber
continuation are off the quality/work frontier. Retain Huber capability for a
future globalization mechanism; keep tuned L2 C1+C5 as the promoted stack.

## Claim Gate

The proposed method must improve a Pareto frontier in at least one reproducible
regime: standard objective gap versus time, synchronization rounds, bytes, or
maximum worker memory. A gain over weak ADMM without its over-relaxation,
adaptive penalty, LM, and block-PCG preconditioner is not sufficient.
