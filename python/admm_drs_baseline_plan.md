# CPU ADMM to Variable-Metric DRS Baseline Plan

Updated: 2026-07-26

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

Status: implemented as a first CPU baseline and validated on BAL-49. The first
K2/100 run reaches SSE `27,602.67` versus centralized Ceres `26,688.53`.
Ladybug and larger matrices remain pending.

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

### C1. Safeguarded Fast DRS

- Themelis fast-DRS/Nesterov search direction on the DRS state.
- Merit-function line search with plain-DRS fallback.
- Adaptive restart and exact best-state restoration.
- Non-finite and metric-ceiling safeguards.

Compare plain DRS, unsafeguarded inertia, fallback only, fallback plus restart,
and the full safeguard.

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

### C5. Inexact Local Solves and Adaptive Work

- 1, 2, 5, and converged accepted local LM steps;
- fixed versus outer-residual-driven inner tolerance;
- local stationarity error versus consensus residual;
- warm-started trust-region radius and linear state.

This determines whether finite local solves preserve the observed DRS gains and
which inexact convergence assumption is defensible.

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

1. Implement and unit-test Stage A on synthetic and BAL-49.
2. Run Stage A on Ladybug-1723 at `K = 4, 8, 16, 30`.
3. Implement Stage B using the same prox request and rerun the matrix.
4. Add C1, then C2, then C3, with an ablation after each.
5. Add C4 PCG-versus-Nesterov only after outer algorithms are stable.
6. Add C5 adaptive local accuracy.
7. Expand to Venice, Final, and verified 1DSfM.
8. Repeat the winning L2 modes with Huber.

## Claim Gate

The proposed method must improve a Pareto frontier in at least one reproducible
regime: standard objective gap versus time, synchronization rounds, bytes, or
maximum worker memory. A gain over weak ADMM without its over-relaxation,
adaptive penalty, LM, and block-PCG preconditioner is not sufficient.
