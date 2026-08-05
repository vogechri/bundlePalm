# Active `main.cpp` Parameters

Date: 2026-08-04

This inventory describes the maintained `serverTest/main.cpp` worker path used
for the validated metric `75`, block-regularization `1e-4` experiments. Values
are taken from `run_1dsfm_metric_regularization_grid.sh`, the result JSONL, and
the defaults in `main.cpp`.

## Scope

The worker and coordinator do not own the same parameters:

- `main.cpp` owns local bundle-adjustment solves, camera/landmark
  preconditioning, local trust regions, and worker parallelism;
- `client_drs.py` owns outer DRS, acceleration, safeguards, clustering, and
  block regularization;
- some request-driven coordinator settings are consumed by `main.cpp` and are
  therefore listed below.

The selected block regularization `1e-4` is **not a `main.cpp` environment
parameter**. The runner passes it to the coordinator as
`BLOCK_REGULARIZATION=1e-4`/`--block-regularization 1e-4`.

## Active Worker Configuration

| Parameter | Effective value | Source | Role |
|---|---:|---|---|
| `BUNDLE_PALM_CAMERA_UPDATE` | `se3_left` | runner override | Applies left-SE3 camera updates instead of additive updates. |
| `BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE` | `75` | runner override; default `25` | Scales the camera block-diagonal metric used by local proximal solves and consensus. This is the principal tuned worker parameter. |
| `BUNDLE_PALM_CAMERA_TRUST_DIAGONAL_SCALE` | `1e-4` | explicit runner value; same as default | Scales diagonal damping when local trust-region curvature changes. |
| `BUNDLE_PALM_CAMERA_DIAGONAL_FLOOR` | `1e-48` | explicit runner value; same as default | Relative eigenvalue floor for 9-by-9 camera metric blocks. |
| `BUNDLE_PALM_CAMERA_DIAGONAL_MAXIMUM_GUARD` | `1e-32` | default | Guards a zero/tiny maximum camera diagonal before relative flooring. |
| `BUNDLE_PALM_CAMERA_PRECONDITIONER_DIAGONAL_FLOOR` | `1e-36` | default | Floors near-zero camera preconditioner entries. |
| `BUNDLE_PALM_LANDMARK_PRECONDITIONER_FLOOR` | `1e-24` | default | Floors landmark preconditioner entries; retained as a numerical guard. |
| `BUNDLE_PALM_MINIMUM_TRUST_REGION_RADIUS` | `1e-4` | default | Lower bound for persistent local trust-region radii. |
| `BUNDLE_PALM_INITIAL_TRUST_REGION_RADIUS` | `10` | default | Initial local trust-region radius. |
| `BUNDLE_PALM_MAXIMUM_TRUST_REGION_RADIUS` | `1e6` | default | Upper bound for local trust-region radii. |
| `BUNDLE_PALM_NESTEROV_SCHUR_LIPSCHITZ` | `0.9` | default | Initial Lipschitz estimate for the worker's Nesterov Schur solver. |
| `BUNDLE_PALM_NESTEROV_STOP_CHECK_INTERVAL` | `1` | default | Checks the inner Nesterov stopping condition every iteration. |
| `BUNDLE_PALM_LOCAL_ACCEPTANCE_RATIO` | `0.9999` | default | Acceptance threshold comparing the local candidate with the nominal proximal objective. |
| `BUNDLE_PALM_BATCHED_EVALUATION` | enabled | default unless set to `0` | Uses batched residual evaluation for the standard pixel objective. |
| `BUNDLE_PALM_THREADS_PER_CLUSTER` | `1` | runner override | Sets Eigen and Ceres threads for each of the 24 worker clusters. |
| `_num_threads_machine_` | `31` | compile-time constant | Caps worker thread allocation and supplies the default `max(1, 31 / K)`. |
| Ceres linear solver | `ITERATIVE_SCHUR` | hard-coded | Local Ceres linear solver. |
| Ceres preconditioner | `SCHUR_JACOBI` | hard-coded | Preconditioner for the local Ceres solve. |
| Ceres `max_num_iterations` | `1` | hard-coded | One accepted/attempted Ceres iteration per local Ceres invocation. |
| Ceres progress output | disabled | hard-coded | Suppresses per-solve Ceres progress output. |

The metric scale, camera update mode, camera metric floor, and trust diagonal
are materially active in the selected trajectory. The other floors are active
numerical guards but were rarely or never binding in measured runs.

## Active Request-Driven Settings

These values originate in the coordinator runner but are decoded and used by
`main.cpp` for each local solve.

| Parameter | Effective value | Worker behavior |
|---|---:|---|
| clusters | `24` | Creates 24 local cluster programs. |
| local steps | `1` | Performs one local solve per outer request. |
| local solver | `nesterov` | Uses the worker Nesterov Schur path rather than the plain Ceres local solver. |
| Nesterov maximum iterations | `300` | Upper bound for the standard inner solve. |
| enhanced-inner maximum iterations | `300` | Upper bound during the enhanced inner window. |
| Nesterov minimum iterations | `1` | Requires at least one inner iteration. |
| Nesterov stop tolerance | `1e-2` | Inner convergence threshold. |
| enhanced-inner window | outer iterations `0-29` | Uses enhanced inner solves for the first 30 outer iterations. |
| trust-region policy | `drs` | Uses DRS local trust-region acceptance/recovery. |
| persistent trust region | enabled | Carries each local trust radius between outer iterations. |
| trust-region recovery ratio | `0.5` | Shrinks the radius by one half during recovery. |
| proximal metric | full block | Uses 9-by-9 camera blocks rather than a scalar proximal metric. |
| consensus metric | full block | Returns and combines full camera metric blocks. |
| block curvature multiplier | `0.4` | Adds curvature regularization to the local block model. |
| block recovery mode | `curvature` | Recovers failed local proposals by increasing block curvature. |
| maximum curvature multiplier | `64` | Caps recovery curvature. |
| curvature decay | after 5 accepted steps, ratio `0.5` | Reduces accumulated recovery curvature after sustained acceptance. |
| metric proposal disagreement scale | `0.6` | Scales proposal disagreement in the metric-aware outer request. |
| relaxation | `1` | Uses unrelaxed DRS reflection/projection updates. |
| worker-owned landmarks | enabled | Keeps exclusive landmark state on workers. |
| worker-owned cameras | disabled | Camera state remains coordinator-owned/shared. |
| packed request buffers | enabled | Uses packed request serialization. |
| objective model | standard pixel reprojection | Uses the two-dimensional BAL residual, not the DABA weighted-ray objective. |

## Coordinator-Only Context

These settings define the validated run but do not belong to `main.cpp`:

| Parameter | Effective value |
|---|---:|
| block regularization | `1e-4` |
| outer iterations | 1DSfM `1000`; BAL `90` |
| outer acceleration | Nesterov |
| line-search grid | `0,1` |
| acceleration restart | after `3` rejected accelerated steps |
| safeguard | relative DRE increase `0.01`, minimum primal ratio `1.001` |
| clustering | `landmark_scalable` |
| scene normalization | `points_p95` |
| camera scaling | `jacobi_initial` |
| consensus execution | coordinator |

## Disabled or Inactive Paths

The following available knobs are not active in the selected configuration:

| Parameter/path | Effective state | Reason |
|---|---:|---|
| `BUNDLE_PALM_FREEZE_BLOCK_METRIC` | `0` | Camera block metrics continue to update. |
| `BUNDLE_PALM_DISABLE_LANDMARK_PRECONDITIONING` | `0` | Landmark preconditioning remains enabled. |
| `BUNDLE_PALM_DISABLE_LOCAL_PROXIMAL_TERM` | `0` | The local proximal term remains enabled. |
| `BUNDLE_PALM_DIAGONAL_TRUST_DAMPING` | `0` | Optional diagonal-only damping mode is disabled. |
| `BUNDLE_PALM_BAE_TRUST_SCHEDULE` | `0` | BAE trust schedule is disabled. |
| `BUNDLE_PALM_CUMULATIVE_DIAGONAL_DAMPING` | `0` | Cumulative diagonal damping is disabled. |
| `BUNDLE_PALM_NESTEROV_RELATIVE_RESIDUAL` | `0` | Absolute inner stopping criterion is used. |
| `BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS` | `0` | Direct tangent normal equations are disabled. |
| `BUNDLE_PALM_POBA_DIAGNOSTIC_ITERATIONS` | `0` | POBA diagnostics are disabled. |
| `BUNDLE_PALM_POBA_BLOCK_RELATIVE_FLOOR` | `0` | No POBA block flooring is applied. |
| `BUNDLE_PALM_LOCAL_SOLVE_METRICS` | unset | Detailed worker diagnostic logging is disabled. |
| scalar proximal prior | disabled | Full block proximal metrics are used. |
| split camera penalty | disabled | One full camera penalty is used. |
| local Ceres solver mode | disabled | Nesterov is the selected local solver. |
| DABA objective/trust cap | inactive | Standard pixel reprojection objective is selected. |
| Schur PCG controls | inactive | The selected Nesterov path does not use the optional PCG branch. |
| centralized Ceres controls | inactive during DRS | They apply only when requesting the centralized comparison solve. |
| `_const_diag_` | not compiled | The macro is commented out. |
| `__ceresVersion__` | not compiled | The legacy Ceres metric path is commented out. |
| `_ceres_num_threads_` | not compiled | Thread count is selected through `BUNDLE_PALM_THREADS_PER_CLUSTER`. |

## Reproduction Summary

The defining worker settings are:

```text
BUNDLE_PALM_CAMERA_UPDATE=se3_left
BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE=75
BUNDLE_PALM_CAMERA_TRUST_DIAGONAL_SCALE=1e-4
BUNDLE_PALM_CAMERA_DIAGONAL_FLOOR=1e-48
BUNDLE_PALM_THREADS_PER_CLUSTER=1
```

Together with the request configuration above and coordinator block
regularization `1e-4`, these reproduce the selected parameter family. Ports are
deployment details and may be changed without changing solver behavior.