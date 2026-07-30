# DRS benchmark runner usage

This guide covers `run_drs_failure_top3_live.sh`, the recommended wrapper for
running BAL scenes with the resident C++ worker and Python DRS coordinator.

Run commands from `serverTest`:

```bash
cd /home/chvogel/bundlePalm/python/serverTest
```

The wrapper starts and stops the worker, runs each requested scene/cluster
combination, applies a per-case timeout, and stores JSON results, logs, states,
status, timing, and memory measurements under `OUTPUT_DIR`.

## Your current 3068 run

The same command, split into groups and with the cache behavior made explicit:

```bash
ALL_PROBLEMS=1 \
PROBLEM_FILTER=3068 \
CLUSTERS_LIST=30 \
ITERATIONS=90 \
THREADS_PER_CLUSTER=1 \
LOCAL_STEPS=1 \
PARTITION_CACHE=auto \
CURVATURE_DECAY_AFTER=5 \
CURVATURE_DECAY_RATIO=0.5 \
BUNDLE_PALM_CAMERA_DIAGONAL_FLOOR=1e-48 \
BUNDLE_PALM_LOCAL_SOLVE_METRICS=0 \
PERSISTENT_TRUST_REGION=1 \
OUTER_ACCELERATION=nesterov \
LINE_SEARCH_GRID=0,1 \
ACCELERATION_RESTART_AFTER=3 \
LOCAL_SOLVER=nesterov \
TRUST_REGION_POLICY=daba \
PROXIMAL_METRIC=block \
CONSENSUS_METRIC=full \
CONSENSUS_EXECUTION=single-node \
BLOCK_CURVATURE_MULTIPLIER=0.1 \
BLOCK_RECOVERY_MODE=curvature \
BLOCK_REGULARIZATION=5e-5 \
MAXIMUM_BLOCK_CURVATURE_MULTIPLIER=64 \
SAFEGUARD_MODE=relative \
MINIMUM_PRIMAL_RATIO=1.001 \
SCENE_NORMALIZATION=points_p95 \
CAMERA_SCALING=jacobi_initial \
DEBUG_OUTPUT=1 \
LIVE_OUTPUT=1 \
OVERWRITE=1 \
OUTPUT_DIR="$PWD/../benchmark_results/drs_3068" \
./run_drs_failure_top3_live.sh
```

Important differences from relying on defaults:

- `ALL_PROBLEMS=1` is required for scene 3068. The built-in short list does not
  include it.
- `PARTITION_CACHE=auto` makes cache reuse explicit.
- `CONSENSUS_EXECUTION=single-node` is appropriate when all 30 clusters are in
  this one worker process. Use `coordinator` when intentionally simulating one
  cluster per remote node.
- `OUTER_ACCELERATION=nesterov` accelerates the **outer DRS iteration**.
  `LOCAL_SOLVER=nesterov` and the `NESTEROV_*` settings control the distinct
  **inner local linear solve**.

## Partition cache and repartitioning

The cache stores only the landmark-to-cluster assignment. It does not cache
camera scaling, worker state, local solves, or optimization results.

Partition controls:

| Variable | Default | Meaning |
|---|---:|---|
| `CLUSTERING` | `landmark_scalable` | `landmark_scalable` or deterministic `landmark_scalable_stable`. |
| `RESIDUAL_BALANCE_SLACK` | `0.01` | Allowed residual-load imbalance during partitioning. |
| `MINIMUM_CAMERA_LANDMARKS` | `20` | Target minimum landmark support used by camera-aware repair. |
| `MAX_REFINEMENT_PASSES` | `3` | Maximum partition refinement passes. |
| `PARTITION_CACHE` | `auto` | `auto`, `refresh`, or `off`; behavior is detailed below. |
| `PARTITION_CACHE_DIRECTORY` | `~/.cache/bundle_palm/partitions` | Directory containing keyed `.npz` assignments. |

| Setting | Reads an existing entry | Runs partitioner | Writes entry |
|---|---:|---:|---:|
| `PARTITION_CACHE=auto` | Yes, if valid | Only on miss | On miss |
| `PARTITION_CACHE=refresh` | No | Always | Yes, replacing the matching entry |
| `PARTITION_CACHE=off` | No | Always | No |

The default directory is:

```text
~/.cache/bundle_palm/partitions
```

Override it with, for example:

```bash
PARTITION_CACHE_DIRECTORY="$PWD/../benchmark_results/cache_experiment_A"
```

The cache key includes the partitioner, cluster count, BAL incidence structure,
camera and landmark counts, residual-balance slack, minimum camera-landmark
target, and refinement-pass count. Changing one of those settings selects a
different cache entry automatically.

### Normal rerun: reuse the partition

```bash
OVERWRITE=1 PARTITION_CACHE=auto ... ./run_drs_failure_top3_live.sh
```

### Force repartitioning and update the cache

```bash
OVERWRITE=1 PARTITION_CACHE=refresh ... ./run_drs_failure_top3_live.sh
```

### Repartition once without reading or changing the cache

```bash
OVERWRITE=1 PARTITION_CACHE=off ... ./run_drs_failure_top3_live.sh
```

`OVERWRITE` and `PARTITION_CACHE` are independent:

- `OVERWRITE=1` removes the matching result/status row and reruns the case.
- It does **not** invalidate or refresh the partition cache.
- `OVERWRITE=0` skips a case already present in the result JSONL.

Each JSON result records `partitionCacheMode`, `partitionCacheStatus`
(`hit`, `written`, or `disabled`), `partitionCachePath`, and
`partitionSeconds`. It also records the worker's configured relative camera
diagonal floor as `cameraDiagonalRelativeFloor`.

## Selecting problems and experiment size

| Variable | Default | Meaning |
|---|---:|---|
| `ALL_PROBLEMS` | `0` | `0`: use the built-in 12-scene list. `1`: scan all 29 top-level `problem-*-pre.txt` files. |
| `PROBLEM_FILTER` | `646 931 1266` | Space-separated scene camera counts to run, e.g. `"89 3068"`. Empty means all scenes in the selected list. |
| `CLUSTERS_LIST` | `10 20 30` | Space-separated cluster counts, e.g. `"10 30"`. |
| `ITERATIONS` | `30` | Number of outer DRS iterations. |
| `LOCAL_STEPS` | `1` | Local proximal solves per DRS oracle call. |
| `THREADS_PER_CLUSTER` | `1` | OpenMP threads assigned to each cluster. Total runnable worker threads can approach `clusters × threads`. |
| `CASE_TIMEOUT_SECONDS` | `3600` | Timeout for each scene/cluster case. |

Examples:

```bash
# One scene from the built-in list.
PROBLEM_FILTER=931 CLUSTERS_LIST=30 ./run_drs_failure_top3_live.sh

# Scene 3068, which requires the full scanned list.
ALL_PROBLEMS=1 PROBLEM_FILTER=3068 CLUSTERS_LIST=30 \
  ./run_drs_failure_top3_live.sh

# All 29 available scenes at K=10 and K=30.
ALL_PROBLEMS=1 PROBLEM_FILTER='' CLUSTERS_LIST='10 30' \
  ./run_drs_failure_top3_live.sh
```

## Local solver and trust region

| Variable | Default | Values / meaning |
|---|---:|---|
| `LOCAL_SOLVER` | `nesterov` | `nesterov`: custom accelerated inner solve. `schur_pcg`: custom Schur-PCG solve. `ceres_pcg`: Ceres local solve. |
| `NESTEROV_MAX_ITERATIONS` | `100` | Hard maximum for the inner Nesterov solve; valid range in the worker is 1–1000. |
| `NESTEROV_MIN_ITERATIONS` | `1` | Minimum completed inner iterations before the stopping criterion may terminate the solve; must not exceed the maximum. |
| `NESTEROV_STOP_TOLERANCE` | `1e-2` | Inner relative stopping tolerance in `(0,1)`. Larger is cheaper/looser. |
| `TRUST_REGION_POLICY` | `daba` | `daba`, `drs`, or `ceres`. |
| `PERSISTENT_TRUST_REGION` | `0` | `1`: carry the local trust-region radius between oracle calls. |
| `TRUST_REGION_RECOVERY_RATIO` | `0.5` | Radius recovery factor in `(0,1]`. |
| `BUNDLE_PALM_CAMERA_DIAGONAL_FLOOR` | `1e-48` | Relative C++ floor used when regularizing camera diagonal terms. Non-default values are included in the variant name and result metadata. |
| `BUNDLE_PALM_BLOCK_SQRT_EIGENVALUE_FLOOR` | `1e-16` | Absolute floor applied after taking square roots of camera-block eigenvalues in the optional `__ceresVersion__` path; inactive in the standard Nesterov build. |
| `BUNDLE_PALM_LANDMARK_PRECONDITIONER_FLOOR` | `1e-24` | Floor on the landmark Hessian diagonal before inverse-square-root preconditioning. |
| `BUNDLE_PALM_CAMERA_TRUST_DIAGONAL_SCALE` | `1e-4` | Relative weight of the camera diagonal in trust-region damping. |
| `BUNDLE_PALM_CAMERA_DIAGONAL_MAXIMUM_GUARD` | `1e-32` | Absolute guard used when forming relative camera diagonal floors. |
| `BUNDLE_PALM_MINIMUM_TRUST_REGION_RADIUS` | `1e-4` | Lower trust-region radius used by recovery clamping and failed-attempt termination. |
| `BUNDLE_PALM_CAMERA_PRECONDITIONER_DIAGONAL_FLOOR` | `1e-36` | Absolute floor on initial camera Hessian diagonal entries. |
| `BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE` | `1e1` | Multiplier on the active camera diagonal metric used by proximal and trust-region terms. |
| `BUNDLE_PALM_CAMERA_BLOCK_SCALE` | `1e1` | Scale used when constructing the camera block step metric in the optional `__ceresVersion__` path; inactive in the standard Nesterov build. |
| `BUNDLE_PALM_LEGACY_LANDMARK_JACOBIAN_SQRT_FLOOR` | `1e-10` | JlJ square-root floor in the optional `__ceresVersion__` path; inactive in the standard Nesterov build. |
| `BUNDLE_PALM_CONST_DIAGONAL_MAXIMUM_FLOOR` | `1e-32` | Maximum-diagonal floor in the optional `_const_diag_` path; inactive in the standard build. |
| `BUNDLE_PALM_CONST_DIAGONAL_RELATIVE_FLOOR` | `1e-3` | Relative diagonal floor in the optional `_const_diag_` path; inactive in the standard build. |

All defaults above preserve the original hard-coded values. The standard build
defines neither `__ceresVersion__` nor `_const_diag_`, so the corresponding
controls have no effect unless those experimental code paths are explicitly
enabled at compile time. In the standard Nesterov build, the landmark
preconditioner floor and camera trust diagonal scale are active.

Compatibility rules:

- `ceres_pcg` requires `TRUST_REGION_POLICY=ceres` and scalar proximal metrics.
- `nesterov` and `schur_pcg` require `TRUST_REGION_POLICY=drs` or `daba`.
- The measured ten-scene inner-iteration distribution at tolerance `1e-2` was:
  median 18, p90 33, p95 36, p99 43, maximum 85. Thus 40 is an aggressive
  experimental cap, 70 is a conservative tail cap, and 100 is the baseline.

Example inner-solver sweep:

```bash
for cap in 40 70 100; do
  for tolerance in 3e-2 1e-2 3e-3; do
    OUTPUT_DIR="$PWD/../benchmark_results/nesterov_m${cap}_t${tolerance}" \
    NESTEROV_MAX_ITERATIONS="$cap" \
    NESTEROV_STOP_TOLERANCE="$tolerance" \
    PARTITION_CACHE=auto OVERWRITE=1 \
    ... ./run_drs_failure_top3_live.sh
  done
done
```

Use a separate `OUTPUT_DIR` for every parameter combination. Several numerical
parameters are not included in the generated variant name, so sharing one
directory can otherwise overwrite or skip a logically different experiment.

### Ten-scene stopping grid

Run the standard ten-scene K30/I90 grid for maximum iterations
`50, 100, 200` and stopping epsilon `1e-2, 1e-3, 5e-3` with:

```bash
./run_nesterov_stopping_grid.sh
```

The wrapper runs 90 cases and writes each parameter pair to a separate
directory under:

```text
../benchmark_results/nesterov_stopping_grid_ten_scene_k30_i90/
```

The generated `comparison.md` uses `(max iterations=100, epsilon=1e-2)` as the
baseline. It reports final `qualityMetrics.sumSquaredError` changes and both
optimization-loop and overall speedups for every scene and grid cell.

The run is resumable: completed cases are skipped by default. Set `OVERWRITE=1`
to rerun every case. `SCENE_LIST` may select a different cohort but must contain
at least ten scene IDs. `ITERATIONS`, `CLUSTERS`, `OUTPUT_ROOT`, and the two grid
lists can also be overridden.

To run a fresh matched epsilon `1e-2` versus `2e-2` comparison without requiring
the full 3×3 analysis grid:

```bash
OUTPUT_ROOT="$PWD/../benchmark_results/nesterov_epsilon_2e-2_paired_ten_scene_k30_i90" \
MAX_ITERATIONS_LIST=100 \
STOP_TOLERANCE_LIST='1e-2 2e-2' \
ANALYZE=0 \
./run_nesterov_stopping_grid.sh
```

`ANALYZE=0` permits a partial grid and leaves analysis to a dedicated paired
report. The measured ten-scene result was a 1.0532× geometric-mean optimization
speedup and +0.0700% geometric-mean final-cost change at epsilon `2e-2`.

### Ten-scene block Lipschitz grid

Sweep `BLOCK_CURVATURE_MULTIPLIER` (`Lip`) over
`1.0, 0.95, 0.925, 0.9, 0.85, 0.8` with the validated inner Nesterov settings using:

```bash
./run_block_lipschitz_grid.sh
```

This runs 60 K30/I90 cases over the standard ten-scene cohort. Results are
stored under:

```text
../benchmark_results/block_lipschitz_grid_ten_scene_k30_i90/
```

The generated `comparison.md` uses Lip `1.0` as the within-grid baseline and
records absolute final SSE, optimization time, overall time, cost change, and
speedup for every scene. The aggregate table includes summed times and
geometric-mean ratios. Runs are resumable unless `OVERWRITE=1` is set.

## Scene normalization, camera scaling, and metrics

| Variable | Default | Values / meaning |
|---|---:|---|
| `SCENE_NORMALIZATION` | `points_p95` | `points_p95`: existing `client_acc.py`-compatible median centering and p95 landmark-radius scaling to 100. `none`: preserve raw spatial coordinates after focal-sign canonicalization. |
| `CAMERA_SCALING` | `jacobi_initial` | `jacobi_initial`: compute `sqrt(diag(J_camera^T J_camera))` and normalize it to geometric mean one before the DRS cluster factor. `none`: identity scaling. |
| `CAMERA_SCALING_MAXIMUM_RATIO` | unset | Optional maximum ratio between the largest and smallest gmean-normalized Jacobi scales. |
| `CAMERA_SCALING_CLIPPING_PERCENTILE` | unset | Optional per-parameter lower/upper percentile clipping before gmean normalization, in `[0, 50)`. |
| `PROXIMAL_METRIC` | `block` in runner | `scalar` or full 9×9 camera `block`. |
| `CONSENSUS_METRIC` | `full` in runner | `arithmetic`, `scalar`, `diagonal`, or `full`. |
| `CONSENSUS_EXECUTION` | `coordinator` | `coordinator`: Python reduction, representing remote nodes. `single-node`: exact in-worker reduction when all clusters share this worker. |
| `RELAXATION` | `1.0` | DRS consensus relaxation in `(0,2)`. |
| `PENALTY_MULTIPLIER` | `1.0` | Positive scalar proximal penalty multiplier. |

Result JSON records the effective camera-scaling minimum, maximum, ratio, and
geometric mean so conditioning experiments can be compared directly.

Compatibility rules:

- Use `SCENE_NORMALIZATION=none` only as a numerical-conditioning ablation. The
  runner appends `_scene_raw` to its variant name to keep those results separate.
- Scalar proximal mode only supports `CONSENSUS_METRIC=arithmetic`.
- `CONSENSUS_EXECUTION=single-node` currently requires
  `PROXIMAL_METRIC=block CONSENSUS_METRIC=full`.
- `coordinator` and `single-node` implement the same mathematical full-block
  projection, but NumPy/LAPACK and Eigen reduction/solve ordering is not
  bitwise identical. Accelerated nonlinear trajectories can therefore diverge
  after initially negligible floating-point differences.

## Block metric recovery and decay

| Variable | Default | Meaning |
|---|---:|---|
| `BLOCK_REGULARIZATION` | `5e-5` | Positive initial regularization added to block metrics. |
| `BLOCK_CURVATURE_MULTIPLIER` | `0` | Initial nonnegative curvature multiplier. Must be positive for curvature recovery. |
| `BLOCK_RECOVERY_MODE` | `regularization` | `regularization`, `curvature`, or `measured_curvature`. |
| `MAXIMUM_BLOCK_CURVATURE_MULTIPLIER` | `16` | Positive recovery cap, not below the initial multiplier. |
| `MAXIMUM_BLOCK_REGULARIZATION` | `0.5` | Maximum regularization during recovery. |
| `CURVATURE_DECAY_AFTER` | `0` | Number of accepted iterations before reducing recovered curvature; `0` disables decay. |
| `CURVATURE_DECAY_RATIO` | `0.5` | Multiplicative decay ratio in `(0,1)`. |
| `METRIC_DIAGNOSTIC_ITERATIONS` | `0` | Diagnostic power iterations, range 0–100. Required and positive for `measured_curvature`. |
| `TARGET_TRANSFORMED_LIPSCHITZ` | `0.475` | Target in `(0,1)` used by measured-curvature recovery. |

Your settings start at multiplier `0.1`, increase it after failed safeguards up
to `64`, and halve recovered curvature after five accepted iterations.

## Outer acceleration and safeguards

| Variable | Default | Values / meaning |
|---|---:|---|
| `OUTER_ACCELERATION` | `none` | `none`, `nesterov`, `lbfgs`, or `anderson`. |
| `LINE_SEARCH_GRID` | `0,1` | `0,1`: compare nominal and full accelerated trial. `0,0.5,1`: also try a half step. |
| `ACCELERATION_RESTART_AFTER` | `3` | Positive restart/failure-history threshold used by acceleration control. |
| `SAFEGUARD_MODE` | `relative` | `relative`, `catastrophic`, or `none`. |
| `DRE_RELATIVE_INCREASE` | `0.01` | Allowed relative DRE increase; nonnegative. |
| `MINIMUM_PRIMAL_RATIO` | `1.001` | Minimum safeguard ratio; must be at least 1. |
| `SAFEGUARD_RELATIVE_DEADBAND` | `0` | Nonnegative relative numerical deadband around DRE and primal rejection thresholds; `0` preserves strict comparisons. |
| `CATASTROPHIC_RATIO` | `1000000` | Rejection threshold for catastrophic mode; at least 1. |
| `RECOVERY_PENALTY_RATIO` | `2.0` | Recovery growth factor; must exceed 1. |

Additional outer-accelerator environment controls are documented at the top of
`outer_acceleration.py`, including `BUNDLE_PALM_ACCEL_MAX_STEP_RATIO`, L-BFGS
memory/scaling controls, Anderson controls, and
`BUNDLE_PALM_NESTEROV_MAX_BETA`.

## Worker ownership and transport

These defaults are optimized for the resident single-machine worker:

| Variable | Default | Meaning |
|---|---:|---|
| `WORKER_OWNED_LANDMARKS` | `1` | Keep landmark state in the worker when possible. |
| `WORKER_OWNED_CAMERAS` | `1` | Keep camera state in the worker; requires worker-owned landmarks. |
| `PACKED_REQUEST_BUFFERS` | `1` | Send dense numeric arrays as packed byte buffers. |
| `SUPPRESS_ACCELERATED_LANDMARK_REPLIES` | `0` | Avoid selected landmark replies during accelerated trials. Requires outer acceleration. |
| `WORKER_SSE_SHADOW` | `0` | Compute worker-side SSE shadow diagnostics. |
| `WORKER_CONSENSUS_SHADOW` | `0` | Compare worker consensus against coordinator reference. Requires block/full metrics and no outer acceleration. |

Normally leave the first three enabled and the shadow modes disabled. Shadow
modes are validation tools, not production speed settings.

## Landmark refinement

| Variable | Default | Meaning |
|---|---:|---|
| `LANDMARK_REFINEMENT_STEPS` | `0` | Local landmark refinement steps, range 0–20. |
| `CONSENSUS_LANDMARK_REFINEMENT_STEPS` | `0` | Consensus landmark refinement steps, range 0–20. |
| `CONSENSUS_LANDMARK_REFINEMENT_POLICY` | `safeguard` | `safeguard`, `reporting`, or `final`. |

With outer acceleration or worker-owned landmarks, positive consensus
refinement currently requires `CONSENSUS_LANDMARK_REFINEMENT_POLICY=final`.

## Output, reruns, and diagnostics

| Variable | Default | Meaning |
|---|---:|---|
| `OUTPUT_DIR` | `benchmark_results/drs_failure_top3_i30_k10_k20_k30` | Root for this experiment. Prefer one directory per parameter combination. |
| `OVERWRITE` | `0` | `0`: skip matching completed rows. `1`: remove matching result/status row and rerun. |
| `LIVE_OUTPUT` | `1` | `1`: stream coordinator output through `tee`. `0`: log only. |
| `DEBUG_OUTPUT` | `1` | Emit detailed coordinator setup and per-iteration records. |
| `BUNDLE_PALM_LOCAL_SOLVE_METRICS` | unset/`0` | `1`: emit detailed C++ local, dispatch, and completion timing records. |
| `BUNDLE_PALM_REQUEST_PORT` | `6656` in runner | Worker request port. Change both ports for concurrent benchmark runners. |
| `BUNDLE_PALM_RESULT_PORT` | `6657` in runner | Worker result port. |
| `PYTHON` | `serverTest/.venv/bin/python` | Python interpreter. |
| `WORKER` | `serverTest/build/zeromq_cpp_server_ex` | Worker executable. |

Each `OUTPUT_DIR` contains:

```text
<variant>.jsonl       one final result row per case
status.tsv            completion/failure, elapsed time, and max RSS
logs/*.log            coordinator logs
logs/*_worker.log     C++ worker logs
states/*.npz          saved best state
memory/*.time         /usr/bin/time reports
```

The skip/overwrite key uses variant, dataset, clusters, outer iterations, local
steps, and threads per cluster. It does not include every tuning parameter.
Therefore use separate output directories for Nesterov tolerances, caps,
safeguard thresholds, or other numerical sweeps.

### Normal production-style run

```bash
PARTITION_CACHE=auto \
BUNDLE_PALM_LOCAL_SOLVE_METRICS=0 \
DEBUG_OUTPUT=1 LIVE_OUTPUT=1 OVERWRITE=0 \
... ./run_drs_failure_top3_live.sh
```

### Reprofile an existing case

```bash
PARTITION_CACHE=auto \
BUNDLE_PALM_LOCAL_SOLVE_METRICS=1 \
DEBUG_OUTPUT=0 LIVE_OUTPUT=0 OVERWRITE=1 \
OUTPUT_DIR="$PWD/../benchmark_results/profile_name" \
... ./run_drs_failure_top3_live.sh
```

Summarize detailed timing logs with:

```bash
./.venv/bin/python summarize_drs_timing.py \
  --results ../benchmark_results/profile_name/<variant>.jsonl \
  --log-dir ../benchmark_results/profile_name/logs \
  --output ../benchmark_results/profile_name/report.md
```

## Recommended workflow for a new experiment

1. Use a unique `OUTPUT_DIR` describing the parameter change.
2. Start with one scene and a short iteration budget.
3. Use `PARTITION_CACHE=refresh` only when you intentionally want a new
   partition; otherwise use `auto`.
4. Keep `BUNDLE_PALM_LOCAL_SOLVE_METRICS=0` for clean runtime measurements and
   enable it only for profiling.
5. Inspect `status.tsv` and the JSONL result before launching the full cohort.
6. Use `OVERWRITE=0` for resumable cohorts and `OVERWRITE=1` only when
   deliberately replacing matching cases.