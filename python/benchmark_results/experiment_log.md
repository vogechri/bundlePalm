# Experiment Log

## 2026-07-26: E0 Ladybug-49 harness validation

Purpose: validate common final-state metrics and matched-initialization
time-to-target machinery. This is not a paper performance result.

Hardware: Intel i9-13900K, 32 GiB RAM. Objective: L2. Both solvers used the
same focal-sign-adjusted and scene-normalized BAL initialization with initial
SSE `1,701,824.92136168`.

| Method | Configuration | Final SSE | Mean reprojection error | Time to DRS final SSE |
|---|---|---:|---:|---:|
| Variable-metric DRS | 20 outer iterations, 30 clusters | 27,081.306 | 0.5866 px | 2.692 s |
| Ceres iterative Schur | 16 threads, Schur-Jacobi | 26,688.552 | 0.5796 px | 0.096 s |

Outcome: the common evaluator agrees with native DRS SSE to `8.4e-8` relative
error. Ladybug-49 does not support a DRS speed claim under this configuration.
No repeated-run, converged-method, memory, distributed, or cloud-cost claim is
made from this diagnostic.

Artifacts:

- `ladybug49_drs_matched_trajectory_v2.jsonl`
- `ladybug49_ceres_canonical_i40_t16.json`
- `ladybug49_ceres_canonical_i40_t16.metrics.json`

## 2026-07-26: DABA-protocol L2 matrix

Status: prepared. Run order is Ladybug-1723, Venice-1778, Final-13682; each at
`K = 4, 8, 16, 32` for 1000 DRS outer iterations. The current CPU DRS solver,
partitioner, scaling, and L2 local solves are used unchanged.

This phase reproduces the *evaluation design* of DABA Tables I, III, and IV for
our method. It does not run or port DABA and does not implement Huber.

Accuracy outputs:

- common-evaluator mean reprojection error after 1000 iterations;
- SSE and both MSE conventions;
- best state and complete objective trajectory.

Efficiency outputs:

- setup-inclusive and partition time;
- first time to each common normalized-gap target;
- first time to the published DABA/Ceres reference where objective conventions
  can be matched.

Resource outputs:

- coordinator and worker peak RSS, reported separately and in aggregate;
- serialized coordinator-to-worker and worker-to-coordinator bytes;
- outer-iteration bytes and average bytes per outer iteration;
- number of clusters as the logical-device count. These are single-host logical
  partitions, not measured network scaling.

The Huber matrix is a separate follow-up only after the L2 matrix has produced
numbers. GPU DRS and official multi-GPU DABA execution are also separate future
experiments.

Pre-launch validation: a `K=4`, two-iteration Ladybug-1723 probe initially
showed that native worker cost was evaluated against hidden worker landmarks,
not necessarily the candidate landmarks returned for the saved state. Cost
queries now include the explicit physical candidate landmarks. The unchanged
L2 probe then achieved native/common SSE relative error `2.23e-9` and measured
`10,249,773` serialized bytes per outer iteration. This is a provenance and
measurement fix, not an optimization change.

## Queued: DABA Table II 1DSfM extension

The six large 1DSfM rows are part of the intended DABA-protocol replication,
but are queued after the active three-dataset BAL matrix to avoid download and
conversion I/O perturbing timings.

The public 1DSfM numerical archive is available as a 682,117,483-byte tarball.
Each dataset includes a Bundler `bundle.out` reconstruction. Bundler uses the
same camera model required by our BAL solver: focal length, two radial
distortion coefficients, world-to-camera rotation/translation, 3D points, and
centered pixel observations. The vendored DABA source also contains
`BundlerDatasetToBALDataset`, confirming the intended conversion path.

Dataset identity is a hard gate. Converted files must match DABA Table II:

| Dataset | Cameras | Points | Observations | DABA trivial-loss Init |
|---|---:|---:|---:|---:|
| Gendarmenmarkt | 706 | 93,672 | 364,029 | 8.181 |
| Piccadilly | 2,289 | 209,504 | 999,878 | 11.26 |
| Roman Forum | 1,063 | 265,047 | 1,292,756 | 6.407 |
| Trafalgar | 5,032 | 388,956 | 1,826,071 | 10.77 |
| Union Square | 796 | 46,066 | 230,811 | 10.56 |
| Vienna Cathedral | 836 | 265,553 | 1,333,280 | 9.506 |

The 1DSfM documentation warns that its supplied `bundle.out` can contain a few
extra or missing cameras relative to the connected component. Counts and the
common evaluator's initial trivial-loss error must match the published values.
If either differs, report it as a distinct 1DSfM reconstruction and do not place
it in an exact DABA Table I/III/IV comparison until the authors' preprocessed BA
files, metric convention, or filtering rule are recovered.

Once identity is verified, run the same unchanged L2 matrix at
`K = 4, 8, 16, 32`, 1000 iterations, with the same accuracy, efficiency, RSS,
and serialized-communication fields as the BAL matrix. Huber remains a later
second pass.

## 2026-07-26: Ladybug-1723 K=8 numerical failure

The first 1000-iteration `K=8` attempt failed at outer iteration 77. It was not
an OOM or worker crash: coordinator peak RSS was 1,418,468 KiB and worker peak
RSS was 601,244 KiB. One cluster's candidate cost grew from a best global SSE
near `1.391e6` to `5.99e12`, then `1.79e30`; the following DRE became NaN and a
diagnostic `round(NaN)` raised `ValueError`.

Root cause: the safeguard required the block metric to be below its upper bound
before rejecting a bad trial. Once all metrics reached `0.5`, rejection was
disabled. Rejected trials also replaced the finite DRE reference with the bad
trial value, weakening the next safeguard. The matrix was stopped before later
rows mixed this known defect with valid runs.

Fix: final non-finite trials are always rejected; finite trials that violate
both DRE and primal-cost bounds are rejected even at the metric ceiling; a
rejection retains the finite best-cost reference. This changes only safeguard
failure handling, not the L2 objective, partitioning, or local solve.

## 2026-07-26: Ladybug-1723 L2 result, K=4

Status: completed 1000 outer iterations. The independently evaluated final SSE
matches native worker SSE with relative difference `2.81e-8`.

### Accuracy values with different metric conventions

| Method | Devices/partitions | Reported value | Metric/source |
|---|---:|---:|---|
| DR | 4 GPUs | 0.837 | DABA weighted ray cost / observations |
| ADMM | 4 GPUs | 0.698 | DABA weighted ray cost / observations |
| DABA | 4 GPUs | 0.690 | DABA weighted ray cost / observations |
| Ceres | 1 device | 0.707 | DABA weighted ray cost / observations |
| DeepLM | 1 GPU | 0.710 | DABA weighted ray cost / observations |
| Variable-metric DRS | 4 logical CPU partitions | 0.758727 | mean Euclidean pixel error |

These values must not be subtracted or ranked directly. DABA's source confirms
that its Table I value is Ceres cost divided by observation count for a weighted
3D ray residual, despite the table's “mean reprojection error” label.

### Local efficiency and resources

| Metric | Value | Scope |
|---|---:|---|
| Overall time | 567.552 s | setup-inclusive local wall clock |
| Partition time | 0.726 s | included in overall time |
| Time excluding partition | 566.826 s | local CPU only |
| Coordinator peak RSS | 1.370 GiB | Python/Torch/SciPy coordinator process |
| Worker peak RSS | 0.571 GiB | C++ Ceres worker process |
| Sum of process peak RSS | 1.941 GiB | conservative sum, peaks not synchronized |
| Serialized payload | 10.289 MiB/outer iteration | both ZeroMQ directions, single host |
| Total outer-loop payload | 10.048 GiB | 1000 outer iterations |

Runtime is not divided by DABA's published 4-GPU time because hardware and
execution models differ. Memory is likewise not compared numerically with
DABA Figure 4 until its modeled per-device values are available in the same
scope. The payload measurement is directly useful for a future communication
model, but currently includes coordinator/worker protocol traffic on one host.

Artifacts:

- `daba_protocol_l2/results.jsonl`
- `daba_protocol_l2/states/1723_k4_i1000.npz`
- `daba_protocol_l2/logs/1723_k4_i1000.log`
- `daba_protocol_l2/memory/1723_k4_i1000_coordinator.time`
- `daba_protocol_l2/memory/1723_k4_i1000_worker.time`

## 2026-07-26: Ladybug-1723 L2 result, K=8 retry

Status: completed 1000 outer iterations after the safeguard repair. The common
evaluator and native worker SSE agree to `3.91e-8` relative difference.

| Method | Devices/partitions | Reported value | Metric |
|---|---:|---:|---|
| DR | 8 GPUs | 0.846 | DABA weighted ray cost / observations |
| ADMM | 8 GPUs | 0.703 | DABA weighted ray cost / observations |
| DABA | 8 GPUs | 0.690 | DABA weighted ray cost / observations |
| Variable-metric DRS | 8 logical CPU partitions | 0.826987 | mean Euclidean pixel error |

The cross-metric values are contextual, not a ranking. DRS's best state occurred at outer
iteration 20 with SSE `1,017,369.126`; subsequent trials were rejected after
the metric reached its ceiling. Thus the safeguard prevented numerical failure
but did not restore useful convergence for this configuration.

| Metric | Value | Scope |
|---|---:|---|
| Overall time | 526.468 s | setup-inclusive local wall clock |
| Partition time | 1.395 s | included in overall time |
| Coordinator peak RSS | 1.366 GiB | Python/Torch/SciPy coordinator process |
| Worker peak RSS | 0.574 GiB | C++ Ceres worker process |
| Sum of process peak RSS | 1.940 GiB | conservative sum, peaks not synchronized |
| Serialized payload | 12.760 MiB/outer iteration | both ZeroMQ directions, single host |

The K8 result is worse than K4 in both quality and communication, with no
material memory reduction in the current single-worker implementation.

## 2026-07-26: Historical primary configuration, K=30 stable/90

The known configuration from `serverTest/results_server.json` was reproduced:
`K=30`, 90 iterations, `landmark_scalable_stable`, legacy worker-internal
landmark cost semantics, and the legacy metric-ceiling safeguard.

| Checkpoint | Historical | Reproduction | Relative difference |
|---|---:|---:|---:|
| Best native cost | 761,563 | 761,294 | -0.035% |
| Best through iteration 30 | 763,339 | 763,152 | -0.024% |
| Best through iteration 60 | 762,101 | 761,959 | -0.019% |

This confirms the historical solver behavior. The initial attempt to export
the state associated costs with mutable worker landmark buffers and was wrong.
Cost replies and best-state notifications now carry the exact landmark snapshot
used for each cost. After this atomic association fix, native cost and E0 SSE
agree to `2.53e-9` relative error:

| Global-state metric | Value |
|---|---:|
| Mean reprojection error | 0.758406 px |
| Median reprojection error | 0.535518 px |
| 90th percentile | 1.661666 px |
| 95th percentile | 2.222348 px |
| Global-state SSE | 761,437.918 |
| Native worker-state cost | 761,437.916 |
| RMSE per scalar residual | 0.748958 px |
| Maximum reprojection error | 22.233 px |

`bestCost`, `bestCost30`, and `bestCost60` are the same BA SSE objective. The
suffixes only denote the best value observed through 30 or 60 iterations; they
are not separate metrics.

### Exact DABA Ceres baseline reproduction

The CPU-only `daba_ceres_bal_runner` preserves DABA's camera manifold, weighted
3D ray residual, Huber scale, and 40-iteration iterative-Schur configuration.
On Ladybug-1723 with trivial loss:

| Quantity | Local reproduction | DABA Table I |
|---|---:|---:|
| Initial reported metric | 10.479275 | 10.48 |
| Final reported metric | 0.707147 | 0.707 |
| Solve time | 12.186 s | unlike hardware; contextual only |
| Peak RSS | 0.825 GiB | not reported in the table |

This proves that DABA Table I reports weighted ray Ceres cost divided by the
number of observations, not mean Euclidean pixel distance.

### Fair same-objective Ceres comparison

Standard Ceres and DRS were run on the identical canonical Snavely 2D pixel
objective. Ceres used 40 iterations, iterative Schur, Schur-Jacobi, and 32
threads; DRS used K30 and 90 outer iterations.

| Method | SSE | Mean pixel error | Time |
|---|---:|---:|---:|
| Variable-metric DRS K30 | 761,437.918 | 0.758406 px | 66.372 s overall |
| Standard Ceres | 770,093.439 | 0.762970 px | 6.790 s solve |

DRS reaches `1.12%` lower SSE and `0.004564 px` lower mean error, while Ceres
is about `9.8x` faster by the displayed solve/overall times. Setup-adjusted and
repeated-run timing remains to be reported before making a formal speed ratio.

### Local resources

| Metric | Value |
|---|---:|
| Overall time | 66.372 s |
| Partition time | 3.551 s |
| Coordinator peak RSS | 1.357 GiB |
| Worker peak RSS | 0.590 GiB |
| Sum of process peak RSS | 1.947 GiB |
| Instrumented serialized payload | 17.633 MiB/outer iteration |

The payload includes exact landmark snapshots added to validate cost/state
identity. It is not the base algorithm's communication load and must be
separated before a Figure 4-style communication comparison.

K30 and K4 have nearly identical common mean errors (`0.758406` versus
`0.758727 px`), while K30 reaches the slightly better SSE in 90 rather than
1000 iterations.

Artifacts:

- `ours_k30_atomic/results.jsonl`
- `ours_k30_atomic/states/1723_k30_i90.npz`
- `ours_k30_atomic/logs/1723_k30_i90.log`
- `daba_ceres/ladybug1723_t64.json`
- `ceres_ladybug1723/ceres_t32.json`
- `ceres_ladybug1723/ceres_t32.metrics.json`

## 2026-07-26: Ladybug-1723 objective-family comparison

DABA's released Ceres source was reproduced in the CPU-only
`daba_ceres_bal_runner`. Its Table I metric is weighted 3D-ray Ceres cost per
observation, not mean Euclidean pixel error. The implementation reproduces the
paper exactly enough to identify the convention:

| DABA ray metric | Initial | Final | Solve time |
|---|---:|---:|---:|
| DABA Ceres paper | 10.48 | 0.707 | unlike hardware |
| DABA Ceres local exact source | 10.479275 | 0.707147 | 12.186 s |

To score standard BA geometry in this objective, observation normalization was
held fixed from the original BAL input and the three DABA intrinsics were
minimized independently per camera while poses and points remained fixed:

| Fixed geometry | Best DABA ray metric |
|---|---:|
| Variable-metric DRS K30 | 3.413079 |
| Standard Snavely Ceres | 3.591498 |
| DABA Ceres, jointly optimized ray formulation | 0.707147 |

Both standard BA geometries are poor under DABA's different ray formulation;
DRS is slightly better than standard Ceres within this post-hoc fixed-geometry
test. This does not alter their Snavely pixel-objective comparison:

| Standard Snavely objective | SSE | Mean pixel error |
|---|---:|---:|
| Variable-metric DRS K30/90 | 761,437.918 | 0.758406 px |
| Ceres iterative Schur, 40 iterations/32 threads | 770,093.439 | 0.762970 px |

Thus DRS reaches slightly better standard BAL quality, while DABA Ceres reaches
much better DABA-ray quality. Cross-objective scalar comparisons are invalid.

The reverse evaluation inverts DABA's optimized radial ray camera and measures
standard pixel residuals:

| DABA-Ceres state in pixels | Value |
|---|---:|
| Mean error | 0.775256 px |
| Median error | 0.545465 px |
| 90th / 95th percentile | 1.694450 / 2.262608 px |
| RMSE | 1.099343 px |
| Maximum | 49.326969 px |
| Observations with no real inverse projection | 10 / 678,718 |

This is worse than standard Ceres (`0.762970 px`) and DRS (`0.758406 px`) on
the standard objective. It demonstrates measurable metric misalignment and a
small invertibility issue, but one dataset and the Ceres baseline are not enough
to rule out DABA. The actual optimized DABA state must be exported and evaluated
on multiple BAL/1DSfM datasets before making that claim.

Artifact: `ladybug1723_objective_comparison.json`.

## Queued: standard-pixel ADMM baseline

DABA includes a complete CUDA/MPI ADMM implementation, but it optimizes the
DABA weighted-ray objective. Source audit:

| Feature | DABA ADMM |
|---|---|
| Local nonlinear solver | LM, up to 20 inner iterations / 15 accepted |
| Linear solver | Schur PCG, up to 400 iterations, relative reduction 0.1 |
| Preconditioner | block inverse of reduced camera Hessian |
| Consensus variables | camera extrinsics and intrinsics; points are owned |
| Over-relaxation | fixed factor 1.5 in the dual update |
| Penalty adaptation | primal/dual residual balancing; factors 1.5 and 0.8 |
| Communication | three rounds per outer iteration; camera data only |
| Nesterov/Anderson | absent |
| Adaptive restart | absent |
| Full variable consensus metric | absent; scalar extrinsic/intrinsic penalties |
| Coordinate equilibration/Ruiz scaling | absent beyond ray reparameterization |

A fair standard-metric reproduction can reuse the current CPU Ceres worker and
partitioning. Add a coordinator baseline mode with:

1. local Snavely L2 solve plus scalar camera consensus penalty;
2. Euclidean consensus of duplicated cameras;
3. dual update with over-relaxation `alpha = 1.5`;
4. DABA's initial penalty `2.5 * observations / cameras` and residual-balanced
  `1.5 / 0.8` adaptation;
5. the same atomic state snapshots, E0 pixel metrics, timing, RSS, and payload
  accounting as DRS.

Run this published-like ADMM first. Only then add a strong-ADMM ablation with
our coordinate scaling and full 9x9 block metric. Acceleration should be a
separate ablation because it is not present in DABA's ADMM baseline.

The complete implementation and ablation order is in
`../admm_drs_baseline_plan.md`.

## 2026-07-26: Parallel CPU ADMM Stage A implementation

Implemented a separate standard-pixel ADMM baseline:

- coordinator: `serverTest/client_admm.py`;
- tested consensus/dual/penalty kernel: `serverTest/admm_consensus.py`;
- parallel cluster requests through the existing ZeroMQ worker;
- explicit scalar `rho * I` camera prior in the worker protocol;
- DABA arithmetic camera reference followed by dual over-relaxation `1.5`;
- DABA initial penalty and residual-balanced `1.5 / 0.8` adaptation;
- selectable `ceres_pcg` and `nesterov` local solvers;
- atomic state export, E0 metrics, trajectory, and transport bytes;
- configurable ports for isolated workers.

The original user-owned DRS server remained on ports 5556/5557. ADMM tests ran
with an isolated build and worker on 6556/6557.

### Validation

- Four NumPy reference tests cover occurrence masks, arithmetic consensus,
  over-relaxed dual update, penalty adaptation, and K1 consensus.
- ASan found and fixed an uninitialized scalar-prior block buffer.
- K1 has zero primal consensus residual to numerical precision.
- K2 parallel requests and state export complete under release and ASan builds.

### BAL-49 K2 baseline

Configuration: 100 outer iterations, 20 Ceres LM iterations per local solve,
iterative Schur, Schur-Jacobi, `alpha=1.5`, standard Snavely L2.

| Method | SSE | Mean pixel error | Time |
|---|---:|---:|---:|
| Centralized Ceres, 40 iterations | 26,688.53 | 0.57962 px | 0.76 s solve |
| CPU ADMM K2 | 27,602.67 | 0.58545 px | 5.43 s overall |

ADMM's best occurs at outer iteration 97. The final primal and dual residual
squares are `2.46e-24` and `1.85e-25`. The published penalty rule increases the
scalar penalty to `1.58e20`, because its neutral branch multiplies by `1.005`
per update and residual balancing repeatedly selects increases. Preserve this
behavior for the faithful baseline, but include fixed/capped/neutral-1.0 penalty
ablations before calling it robust.

Current parameterization caveat: standard Snavely stores all nine camera
parameters together, so Stage A applies one scalar penalty to pose and
intrinsics. DABA uses separate extrinsic and intrinsic penalties in its ray
camera. A two-penalty standard-camera extension is an ablation, not required to
establish the first CPU baseline.

### Preliminary local-solver substitution

Replacing Ceres-PCG with the current custom Nesterov local solver under the same
ADMM settings produced non-finite states in the short matched test. Before
moving this into Stage C4, local Nesterov iterations were moved inside one
worker request so they no longer multiply communication rounds. The instability
supports the planned damping, coordinate-scaling, spectral-bound, and safeguard
ablations; it is not yet evidence of a Nesterov advantage.

Artifacts:

- `admm_bal49_k2/results.jsonl`
- `admm_bal49_k2/state.npz`
- `admm_smoke/results.jsonl`
- `admm_k1_smoke/results.jsonl`

## 2026-07-26: ADMM innovation matrix launch

Dataset subset:

- Ladybug: 1723, 1266, 1064, 931;
- Venice: 52, 245, 744;
- Final: 394, 871, 3068.

Cluster counts: `K = 5, 15, 30`. Baseline protocol: 100 outer iterations,
20 local Ceres LM iterations, iterative Schur/Schur-Jacobi, DABA arithmetic
consensus, dual over-relaxation `1.5`, and DABA adaptive scalar penalty.

The resumable runner is `serverTest/run_admm_innovation_matrix.sh`; variant
readiness and exact arguments are in `admm_innovation_manifest.json`. Results
are paired and ranked by `analyze_admm_innovation_matrix.py`.

Execution order:

1. complete all 30 baseline rows;
2. run each ready control/innovation alone on the same 30 pairs;
3. exclude any variant that fails to reach the baseline's final SSE on a paired
  case from the greedy ranking;
4. order eligible innovations by paired final SSE and time-to-baseline-target;
5. add the largest-gain innovation to the baseline, then evaluate only the next
  candidate additions; do not rerun the already measured winning prefix alone;
6. preserve failed/non-finite rows as failures rather than dropping them.

Current ready modes: baseline, no-over-relaxation control, fixed-penalty
control, capped-penalty control, and experimental Nesterov local solve. Plain
DRS, fast DRS, coordinate scaling, and full block metric remain explicitly
pending and will enter the same harness as they are implemented.
# Experiment Log

## 2026-07-26: E0 common-evaluator harness validation

Purpose: validate identical BAL initialization, final-state export, common
quality metrics, and time-to-target machinery. This is not a paper performance
result.

Dataset: Ladybug-49 (`problem-49-7776-pre.txt`), 49 cameras, 7,776 points,
31,843 observations. Both methods used the exact DRS canonicalization of the
same BAL initialization and the trivial/L2 objective.

| Method | Initial SSE | Final SSE | Mean reprojection error px | Time to DRS final SSE s | Configuration |
|---|---:|---:|---:|---:|---|
| Variable-metric DRS | 1,701,824.921 | 27,081.306 | 0.5866 | 2.692 | 20 outer iterations, 30 clusters |
| Ceres iterative Schur | 1,701,824.921 | 26,688.552 | 0.5796 | 0.096 | first crossing at iteration 3, 16 threads |

Validation:

- DRS common-evaluator SSE versus native worker SSE relative difference:
  `8.4e-8`.
- Ceres and DRS initial SSE agree to floating-point precision.
- The run rejects Ladybug-49 as evidence for a DRS speed advantage under this
  configuration.
- No repeated-run statistics, converged-method comparison, comparable memory
  accounting, cloud cost, or distributed scaling claim is made.

Artifacts:

- `ladybug49_canonical.txt`
- `ladybug49_ceres_canonical_i40_t16.json`
- `ladybug49_ceres_canonical_i40_t16.metrics.json`
- `ladybug49_drs_matched_trajectory_v2.jsonl`
- `states/ladybug49_ceres_canonical_i40_t16.txt`

## 2026-07-26: DABA-protocol replication decision

Goal: reproduce the evaluation design of DABA Tables I, III, and IV and Figure
4 for variable-metric DRS, without changing the current CPU implementation to
CUDA.

Protocol:

- Table I analogue: mean reprojection error after 1,000 outer iterations for
  `K = 4, 8, 16, 32`, separately for trivial and Huber losses.
- Table III analogue: count datasets on which DRS beats each centralized
  reference. Report available BAL subsets separately until all 20 BAL/1DSfM
  datasets are available; do not label a subset count as `All` or `Largest`.
- Table IV analogue: for each centralized reference, define `F_ref` as its best
  objective in 40 iterations and
  `F_delta = F_ref + 2.5e-4 * (F_init - F_ref)`. The centralized time is its
  first crossing of `F_delta`; DRS time is its first crossing of the stricter
  `F_ref`. Report time and speed ratio only within a shared hardware class.
- Figure 3 analogue: performance profiles by outer iteration at
  `delta = 1e-4`.
- Figure 4 analogue: maximum memory per partition/device and total
  communication payload per outer iteration for `K = 1, 2, 4, 8, 16, 32`.
  Also report aggregate memory, coordinator memory, message count, and
  synchronization rounds for DRS.

Current executable scope:

- Trivial/L2 DRS rows can run on the current CPU implementation.
- Huber is not yet an optimization option: the Ceres worker currently adds
  residual blocks with a null squared loss. Evaluating an L2 solution with a
  Huber metric is not a substitute.
- The official DABA repository is retained as published evidence and future
  same-GPU baseline. Its released paper executables require CUDA, NCCL, and
  MPI. No CUDA port of DRS is part of this experiment phase.
- Running 4/8/16/32 DABA MPI ranks on one RTX 4080 would map all ranks to the
  same physical GPU and is not a multi-device reproduction.

Cost reporting:

- East US Linux pay-as-you-go reference: Azure `Standard_F16s_v2` is
  `$0.677/VM-hour`; `Standard_NV36ads_A10_v5` is `$3.200/VM-hour`.
- The listed VM-price ratio is `4.73`, not an algorithmic speed or cost factor.
- Dollar cost is reported only for experiments actually run on the named SKU:
  `wall_seconds / 3600 * VM_price`, summed over nodes.

## 2026-07-26: CPU ADMM matrix and plain-DRS Stage B

The resumable CPU ADMM baseline matrix uses the ten selected BAL datasets,
`K = 5, 15, 30`, 100 outer iterations, and 20 Ceres local iterations. Each
case starts a fresh isolated worker on ports 6556/6557 and records exact final
state, trajectory, elapsed time, coordinator/worker RSS, and transport bytes.
The user-owned DRS server on ports 5556/5557 remains untouched.

Interim evidence after 15 of 30 baseline rows:

- All completed adaptive-penalty trajectories increase the initial scalar
  penalty by approximately `1.45e17` to `2.17e17`, ending between `2.14e20`
  and `3.62e21`.
- Ladybug-1723 at K15 and K30 never improves the initial common-evaluator SSE;
  K5 reaches mean reprojection error `0.787146` px.
- These observations motivate a later capped-penalty sensitivity control.
  They do not change the primary innovation ranking order.

Plain scalar-metric product-space DRS is implemented in the shared CPU
coordinator with the same partitioning, Ceres proximal worker, initial scalar
penalty, and common evaluator as ADMM. Its outer update is

`x = prox_f(z); y = P_C(2x - z); z_next = z + y - x`.

The exact reflected-projection algebra has a unit test. A two-iteration
BAL-52/K5 integration smoke on isolated ports 6656/6657 completed with a
finite trajectory and improved SSE from `22,304,125.55` to `12,848,961.86`.
This smoke validates execution only and is not a performance result.

Safeguarded Fast DRS remains pending. The existing implementation evaluates
plain and accelerated proximal trials and can restore the complete local
camera/landmark state after rejection. The simplified shared coordinator's
worker currently mutates local landmarks without a trial rollback boundary;
adding inertia without that boundary would not reproduce the existing
Themelis safeguard.

The matrix analyzer now includes `initialQualityMetrics` as a valid time-zero
target crossing. This fixes initial-best rows being incorrectly reported as
unsolved; time-zero pairs count as solved but are excluded from speedup ratios
because their ratio is undefined.

## 2026-07-26: revised 60-iteration one-core protocol

The active 100-iteration baseline is allowed to finish unchanged. Subsequent
paired comparisons use a separate matrix directory with:

- 60 outer iterations;
- 20 local iterations;
- `K = 10, 30` for the final baseline and all innovation comparisons;
- exactly one Ceres/Eigen thread per cluster solve;
- a fresh worker and exact state export for every case.

The worker now accepts `BUNDLE_PALM_THREADS_PER_CLUSTER`; the runner sets it to
one, passes the declared value to the coordinator, includes it in artifact and
completion identity, and records it in every result/status row. Legacy runs
retain the previous computed default when the variable is absent.

A fresh 60-iteration, one-core baseline must be measured before the isolated
variants. The existing 100-iteration baseline used `max(1, 31 / K)` Ceres
threads per cluster and therefore is not a valid timing reference for the new
resource condition.

An initial one-core baseline run produced partial K5/K15/K30 rows. To reduce
experiment time, the final comparison grid was changed to K10/K30. Existing
K30 rows remain valid and resumable; K5/K15 rows are retained as exploratory
evidence but excluded from innovation pairing and ranking.

The primary experiment order remains the agreed innovation program:

1. safeguarded Nesterov/Fast DRS outer acceleration;
2. coordinate preconditioning/equilibration (Jacobi, then Ruiz);
3. variable/full 9x9 block consensus metric;
4. replace Schur-PCG with a stabilized Nesterov or Chebyshev local solver;
5. adaptive/inexact local solve work.

Each innovation is first measured alone against the same ADMM baseline. The
eligible innovations are then ordered by paired gain and added greedily to the
winning prefix. Capped penalty, no over-relaxation, fixed penalty, and plain
DRS are separate sensitivity/formulation controls, not innovations in this
ranking. They remain pending until the innovation matrix is complete.

## 2026-07-26: BAL-52 objective provenance

The apparent discrepancy between historical DRS `bestCost30 = 485,348` and
the one-core ADMM baseline `best pixel SSE <=30 = 2,469,787` is not a metric
change. Both use the standard BAL/Snavely pixel residual and report

`sum_observations ((predicted_x - observed_x)^2 +
                   (predicted_y - observed_y)^2)`.

The worker adds these residual blocks with a null squared loss. Ceres
internally reports half-SSE; worker-native DRS cost and the common evaluator
report twice that value. ADMM local subproblems add the required scalar
consensus proximal term, while the reported global objective is independently
recomputed pixel SSE and excludes the proximal term. DABA's weighted 3D-ray
objective is not used in this matrix.

The historical BAL-52 run did not save a state, so its native value cannot be
retrospectively re-evaluated independently. Its source residual and reporting
formula are nevertheless the same pixel-SSE convention, and repeated
historical K30 runs cluster near `485,000` at iteration 30. The current ADMM
gap therefore reflects much weaker baseline convergence, including very large
consensus-state excursions before its best iteration, rather than objective
or reporting drift.

The CPU baseline must not be labeled a DABA reproduction. Official DABA uses
separate extrinsic and intrinsic consensus variables, duals, and independently
adapted penalties, together with its own camera representation and weighted-ray
objective. The CPU baseline uses one adaptive scalar penalty over all nine raw
Snavely camera coordinates while optimizing standard pixel residuals. It only
borrows DABA's initial-penalty formula, adaptation ratios, and alpha=1.5 dual
update. On BAL-52/K30 its penalty grows from about `1.34e4` initially to
`1.71e9` at iteration 30 and `3.27e14` at iteration 60, after which progress
effectively freezes.

The previously observed closeness between DABA-Ceres and mature DRS refers to
their final states cross-evaluated in standard pixels (about `0.7753` versus
`0.7584` mean px on Ladybug-1723). It does not validate the convergence of this
plain CPU ADMM translation. DABA paper values such as `0.707` are weighted-ray
cost per observation and are not numerically comparable to pixel error.

## 2026-07-26: Jacobi variable-substitution ablation

The first isolated innovation is an initial camera Jacobi variable
substitution computed from the standard pixel reprojection Jacobian:

`D = sqrt(diag(J_camera^T J_camera))`, normalized to geometric mean one.

All ADMM camera quantities are stored consistently in `y = D x` coordinates:
local copies, consensus, centers, and scaled duals. The worker reconstructs
physical Snavely cameras using `x = D^-1 y` inside the unchanged pixel residual.
Only physical cameras are passed to the independent evaluator and state export.
Thus a scalar Euclidean ADMM penalty in scaled coordinates induces the physical
metric `rho D^T D` without changing the data objective.

Focused validation:

- finite-difference pixel-Jacobian test and transform round-trip passed;
- BAL-52/K30, two outer iterations and two local Ceres iterations completed;
- scaling range: `0.0025657` to `52.3173`;
- initial pixel SSE: `22,304,125.55`;
- best two-iteration pixel SSE: `2,935,575.87`;
- saved-state independent reevaluation relative error: exactly zero;
- 12 focused Jacobi, consensus, and report tests passed.

The measurement matrix uses all ten selected datasets at K30 only, 60 outer
iterations, 20 local Ceres iterations, and one thread per cluster. Variant
identity is `jacobi_scaling`; no other ADMM behavior changes.

Later DABA-style stages remain separate:

1. split 6+3 extrinsic/intrinsic consensus and dual bookkeeping under one
  shared penalty (an algebraic identity/control, not a full matrix run);
2. separate extrinsic/intrinsic penalties and scaled-dual rescaling;
3. independently adapt those penalties from block-specific residuals.

The Jacobi matrix was stopped after seven completed scenes and reduced to the
five-scene diagnostic set `1723, 52, 245, 394, 871`. Future isolated ablations
use these five scenes at K30 unless a result justifies expanding the matrix.

Jacobi trajectory diagnosis on the five-scene set:

- BAL-52 and BAL-245 improve smoothly through iteration 60; final/best pixel
  SSE ratio is `1.00`.
- Ladybug-1723 reaches its best pixel SSE at iteration 3, then diverges; its
  iteration-60 current/best ratio is approximately `5.82e7`.
- Final-394 reaches its best at iteration 3 and ends `3.20x` worse.
- Final-871 experiences large transient explosions, recovers, reaches its best
  at iteration 40, and ends `1.07x` worse.

Jacobi improves the best pixel SSE on four of five focused scenes but does not
fix the unstable outer penalty dynamics. Extending these runs or adding more
scenes is lower priority than the separate extrinsic/intrinsic penalty and
independent-adaptation ablation.

## 2026-07-26: K20 stability diagnosis

The focused protocol was reduced to K20 and 30 outer iterations. Short
two-scene diagnostics on Ladybug-1723 and Venice-52 established:

- 20 local Ceres LM iterations per ADMM step cause persistent divergence on
  1723; one local iteration reaches `3.57e6` by outer iteration 11 and ends at
  its best while taking about one quarter of the time.
- A global Jacobi range cap (`Dmax/Dmin = 1e3` through `1e12`) and
  per-parameter 1st/99th percentile clipping suppress useful weak directions
  and make 1723 quality/stability worse.
- Removing dual over-relaxation (`alpha = 1`) modestly reduces excursions and
  preserves recovery.
- Increasing the transformed-coordinate initial penalty stabilizes the first
  step on 1723, but the same multiplier severely degrades BAL-52; no robust
  single multiplier was found.
- A DABA-style 6+3 split with equal fixed penalties reproduces the combined
  implementation bit-for-bit. Independent extrinsic/intrinsic adaptation also
  remains identical over 12 iterations because both groups select the same
  penalty multiplier at every step.

The selected candidate is therefore unbounded initial Jacobi substitution,
one local LM iteration, `alpha = 1`, and the existing adaptive penalty. At K20
and 30 outer iterations:

- 1723 best pixel SSE `2,805,429` at iteration 27; endpoint/best `1.0015`;
- 52 best pixel SSE `1,840,911` at iteration 29; endpoint/best `1.0`.

Transient 1723 excursions remain, so this is a recovery-stable candidate, not
yet a monotone or safeguarded method. The next five-scene run uses this
configuration and one local LM step.

The completed five-scene K20 comparison confirms that candidate:

| Dataset | Baseline best SSE | Candidate best SSE | Candidate / baseline | Candidate best iteration | Endpoint / best |
|---|---:|---:|---:|---:|---:|
| 1723 | 124,050,155 | 2,805,429 | 0.023 | 27 | 1.002 |
| 52 | 2,111,382 | 1,840,911 | 0.872 | 29 | 1.000 |
| 245 | 3,390,104 | 3,079,341 | 0.908 | 29 | 1.000 |
| 394 | 2,695,707 | 720,582 | 0.267 | 28 | 1.000 |
| 871 | 4,703,812 | 4,933,865 | 1.049 | 24 | 1.021 |

The candidate improves four of five scenes and ends within 2.1% of its best
state on every scene. It is 1.0x to 1.76x faster than the matched one-local-step
raw baseline. Large transient current-state excursions remain (especially on
1723), so a safeguard or accepted-state policy is still required before this
can be called nondivergent.

Additional diagnostics found:

- global and percentile Jacobi clipping damages 1723 and does not provide a
  useful stability/quality compromise;
- a larger initial penalty stabilizes 1723 but severely degrades BAL-52;
- the DABA 6+3 fixed equal-penalty split is bit-for-bit identical to the
  combined representation;
- independent extrinsic/intrinsic adaptation remains identical to combined
  adaptation in the tested 12 iterations because both groups choose the same
  multiplier every time.

## 2026-07-26: staged K20 five-scene ADMM gate

After restoring DRS state/trajectory provenance, the ADMM innovation sequence
was repeated on five scenes at K20, 30 outer iterations, and one local LM step.
All reported quality values remain independently evaluated standard pixel SSE.

Stages:

1. raw-coordinate combined-penalty ADMM baseline;
2. unbounded initial Jacobi substitution with `alpha = 1`;
3. split 6+3 extrinsic/intrinsic penalty storage with one shared adaptation
  multiplier (identity control);
4. independently adapted extrinsic/intrinsic penalties and scaled duals.

The shared-adaptation 6+3 representation reproduced the combined trajectory,
residuals, penalties, cameras, and landmarks bit-for-bit. Independent
adaptation produced final `rho_e / rho_i` ratios of `1.0` (1723), `1.0` (245),
`1.49` (394), `0.202` (52), and `11.1` (871), but changed best pixel SSE by at
most `0.02%` relative to the Jacobi stage.

| Dataset | Raw baseline best SSE | Jacobi best SSE | Split-adapt best SSE | Split endpoint / best |
|---|---:|---:|---:|---:|
| 1723 | 124,050,155 | 2,805,429 | 2,805,429 | 1.002 |
| 52 | 2,111,382 | 1,840,911 | 1,840,502 | 1.000 |
| 245 | 3,390,104 | 3,079,341 | 3,079,341 | 1.000 |
| 394 | 2,695,707 | 720,582 | 720,580 | 1.000 |
| 871 | 4,703,812 | 4,933,865 | 4,933,865 | 1.010 |

The no-explosion gate failed. Ladybug-1723 still has a maximum/current-state
excursion approximately `4.6e18` times its best SSE before recovery. Separate
extrinsic/intrinsic adaptation does not address this. The experiment therefore
remains at five scenes; it must not be expanded to ten until an outer accepted-
state safeguard or another mechanism removes catastrophic transient iterates.

Readable report: `benchmark_results/admm_five_scene_i30_k20/report.md`.

## 2026-07-26: local solver x trust-region matrix

Protocol: five scenes (`1723, 52, 245, 394, 871`), K20, 30 ADMM outer
iterations, one local nonlinear iteration, one CPU thread per cluster,
unbounded initial Jacobi camera substitution, `alpha = 1`, adaptive scalar
penalty, and independent standard pixel-SSE evaluation.

A shared custom local path was implemented so the same damped normal equations,
candidate evaluation, and proximal objective can switch independently between:

- block-Jacobi Schur-PCG (maximum 400 iterations, relative residual `1e-2`);
- existing Schur-Nesterov;
- DABA trust-region acceptance/radius update;
- existing DRS trust-region retry/radius update.

Ceres iterative Schur with Ceres internal LM remains a contextual fifth row.
All 25 cases completed. Independent reevaluation of all 25 saved physical
states reproduced recorded pixel SSE exactly (maximum relative error zero).

| Variant | Geomean best SSE | Endpoint stable (<=1.05x best) | Geomean time s | Maximum excursion / best |
|---|---:|---:|---:|---:|
| Ceres Schur-PCG + Ceres LM | 2,241,156 | 5/5 | 19.02 | 4.60e18 |
| Nesterov + DABA TR | 3,110,876 | 4/5 | 13.95 | 2.33e24 |
| PCG + DRS TR | 3,376,614 | 4/5 | 13.49 | 2.85e18 |
| Nesterov + DRS TR | 3,935,580 | 4/5 | 14.10 | 2.00e16 |
| PCG + DABA TR | 4,729,922 | 4/5 | 14.22 | 7.74e11 |

On scenes 52, 245, 394, and 871, custom variants are competitive and generally
faster than Ceres. On 1723 every custom combination fails:

| Variant | 1723 best SSE | Best iteration | Endpoint / best |
|---|---:|---:|---:|
| Ceres Schur-PCG + Ceres LM | 2,805,429 | 27 | 1.00 |
| Nesterov + DABA TR | 16,667,483 | 1 | 2.21e12 |
| PCG + DRS TR | 22,158,530 | 1 | 5.63e4 |
| Nesterov + DRS TR | 52,281,868 | 0 | 1.27e6 |
| PCG + DABA TR | 124,050,155 | -1 | 16.08 |

DABA TR is not a general fix, although it improves Nesterov relative to the
DRS trust policy on 1723. Ceres is the only fully endpoint-stable path in this
matrix and has the best aggregate quality. The next action is to inspect the
custom quadratic-step/damping mismatch on 1723, not to expand to ten scenes.

Readable report: `benchmark_results/admm_solver_trust_i30_k20/report.md`.

## 2026-07-26: catastrophic-only outer recovery

The first outer safeguard attempts rejected candidates above `1.1x` or `2x`
the last accepted global pixel SSE. Both were invalid for ADMM: on 1723 the
first candidate rises from `1.24e8` to `2.71e11`, but the next iteration reaches
`1.67e7`. The strict guards rejected this useful transient, restarted all 12
iterations, doubled runtime, and never improved over initialization.

The replacement guard treats recovery as numerical-failure handling rather
than monotonic acceptance. It rejects only nonfinite candidates or one-step
pixel-SSE growth above `1e6`, restores local cameras, landmarks, consensus,
scaled duals, and persistent worker trust-region state, then doubles the ADMM
penalty. The standard Snavely pixel objective and one-local-step protocol are
unchanged.

Five-scene protocol: datasets `1723, 52, 245, 394, 871`, K20, 30 outer
iterations, one local nonlinear step, one CPU thread per cluster, Nesterov
Schur solve, DABA trust-region policy, initial Jacobi camera substitution, and
`alpha = 1`.

| Dataset | Unguarded best SSE | Guarded best SSE | Unguarded endpoint | Guarded endpoint | Recoveries |
|---|---:|---:|---:|---:|---:|
| 1723 | 16,667,483 | 16,667,483 | 3.68e19 | 30,842,287 | 1 |
| 52 | 1,791,898 | 1,791,898 | 1,791,898 | 1,791,898 | 0 |
| 245 | 2,906,069 | 2,906,069 | 2,906,069 | 2,906,069 | 0 |
| 394 | 724,074 | 724,074 | 724,074 | 724,074 | 0 |
| 871 | 4,635,978 | 4,635,978 | 4,635,978 | 4,635,978 | 0 |

The guard produced exactly the same best SSE and best iteration as the
unguarded path on every scene. It triggered once on 1723 and zero times on all
other scenes, with runtime unchanged within run noise. This passes the original
five-scene no-catastrophic-endpoint gate without static camera over-damping.

Readable report:
`benchmark_results/admm_catastrophic_guard_five_scene/report.md`.

## 2026-07-26: repeated guard comparison with Ceres diagnostics

The five-scene K20/30 experiment was repeated with unguarded and guarded
Nesterov + DABA TR and unguarded and identically guarded Ceres iterative Schur
+ Ceres LM. Ceres local summary instrumentation recorded accepted and rejected
LM trials without changing normal execution when metrics are disabled.

The earlier claim that Ceres never experiences catastrophic outer excursions
was falsified: unguarded Ceres reaches `1.290e25` pixel SSE on 1723 iteration 0
and later recovers. Its local augmented objective is nevertheless nonincreasing
on every call; 28 of 600 calls reject their sole LM trial. Thus the excursion
is created by global consensus/landmark assembly, not by Ceres accepting a
locally increasing step.

The Nesterov guard again triggered once on 1723, preserved all five best SSEs,
and changed its 1723 endpoint from `3.682e19` to `3.084e7`. The same guard on
Ceres triggered four times on 1723, nowhere else, and improved Ceres's 1723
best SSE from `2,805,429` to `2,290,486`. Neither guard increased aggregate
runtime.

Complete analysis:
`benchmark_results/admm_guard_ceres_comparison/analysis.md`.

## 2026-07-26: global recovery mechanism ablation

Three recovery actions were compared on scenes 1723, 52, 245, 394, and 871 at
K20, 30 outer iterations, one local Nesterov update, DABA trust policy, and one
CPU thread per cluster. All used the same nonfinite or `1e6x` global pixel-SSE
trigger and exact accepted-state rollback.

| Recovery | Geomean best SSE | Geomean endpoint SSE | Recoveries | Total time s |
|---|---:|---:|---:|---:|
| Double ADMM penalty | 3,110,876 | 3,518,342 | 1 | 99.47 |
| Temporary proximal majorizer | 3,110,876 | 3,805,414 | 1 | 100.28 |
| Persistent trust radius | 3,110,876 | 5,360,211 | 10 | 99.84 |

Penalty doubling and temporary proximal damping both passed the five-scene
gate and reproduced the same best SSE/iteration on every scene. Temporary
damping activated only on 1723 and decayed to zero after four recorded
iterations, validating separation of the ADMM penalty from a vanishing recovery
metric. Its 1723 endpoint (`45.65M`) was moderately worse than penalty doubling
(`30.84M`) under the predeclared schedule.

Trust-radius-only recovery was made identical to the control before the first
rejection and used cumulative radius shrink afterward. It still accumulated ten
recoveries on 1723 and failed the recovery-count gate. This supports the claim
that the failure occurs at the global consensus composition, not solely in the
local trust-region controller.

Design, DABA partition audit, and next K10/K30 protocol:
`global_safeguard_design.md`.

## 2026-07-27: DRS consensus projection metric ablation

Ladybug-1723 was run for 90 DRS outer iterations with K10 and the supplied
`landmark_scalable` partition settings. The local nonlinear prox, pixel
objective, scaling, acceleration, and relative DRE/primal safeguard were fixed.
Only the metric used by reflected-camera projection and DRE terms changed.

| Projection metric | Best pixel SSE | Hard rejections | Fallbacks | Final hard `Be` |
|---|---:|---:|---:|---:|
| Arithmetic identity | 124,050,155 | 19 | 48 | 0.5 |
| Scalar per camera copy | 866,306 | 14 | 47 | 0.5 |
| Diagonal 9-parameter | 805,805 | 8 | 37 | 0.0128 |
| Full regularized 9x9 block | 764,872 | 7 | 32 | 0.0064 |

Full blocks improve SSE by 5.35% over diagonal and 13.26% over scalar. They
also require the fewest safeguard interventions and least metric escalation.
Arithmetic consensus generates nonfinite current trials and never improves the
saved initial best state; its finite result is best-state recovery, not a stable
endpoint. Every saved state independently reproduces recorded pixel SSE.

Readable report:
`benchmark_results/drs_consensus_metric_1723_k10_i90/report.md`.

## 2026-07-27: five-scene consensus projection breadth gate

Arithmetic, scalar, diagonal, and full block projection were compared on the
fixed five-scene cohort at K10 and 30 DRS iterations. Full blocks achieved the
lowest independently evaluated pixel SSE on all five scenes. Aggregate
geometric-mean SSE was `1,076,627` for full, versus `1,148,814` for diagonal,
`1,209,725` for scalar, and `4,630,080` for arithmetic. Full also required the
fewest hard rejections (`14`) and fallback trials (`42`).

All modes transmitted full worker metrics, so this is a projection-quality
ablation rather than a communication compression comparison. All 20 saved
states reproduce recorded pixel SSE exactly.

Readable report:
`benchmark_results/drs_consensus_metric_five_scene_k10_i30/report.md`.

## 2026-07-27: matched pixel versus DABA-ray objective ablation

The standard pixel and DABA regularized-ray objectives were optimized with the
same full-block DRS solver, five scenes, K10, 30 iterations, one local step,
partition, scaling, acceleration, and safeguard. All ten cases completed and
saved states reproduce their native objective exactly.

Ray optimization reduced hard rollbacks from 12 to 4, supporting its better
conditioning. Cross-evaluated geometric-mean mean pixel error was 2.48% worse,
and pixel RMSE was 35.34% worse. The discrepancy is tail-dominated: on scene
245, ray optimization improves median/p90/p95 but increases maximum pixel error
from `81.66` to `889.66` and RMSE from `1.29` to `5.76`. Ray 1723 has 14
noninvertible observations.

Readable report:
`benchmark_results/drs_objective_five_scene_k10_i30/report.md`.

## 2026-07-27: 29-scene safeguarded outer Nesterov matrix

The accelerated center proposal and plain ADMM fallback were run against the
unaccelerated guarded control on all 29 top-level BAL problems at K10/K20/K30,
30 outer iterations, one local Nesterov/DABA-TR update, and one CPU thread per
cluster. All 174 cases completed and saved states exactly reproduce pixel SSE.

Acceleration improved best SSE in 70/87 paired configurations, tied 9, and
lost 8. Geometric-mean best SSE improved by 5.4%, 8.6%, and 9.9% at K10, K20,
and K30. Runtime increased 26.6%. Endpoint-stable counts were nearly unchanged
(76/87 accelerated versus 75/87 control), because 646, 931, 1064, 1266, and
1723 remain difficult. The line search catches bad accelerated centers but
cannot prevent a bad plain fallback from being accepted into a poor basin.

Readable report:
`benchmark_results/admm_linesearch_overnight_i30_k10_k20_k30/report.md`.
