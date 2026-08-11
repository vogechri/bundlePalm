# BundlePalm DRS Paper Handoff

Updated: 2026-08-10

## 2026-08-11 Crash Recovery Boundary

A WSL/network crash caused an unclean restart and VS Code restored several open
source files to a coherent earlier editor snapshot. This was not an OOM: after
restart WSL had about 26 GB available and zero swap use. All four C5 breadth
campaigns had already completed; all 36 rows, JSONL files, worker timing, and
states are intact.

Source recovery used VS Code Local History for the maintained paper path:
`client_drs.py`, `client_admm.py`, `main.cpp`, `proto/test.proto`, the general
runner, consensus/safeguard modules, and worker-consensus tests. The recovered
state passes 128 maintained tests and rebuilds `zeromq_cpp_server_ex`.

The later exact checkpoint/BAE handoff oracle implementation did not have a
complete final source snapshot. Its benchmark artifacts and conclusions remain
valid historical evidence, but its old checkpoint/staged runners must not be
assumed runnable from the recovered source. Similarly, standalone coupled and
factorized metric math/tests are retained, while the full experimental C3
coordinator execution path requires deliberate reconstruction before any new
C3 campaign. Do not reconstruct either path implicitly while running the
C1--C5 paper matrix.

> Current research restart contract: [DRS Research Objective](DRS_RESEARCH_OBJECTIVE.md).
> Read that file first. It supersedes this historical handoff for the active
> objective, benchmark scoreboards, promotion rules, and next experiment.

## Start Here

This workspace supports a paper about variable-metric Douglas--Rachford
splitting (DRS) for distributed bundle adjustment (BA). The primary objective is
the standard Snavely pixel reprojection objective. The intended paper is not a
claim to be the first decentralized or accelerated BA method. It asks when it is
useful to replace frequent synchronized linear-solver collectives with
independent nonlinear local BA solves and camera-only outer consensus.

On a new PC, read these documents in order:

1. [DRS Research Objective](DRS_RESEARCH_OBJECTIVE.md)
2. [This handoff](CONTINUATION_HANDOFF.md)
3. [Paper plan](paper_plan.md)
4. [Method and cumulative evaluation spine](paper_method_evaluation_spine.md)
5. [Corrected 29-scene benchmark plan](benchmark_results/drs_29_scene_benchmark_plan.md)
6. [Competitor evaluation matrix](competitor_evaluation_matrix.md)
7. [Block-metric consensus derivation](block_metric_consensus_derivation.md)

Historical experiment summaries are in
[EXPERIMENT_STATUS.md](benchmark_results/EXPERIMENT_STATUS.md), but that file was
last refreshed on 2026-07-27. Prefer the dated results and decisions in this
handoff when they conflict.

## Current K1/K2 Inner-Solver Checkpoint

> **Status correction (2026-08-11):** K1 is closed and the K2 portfolio work
> below is historical evidence. The maintained single-trajectory C4 carry-over
> is also closed after K2, K24, six-scene, and all-15 post-crash gates. PCG is
> faster but not a uniform quality replacement (`1.0055x` geometric per-scene
> SSE, `1.1953x` worst regression), so Nesterov remains the named baseline and
> PCG remains a separate C4 factorial level. Resume the retained plain/C1/C5/
> C1+C5 publication matrix. Do not resume checkpoint races, trust interpolation,
> tolerance tuning, or branch portfolios.

K1 is closed: the internal deterministic BAE-style left-SE3 path matches Ceres
aggregate quality on all 15 1DSfM scenes (`0.993529x` SSE). The portable package
retains points-p95 normalization, Jacobi camera scaling, and landmark
preconditioning.

The historical K2 candidate was an exact reset-versus-preserve handoff race:

1. Run the BAE inner package once through I6 and save a full DRS checkpoint.
2. Resume reset and preserve-trust branches from that identical I6 state to I15.
3. Select the lower accepted global SSE and resume that exact I15 checkpoint to
   I30.

Full checkpoints include coordinator product-space/accepted/best state and
opaque per-cluster worker current/rollback/nominal geometry and trust state.
Roman reset and preserve resumes are bitwise identical to uninterrupted runs;
all selected six-scene prefixes pass the same exactness check.

Frozen K2 six-scene result versus the matched current baseline:

- SSE geomean: `0.911654490x`.
- Net 39-iteration optimization cost: `1.387978927x`.
- W/T/L: `3/0/3`; worst case: Gendarmenmarkt `1.082880364x`.
- Preserve selected for Gendarmenmarkt, Piccadilly, Roman, Trafalgar, Vienna;
  reset selected for Union.
- Runner: `serverTest/run_k2_bae_handoff_race.sh`.
- Artifacts: `benchmark_results/k2_bae_handoff_checkpoint_race_six_i15_i30`.

The frozen policy was then evaluated on the canonical nine held-out 1DSfM
scenes with a matched current-code baseline:

- Held-out SSE geomean: `1.014166145x`.
- Held-out net cost: `1.548392067x`; W/T/L: `4/0/5`.
- Combined all-15 SSE geomean: `0.971846277x`.
- Combined all-15 net cost: `1.482114348x`; W/T/L: `7/0/8`.
- All selected and alternate continuations passed exact-prefix checks.

Continuing every unselected held-out branch to I30 showed that the I15 selector
picked the better final reset/preserve branch on all 9/9 scenes. The held-out
losses are therefore branch-portfolio failures, not selector failures: both
handoff branches lose to baseline on Madrid, Montreal, Notre Dame, Piazza, and
Tower.

The exact race improves aggregate all-15 quality but is not a safe per-scene K2
replacement, so it must not be promoted to K24. The next mechanism is a
three-way exact portfolio that adds the unchanged baseline live branch. Do not
resume scalar threshold/cutoff tuning or change frozen I6/I15 before testing
that K2 portfolio. See
`benchmark_results/k2_bae_handoff_checkpoint_race_all15_report.md`.

That baseline safeguard has now been tested as an exact staged policy. The I15
reset/preserve winner is continued to I20 and compared with an independent
unchanged baseline I20 branch before exact continuation to I30.

- Development SSE/cost: `0.912436064x` / `2.345838703x`, W/T/L `2/4/0`.
- Held-out SSE/cost: `0.983534247x` / `2.183130588x`, W/T/L `3/5/1`.
- All-15 SSE/cost: `0.954453267x` / `2.246813826x`, W/T/L `5/9/1`.
- Worst case: Tower `1.009773926x`.
- All I15 and I20 selected prefixes are bitwise exact.

The staged baseline branch largely solves the safety/generalization problem,
but its measured optimization cost is too high for practical promotion. The
I20 selector is correct on 12/15 scenes; exact suppressed continuations show an
endpoint oracle of `0.943303294x`, W/T/L `7/8/0`. Do not tune I6/I15/I20 and do
not run K24. Next work must reduce portfolio cost while preserving the exact
baseline fallback and comparing against this staged oracle. Runner:
`serverTest/run_k2_bae_staged_baseline_race.sh`; report:
`benchmark_results/k2_bae_staged_baseline_race_all15_report.md`.

Cost profiling shows that worker `solveBatch` accounts for 96--98% of branch
optimization time. Checkpoint compression was removed because it saved only 9%
space while making representative writes about 37x slower; checkpoints now use
atomic raw NPZ. This does not materially reduce the algorithmic work ratio.

A cheaper baseline-versus-preserve portfolio and cadence-1 inner stopping were
tested exactly, but both lose Ellis Island and cadence-1 also loses useful
Madrid/Trafalgar basins. They do not replace the staged safety reference.
Balanced concurrency is valid as a wall-time optimization: on this 24-core
machine, running baseline and preserve with six threads per cluster each is
bitwise identical to sequential 12-thread branches and reduced Roman branch
wall time from 65s to 46s. It does not reduce aggregate CPU work (~`2.18x`).
Opt-in runner: `serverTest/run_k2_bae_preserve_baseline_race.sh` with
`PARALLEL_INITIAL_BRANCHES=1 PARALLEL_THREADS_PER_CLUSTER=6`.

The remaining blocker is genuine duplicated local-solve work. Do not run K24
until a cheaper policy retains the exact staged baseline fallback and Ellis
reset rescue.

A single-trajectory log-space midpoint was tested rather than swept. At I6 it
sets each trust radius to `sqrt(100 * learned_radius)` and restores all other
baseline controls. Roman/Trafalgar/Union ratios are `1.135521x`, `0.799454x`,
and `1.364832x`; geomean `1.074045x`. Union still develops severe cluster-radius
divergence. Do not broaden this mechanism or tune interpolation exponents; it
does not replace the staged branch oracle.

Persistent trust coupling was also tested without a parameter sweep. A new
default-off `SHARED_TRUST_REGION_FROM=7` mode prepares the geometric mean of the
learned I6 radii and forces/recomputes a shared starting radius every later
iteration. Roman/Trafalgar/Union ratios are `1.329623x`, `0.869313x`, and
`1.220048x`; geomean `1.121400x`. It moderates Union relative to full preserve
but destroys Roman quality and remains worse than reset. Do not tune the shared
aggregation or start iteration and do not broaden this run.

The K1-to-K2 trust-state transfer investigation is now closed: no scalar,
one-time synchronized, or persistently synchronized trust handoff reproduces
the exact staged oracle at practical cost. Keep the current K2 baseline as the
practical method and the exact staged policy as a diagnostic upper bound. The
next research phase should return to the broader C1--C5 paper spine; do not run
K24 handoff experiments.

## C1 Long-Horizon Publication Gate

Safeguarded C1 was extended from the short I5 breadth result to matched
K24/I30 controls. The frozen protocol uses shared-only block/full DRS,
points-p95/Jacobi coordinates, Schur-PCG at `1e-2`, persistent DABA trust, and
`themelis_nesterov` with grid `0,1`, fallback, and restart after three failures.

- 1DSfM all 15: SSE `0.850594748x`, optimization `1.541090523x`, proximal
  oracles `1.673801203x`, true worker CPU `1.726258059x`, W/T/L `13/0/2`, worst
  Trafalgar `1.045075260x`.
- Large BAL1490/1778/3068: SSE `0.994416/0.977676/1.002193x`, geomean
  `0.991375306x`, optimization `1.724159x`, true worker CPU `1.753382x`.

C1 is now validated as a long-horizon 1DSfM quality component and remains safe
on the tested large-BAL cohort. It is not a universal practical speedup, and
the strong large-BAL I5 gains largely disappear by I30. Keep C1 in the isolated
and cumulative paper matrix; do not tune it from Trafalgar or NYC. Report:
`benchmark_results/c1_themelis_all15_large_bal_i30_report.md`.

## C2, C4, And C5 Long-Horizon Gates

The C1 x C2 sentinel factorial is scene-dependent. Raw C1 strongly improves
Roman and Trafalgar, while points-p95/Jacobi C2 is neutral on Trafalgar by
itself but changes the combined C1 result from `0.614318x` to `1.042354x` raw
control. Roman shows positive C1+C2 synergy; BAL interactions are nearly
multiplicative. Keep C2 isolated rather than automatically cumulative after C1.

The C4 Schur-PCG/Nesterov factorial is also context-sensitive. Equal configured
`1e-2` tolerance does not mean equal work: PCG delivers residuals around
`0.006--0.007`, Nesterov around `0.0008--0.0011`, and Nesterov costs 1.11--1.33x
true worker CPU across contexts. Retain C4 as a finite-work/basin ablation; do
not promote one solver as the cumulative default or claim matched efficiency.
The maintained post-crash all-15 K24/I30 gate confirms that decision: PCG gives
`0.9851x` summed SSE, `1.0055x` geometric-mean per-scene SSE, `8/15` quality
wins, `0.9317x` geometric-mean optimization time, and a `1.1953x` worst-scene
regression. Nesterov remains the named baseline; PCG is an explicit C4 level.

C5 adaptive local depth completed all-15 and large-BAL breadth at I30:

- C5/plain all-15: SSE `0.952889269x`, W/T/L `13/0/2`, no extra proximal
  oracle calls, true worker CPU `1.468371581x`.
- C1+C5/plain all-15: SSE `0.786739413x`, W/T/L `15/0/0`, worst
  `0.947482538x`, true worker CPU `2.544049696x`.
- C1+C5/C1: SSE `0.924928605x`, W/T/L `10/0/5`.
- Large BAL C5/plain: `0.999815407x`; C1+C5/C1: `1.004690704x`.

Those retained artifacts showed selective C5 activation and weak large-BAL
transfer. The fresh current-source global factorial below supersedes their
routing decision: retain C5 and C1+C5 across both families and tune one common
threshold/work policy. Do not configure by scene. Reports:
`benchmark_results/c1_c2_sentinel_factorial_i30_report.md`,
`benchmark_results/c1_c2_c4_sentinel_factorial_i30_report.md`, and
`benchmark_results/c1_c5_all15_large_bal_i30_report.md`.

The consolidated maintained Stage-C decision table is
`benchmark_results/stage_c_long_horizon_decision_table.md`. It defines the
current manuscript matrix: plain, C1, C5, C1+C5 breadth; separate C2 and C4
sentinel factorials; and historical C3 evidence marked artifact-only until its
coordinator path is deliberately reconstructed.

Fresh current-source confirmation now covers the complete global
plain/C1/C5/C1+C5 K24/I30 factorial over all 15 1DSfM and all 29 BAL scenes.
C1/plain is `0.865147x` and `0.983780x`; current C5/plain is `1.015807x` and
`1.000057x`; C1+C5/C1 is `1.005680x` and `0.999855x`. Retain both innovations
and the intended combined architecture under one common configuration. Tune
C5's common thresholds/work policy next; never select settings per scene. The
authoritative report is
`benchmark_results/stage_c_final_all15_all29_k24_i30/comparison_to_ceres_and_plain_drs.md`.
The one-factor tuning ladder and cross-family acceptance criteria are frozen in
`benchmark_results/stage_c_global_c5_tuning_plan.md`.

That ladder is complete. The promoted common C5 policy starts at I5 with
high/low `0.35/0.20`, window/dwell `3/3`, and maximum depth 2. Tuned C1+C5/C1
is `0.980053x` on all-15 1DSfM and `0.999486x` on all-29 BAL; tuned
C1+C5/plain is `0.847890x` and `0.983275x`. Use C1+C5 as the final Stage-C
stack and preserve C1/C5 switches for ablations. Full report:
`benchmark_results/stage_c_tuned_c5_final_report.md`.

The identical Huber `0.5` contract is also implemented and validated across
worker, coordinator safeguards, evaluator, and left-SE3 Ceres. Current
raw-start Huber I30 and L2-to-Huber continuation do not transfer competitively,
so neither is broadened. Tuned L2 C1+C5 remains promoted. Details:
`benchmark_results/stage_c_huber_sentinel_report.md`.

K>1 scaling is now confirmed for the same frozen L2 C1+C5 policy. A five-K
sentinel sweep selected K4 and K16, and both were run unchanged on all 15
1DSfM and all 29 BAL scenes. Retain K4 as the global resource endpoint and K16
as the global latency endpoint; this is not scene routing. Full report:
`benchmark_results/stage_c_scaling_confirmation_k4_16_i30/report.md`.

The machine-readable reproduction index is
`benchmark_results/stage_c_reproducibility_manifest.json`, generated by
`serverTest/build_stage_c_reproducibility_manifest.py`. It validates complete
JSONL/configuration/worker-timing coverage for plain, C1, C5, and C1+C5 across
all-15 1DSfM and the large-BAL cohort. Regenerate it after moving or replacing
any authoritative result artifact.

The manuscript-ready long-horizon results section is
`stage_c_long_horizon_results.tex`. It compiles as a dependency-free input
fragment and contains the breadth and C1/C2/C4 factorial tables. Stage-C labels
are now authoritative throughout the paper plan: C1 safeguarded fast DRS, C2
coordinate equilibration, C3 variable-metric consensus, C4 finite local Schur
solver, C5 adaptive local work, and C6 optional globalization.

The exact staged policy now has an opt-in balanced parallel mode. On the 24-core
host, reset/preserve and handoff/baseline phases run concurrently with six
threads per cluster per branch, while source and selected continuation retain
12 threads. All 15 selected trajectories are bitwise identical to the
sequential staged policy, preserving `0.954453267x` quality and the same tail.
Campaign wall time falls from about 813s to 586s (27.9%), but remains `2.056x`
the matched baseline wall time. The earlier `2.331x` number is summed branch
elapsed, not CPU-seconds. Corrected worker timing on Roman measures true worker
CPU at 79.32s sequential versus 92.12s parallel (`1.161x`). Use
`PARALLEL_BRANCHES=1 PARALLEL_THREADS_PER_CLUSTER=6` with
`serverTest/run_k2_bae_staged_baseline_race.sh`. This is a latency variant, not
an algorithmic-work improvement or a reason to proceed to K24.

## Paper Goal

The central research question is:

> When is it preferable to spend more computation in independent nonlinear
> local BA solves in exchange for fewer global synchronization points?

The proposed method has the following dependency structure:

1. Assign each complete landmark track to one worker.
2. Duplicate only cameras observed by more than one landmark owner.
3. Solve local nonlinear proximal BA problems independently.
4. Reconcile duplicated cameras with full regularized `9x9` block-metric
   consensus.
5. Apply consistent scene normalization and camera coordinate equilibration.
6. Safeguard outer acceleration with physical pixel SSE and the DRS envelope,
   retaining nominal DRS fallback and accepted-state recovery.
7. Use finite Schur power/Nesterov solves for local work.
8. Treat partition construction as a constrained balance, conditioning,
   duplication, memory, and communication problem.

The strongest publishable hypothesis is currently a resource and
synchronization tradeoff, especially lower maximum worker memory and
camera-only communication at matched quality. Raw CPU-versus-GPU speed is not
the primary claim.

## Mathematical Core

The product-space DRS iteration is

```text
u_k = prox^M_{gamma F}(s_k)
v_k = P_C^M(2 u_k - s_k)
s_{k+1} = s_k + lambda (v_k - u_k)
```

For a duplicated camera `c`, full block consensus is

```text
v_c = (sum_i D_i,c)^-1 sum_i D_i,c (2 u_i,c - s_i,c).
```

The practical block metric separates curvature from weak-direction
regularization. `BLOCK_CURVATURE_MULTIPLIER` controls the approximate local
Lipschitz/curvature scale; `BLOCK_REGULARIZATION` protects weak directions.

The clean safeguard rejects a trial only when both the DRS-envelope and
physical-primal thresholds fail, or when values are nonfinite. Recovery restores
the last accepted consistent camera/landmark state and can increase curvature.

## Code Map

The main implementation surface is:

- [Clean coordinator](serverTest/client_drs.py): product-space DRS,
  full/scalar metrics, safeguards, recovery, timing, curvature decay, trust
  policies, and outer acceleration.
- [Worker transport](serverTest/client_admm.py): ZeroMQ protocol, worker state,
  preconditioning update, landmark transport, and optional landmark refinement.
- [C++ worker](serverTest/main.cpp): local BA solve, Schur solver, metric blocks,
  trust-region policies, and optional diagnostics.
- [Protocol](serverTest/proto/test.proto): camera/landmark and metric transport;
  proximal landmarks use double precision.
- [Acceleration methods](serverTest/outer_acceleration.py): legacy Nesterov,
  L-BFGS, Anderson, FISTA-style variants, and line-search interpolation.
- [General runner](serverTest/run_drs_failure_top3_live.sh): resumable runs,
  state/log/timing output, configurable ports, and isolated worker process
  groups.
- [Acceleration tests](serverTest/test_admm_acceleration.py)
- [Safeguard tests](serverTest/test_drs_safeguards.py)

Important runtime contract:

- coordinator PUSH requests / worker PULL on the request port;
- worker PUSH results / coordinator PULL on the result port;
- `serverTest/build_admm/zeromq_cpp_server_ex` is the worker executable;
- `serverTest/.venv/bin/python` is the validated Python environment;
- serial C++ builds (`-j1`) are safer because GCC/Eigen has previously hit an
  internal compiler error.
- `client_admm.py` and the general DRS runner default to `build_admm`; generated
  Python/C++ protobuf code must be rebuilt from `serverTest/proto/test.proto`
  after cloning. Runtime descriptor checks reject stale generated schemas before
  worker communication.

## Critical Correctness Fixes Already Made

Results predating these fixes must not be used directly for final ranking:

1. Worker initialization is now bootstrap work before counted DRS iteration 0.
2. Physical initialization is followed by an explicit preconditioning update.
3. Proximal landmarks are transported as double, not float.
4. Corrected camera scaling includes `2/sqrt(K)`.
5. Reporting-only landmark refinement no longer contaminates hidden worker or
   safeguard state.
6. Default partitioning is `landmark_scalable`; the stability variant is an
   explicit ablation.
7. Worker processes run in isolated process groups (`setsid`), so cleanup cannot
   terminate the parent benchmark runner.
8. Curvature decay counts accepted iterations since an actual curvature
   increase; an unrelated rejection does not erase the counter.

## Superseded Assumption: Lazy Nesterov Fallback

Do not implement the old note that nominal Nesterov fallback can be evaluated
only after an accelerated failure. The selected recurrence is

```text
d_k = (T(s_k) - s_k) + beta_k d_{k-1}.
```

It requires the current nominal fixed-point image `T(s_k)` before the
accelerated proposal exists. The nominal oracle is already reused as fallback.
Removing it would define a different lagged-prediction method. Report both
equal-iteration and equal-prox-oracle comparisons instead.

Some older reports and persistent memory notes still mention lazy nominal
fallback. This paragraph supersedes those statements.

## Verified 29-Scene Results

### Phase 0: Plain Versus Binary Nesterov

Report: [Phase-0 report](benchmark_results/drs_29_scene_phase0_k30_i90/report.md)

Protocol: all 29 local BAL scenes, `K=30`, 90 outer iterations, one local
nonlinear step, reset-DABA trust, full block consensus, Jacobi scaling,
curvature-only recovery, no curvature decay, and no landmark refinement.

- 29/29 plain and 29/29 Nesterov cases completed.
- All 58 saved NPZ states independently reproduce recorded canonical pixel SSE
  exactly.
- Equal iteration Nesterov/plain geomean SSE: `0.982266`, W/T/L `27/2/0`.
- Equal oracle Nesterov/plain geomean SSE: `0.995403`, W/T/L `15/4/10`.
- Oracle calls: `2610/5016` plain/Nesterov.
- Rejections: `62/66`.
- Optimization totals: `1688.45/3018.85` seconds.
- Largest equal-work gains are concentrated in hard scenes 245, 3068, 394,
  and 1723.

Conclusion: binary Nesterov is a strong finite-iteration quality accelerator,
but not a free computational speedup. It belongs in the cumulative method, with
both equal-iteration and equal-work reporting.

### Persistent DABA Versus Persistent DRS Trust

Reports:

- [Persistent DABA progress](benchmark_results/drs_29_scene_persistent_daba_nesterov_k30_i90/progress.md)
- [Persistent DRS report](benchmark_results/drs_29_scene_persistent_drs_nesterov_k30_i90/report.md)

The methods are approximately tied: DRS/DABA geomean SSE `0.999769`, W/T/L
`11/10/8`, and optimization-time ratio `1.00735`. They are complementary:

- DRS closes much of the 3068 and 1723 gap.
- DABA avoids the severe DRS regression on 245.
- Rejection count and final curvature alone cannot choose between them.

The next trust-policy task is per-cluster radius/model-ratio diagnostics and a
hybrid accepted-radius rollback policy.

### Curvature Decay

Reports:

- [Corrected K=6 decay report](benchmark_results/drs_29_scene_persistent_daba_nesterov_decay6_k30_i90/report.md)
- [Corrected K=10 decay report](benchmark_results/drs_29_scene_persistent_daba_nesterov_decay10_corrected_k30_i90/report.md)

Controlled corrected-semantics comparison:

- K10/K6 geomean SSE: `1.001431`.
- K10 direct W/T/L versus K6: `1/19/9`.
- K6 calls/rejections/decays: `4783/242/175`.
- K10 calls/rejections/decays: `4823/224/122`.

K6 is better for quality, especially on 3068 and 1723. Fixed decay remains
optional because it increases rejection cycling and hurts scenes such as 245.
A safer adaptive policy should require sustained high curvature, no recent
rejection, a cooldown, and immediate restoration after a failed decay.

### Full Block Consensus

The five-scene breadth gate found full `9x9` blocks best on all scenes:

- full block geomean SSE: `1,076,627`;
- diagonal: `1,148,814` (`+6.70%`);
- scalar: `1,209,725` (`+12.36%`);
- arithmetic was unstable on 1723.

See the links in [the experiment status](benchmark_results/EXPERIMENT_STATUS.md)
and [the derivation](block_metric_consensus_derivation.md).

### Phase-1 One-Factor Screen

The complete human-readable interpretation is in the
[Phase-1 summary](benchmark_results/drs_29_scene_phase1_k30_i90/summary.md), with
the machine-generated table in
[the Phase-1 report](benchmark_results/drs_29_scene_phase1_k30_i90/report.md).
The optimization matrix completed all 406 variant/scene rows. Its main decisions
are to retain full block consensus, promote DRS trust to the cumulative ladder,
test regularization recovery and curvature 0.05 as compatible additions, keep
binary Nesterov as the default acceleration tradeoff, and retain final landmark
polishing as output postprocessing. All 406 repaired NPZ states independently
reproduce their recorded canonical pixel SSE exactly. Atomic state saving now
prevents interrupted or overlapping runs from exposing partial NPZ archives.

## Completed Experiment: L-BFGS and Anderson

Runner: [run_drs_29_scene_secant_acceleration.sh](serverTest/run_drs_29_scene_secant_acceleration.sh)

Analyzer: [analyze_drs_29_scene_secant_acceleration.py](serverTest/analyze_drs_29_scene_secant_acceleration.py)

Progress: [secant progress](benchmark_results/drs_29_scene_secant_acceleration_k30_i90/progress.md)

The clean coordinator now integrates L-BFGS and Anderson under the same
line-search, safeguard, fallback, restart, recovery, and oracle-accounting rules
as Nesterov. Secant algebra flattens the product-space tensors internally and
reshapes proposals back to `(clusters, cameras, 9)`. Focused tests pass and real
worker smoke runs produced accelerated acceptances.

Final status: 29/29 unique scenes for plain, Nesterov, L-BFGS, and Anderson.
All 58 L-BFGS/Anderson NPZ states independently reproduce their recorded
canonical pixel SSE exactly. There were no failed cases and no active solver at
the final check. Interrupted overlapping resumes created duplicate raw JSONL
rows, but the analyzer deduplicates them by scene.

Final aggregate relative to plain DRS:

- Nesterov equal-iteration ratio `0.982266`, W/T/L `27/2/0`.
- L-BFGS equal-iteration ratio `0.980380`, W/T/L `28/1/0`.
- Anderson equal-iteration ratio `0.981218`, W/T/L `28/1/0`.
- Nesterov/L-BFGS/Anderson equal-oracle ratios:
  `0.995403/0.993269/0.994008`.
- Optimization-time ratios are `1.71/4.05/3.45` respectively over 28 timed
  pairs; scene 49 predates optimization-only timing.
- Calls/rejections/fallbacks are Nesterov `5016/66/119`, L-BFGS
  `5220/62/514`, and Anderson `5220/45/107`.

Direct geomean ratios are L-BFGS/Nesterov `0.998080`, Anderson/Nesterov
`0.998933`, and L-BFGS/Anderson `0.999146`. L-BFGS has the best aggregate
quality, but the gain over Nesterov is only 0.19% and costs substantial Python
full-state secant algebra plus many fallbacks. Anderson is operationally cleaner
and especially strong on 427, 646, 1723, and 1778. Nesterov remains the best
default efficiency tradeoff; retain L-BFGS and Anderson as paper ablations or
hard-scene candidates unless their coordinator algebra is optimized.

To inspect or resume on the current machine:

```bash
serverTest/.venv/bin/python \
  serverTest/analyze_drs_29_scene_secant_acceleration.py \
  benchmark_results/drs_29_scene_phase0_k30_i90 \
  benchmark_results/drs_29_scene_secant_acceleration_k30_i90

BUNDLE_PALM_REQUEST_PORT=17856 \
BUNDLE_PALM_RESULT_PORT=17857 \
OVERWRITE=0 LIVE_OUTPUT=0 DEBUG_OUTPUT=0 \
serverTest/run_drs_29_scene_secant_acceleration.sh
```

The independent saved-state verification is complete.

## Competitors and Positioning

The mandatory competitor families are detailed in
[competitor_evaluation_matrix.md](competitor_evaluation_matrix.md):

- DABA: primary decentralized MM baseline; official code exists but uses
  CUDA/MPI/NCCL and a different published ray metric.
- PenBA: recent distributed penalty method; full reproducible implementation is
  not currently available.
- Global camera consensus and asynchronous/lazy communication methods: required
  positioning and possible communication/straggler baselines.
- STBA: runnable approximate distributed Schur baseline.
- MegBA: primary exact synchronized GPU Schur/PCG baseline.
- LargeBA: memory/scale context; public repository is not a complete solver.
- Ceres/RootBA: strong centralized quality and robustness references.

Comparison policy:

1. Prefer same-machine, same-data, same-initialization, same-objective runs.
2. Keep published GPU times as context, not speedups against local CPU runs.
3. Report objective versus time, rounds, bytes, memory, and local work.
4. The original Snavely pixel objective is primary. DABA ray optimization is a
   separate objective/generalization study.

## Remaining Work, In Priority Order

### Immediate

1. Decide whether Anderson or L-BFGS merits any optimized follow-up beyond an
  acceleration ablation. Current evidence retains Nesterov as the default.
2. Build a unified Phase-1 one-factor matrix runner/analyzer. The design exists,
   but not every row has a one-click runner.
3. Run the remaining one-factor screen against the corrected plain denominator:
   consensus modes, no/Jacobi scaling, recovery modes, initial curvature,
   reset-DABA/persistent-DABA/DRS trust, local landmark refinement, line-search
   depth, final polishing, local linear solver, and partition construction.

### Method Development

4. Collect per-cluster trust-radius and model-ratio traces on 245, 3068, 1723,
   and a neutral scene.
5. Implement and test a trust hybrid: carry accepted radius on good progress;
   restore, shrink, or reset it after rollback.
6. Replace fixed curvature decay with a cautious adaptive policy if diagnostics
   support it.
7. Evaluate Schur-PCG versus local Nesterov at matched products/residuals.
8. Complete the cumulative forward-selection ladder over all 29 scenes.
9. Confirm selected plain, accelerated, and cumulative methods at K10 and K20.

### Paper and External Validity

10. Run controlled same-machine competitors where feasible: Ceres, STBA, and
    available MegBA/DABA configurations.
11. Add communication bytes, synchronization rounds, max-worker and aggregate
    RSS, and time-to-quality plots to the main comparison.
12. Run network latency/bandwidth and straggler experiments only after the
    single-host method is frozen.
13. Extend external validity with the SfM_Init-derived 1DSfM pipeline; do not
    label it a reproduction of DABA Table II.
14. Decide whether finite-local-solve diagnostics support an inexact-prox
    theorem. Do not overclaim practical convergence from exact fixed-metric DRS.

## What Must Be Checked In

Current branch: `user/chvogel/notSamePerformanceNew2c`

Remote: `origin https://github.com/vogechri/bundlePalm.git`

At handoff time, these source changes are staged but not committed:

- `serverTest/client_drs.py`
- `serverTest/outer_acceleration.py`
- `serverTest/test_admm_acceleration.py`

These important files are untracked and should be added:

- `CONTINUATION_HANDOFF.md`
- `benchmark_results/drs_29_scene_benchmark_plan.md`
- `serverTest/run_drs_29_scene_phase0.sh`
- `serverTest/analyze_drs_29_scene_phase0.py`
- `serverTest/run_drs_29_scene_persistent_daba_nesterov.sh`
- `serverTest/run_drs_29_scene_persistent_drs_nesterov.sh`
- `serverTest/run_drs_29_scene_persistent_daba_decay.sh`
- `serverTest/analyze_drs_29_scene_persistent_daba.py`
- `serverTest/analyze_drs_29_scene_decay.py`
- `serverTest/run_drs_29_scene_secant_acceleration.sh`
- `serverTest/analyze_drs_29_scene_secant_acceleration.py`
- `.gitignore`

The protobuf build-contract repair also requires checking in:

- `serverTest/CMakeLists.txt`
- `serverTest/client_admm.py`
- `serverTest/run_drs_failure_top3_live.sh`

Compact reports worth committing:

- `benchmark_results/drs_29_scene_phase0_k30_i90/report.md`
- `benchmark_results/drs_29_scene_persistent_daba_nesterov_k30_i90/progress.md`
- `benchmark_results/drs_29_scene_persistent_drs_nesterov_k30_i90/report.md`
- `benchmark_results/drs_29_scene_persistent_daba_nesterov_decay6_k30_i90/report.md`
- `benchmark_results/drs_29_scene_persistent_daba_nesterov_decay10_corrected_k30_i90/report.md`
- `benchmark_results/drs_29_scene_secant_acceleration_k30_i90/report.md` after
  the suite completes.

Do not blindly run `git add .`. The working tree contains many unrelated
historical outputs, build products, datasets, PDFs, virtual environments, and
binary libraries.

A focused staging command after the active run completes is:

```bash
git add \
  .gitignore \
  CONTINUATION_HANDOFF.md \
  serverTest/CMakeLists.txt \
  serverTest/client_admm.py \
  serverTest/client_drs.py \
  serverTest/outer_acceleration.py \
  serverTest/test_admm_acceleration.py \
  serverTest/run_drs_29_scene_phase0.sh \
  serverTest/analyze_drs_29_scene_phase0.py \
  serverTest/run_drs_29_scene_persistent_daba_nesterov.sh \
  serverTest/run_drs_29_scene_persistent_drs_nesterov.sh \
  serverTest/run_drs_29_scene_persistent_daba_decay.sh \
  serverTest/analyze_drs_29_scene_persistent_daba.py \
  serverTest/analyze_drs_29_scene_decay.py \
  serverTest/run_drs_29_scene_secant_acceleration.sh \
  serverTest/analyze_drs_29_scene_secant_acceleration.py \
  serverTest/run_drs_failure_top3_live.sh \
  benchmark_results/drs_29_scene_benchmark_plan.md \
  benchmark_results/drs_29_scene_phase0_k30_i90/report.md \
  benchmark_results/drs_29_scene_persistent_daba_nesterov_k30_i90/progress.md \
  benchmark_results/drs_29_scene_persistent_drs_nesterov_k30_i90/report.md \
  benchmark_results/drs_29_scene_persistent_daba_nesterov_decay6_k30_i90/report.md \
  benchmark_results/drs_29_scene_persistent_daba_nesterov_decay10_corrected_k30_i90/report.md \
  benchmark_results/drs_29_scene_secant_acceleration_k30_i90/report.md
```

Review `git diff --cached` before committing. The user must explicitly request a
commit/push; no commit is made by this handoff task.

## What Must Be Copied Separately

The six recent benchmark directories total about 1.2 GB because they include
NPZ states, logs, and memory records. Ordinary Git is not a good transport for
all of them.

For exact continuation on another PC, copy or archive:

- the top-level BAL `problem-*-pre.txt` files;
- `serverTest/build_admm/zeromq_cpp_server_ex` only if compatible, otherwise
  rebuild it;
- `serverTest/.venv` only if the Linux environments are compatible, otherwise
  recreate it;
- raw recent benchmark directories if trajectory/state-level analysis is
  needed;
- the local VS Code conversation transcript only as an optional audit archive.

The current conversation transcript is stored outside the repository in VS Code
workspace storage. It is large and machine-specific. This handoff is the
portable semantic summary; copying the transcript is optional and should not be
required to continue.

Suggested artifact policy:

- Commit source, runners, analyzers, plans, and compact Markdown reports.
- Use an external archive, shared storage, or Git LFS for JSONL/NPZ/log trees.
- Never commit `.venv`, `build*`, `__pycache__`, shared libraries, or duplicate
  BAL files accidentally.

## New-PC Bring-Up

1. Clone and check out `user/chvogel/notSamePerformanceNew2c` after it is pushed.
2. Open the `python` workspace directory.
3. Read this handoff and the linked paper/benchmark plans.
4. Restore the 29 BAL files or download/decompress them.
5. Recreate `serverTest/.venv` and rebuild `serverTest/build_admm` if binaries
   were not copied.
6. Run:

```bash
serverTest/.venv/bin/python -m pytest -q \
  serverTest/test_admm_acceleration.py \
  serverTest/test_drs_safeguards.py

bash -n \
  serverTest/run_drs_failure_top3_live.sh \
  serverTest/run_drs_29_scene_phase0.sh \
  serverTest/run_drs_29_scene_secant_acceleration.sh
```

7. Regenerate reports from copied JSONL results before launching new runs.
8. Continue from the priority list above rather than reopening pre-fix
   benchmark conclusions.

## Suggested First Prompt On The New PC

```text
Read CONTINUATION_HANDOFF.md and the linked paper and benchmark plans. Verify the
current branch, staged/untracked handoff files, and whether the 29-scene
L-BFGS/Anderson benchmark is complete. Regenerate its report, validate saved
states, then continue with the highest-priority unfinished item without
restarting completed experiments.
```