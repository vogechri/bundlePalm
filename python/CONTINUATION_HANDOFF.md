# BundlePalm DRS Paper Handoff

Updated: 2026-07-29

## Start Here

This workspace supports a paper about variable-metric Douglas--Rachford
splitting (DRS) for distributed bundle adjustment (BA). The primary objective is
the standard Snavely pixel reprojection objective. The intended paper is not a
claim to be the first decentralized or accelerated BA method. It asks when it is
useful to replace frequent synchronized linear-solver collectives with
independent nonlinear local BA solves and camera-only outer consensus.

On a new PC, read these documents in order:

1. [This handoff](CONTINUATION_HANDOFF.md)
2. [Paper plan](paper_plan.md)
3. [Method and cumulative evaluation spine](paper_method_evaluation_spine.md)
4. [Corrected 29-scene benchmark plan](benchmark_results/drs_29_scene_benchmark_plan.md)
5. [Competitor evaluation matrix](competitor_evaluation_matrix.md)
6. [Block-metric consensus derivation](block_metric_consensus_derivation.md)

Historical experiment summaries are in
[EXPERIMENT_STATUS.md](benchmark_results/EXPERIMENT_STATUS.md), but that file was
last refreshed on 2026-07-27. Prefer the dated results and decisions in this
handoff when they conflict.

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

## Active Experiment: L-BFGS and Anderson

Runner: [run_drs_29_scene_secant_acceleration.sh](serverTest/run_drs_29_scene_secant_acceleration.sh)

Analyzer: [analyze_drs_29_scene_secant_acceleration.py](serverTest/analyze_drs_29_scene_secant_acceleration.py)

Progress: [secant progress](benchmark_results/drs_29_scene_secant_acceleration_k30_i90/progress.md)

The clean coordinator now integrates L-BFGS and Anderson under the same
line-search, safeguard, fallback, restart, recovery, and oracle-accounting rules
as Nesterov. Secant algebra flattens the product-space tensors internally and
reshapes proposals back to `(clusters, cameras, 9)`. Focused tests pass and real
worker smoke runs produced accelerated acceptances.

Status at this handoff: L-BFGS and Anderson each have `25/29` unique completed
scenes. No solver process was active at the final check. The four missing scenes
for both methods are 1490, 1723, 1778, and 3068. Regenerate the report rather
than trusting raw line counts, because interrupted overlapping resumes created
duplicate JSONL rows that the analyzer correctly deduplicates by scene.

The latest provisional aggregate should always be regenerated from the JSONL
files; the final checked count was 25 four-way scenes.

- Nesterov equal-iteration ratio `0.984773`.
- L-BFGS equal-iteration ratio `0.982762`.
- Anderson equal-iteration ratio `0.982997`.
- Nesterov/L-BFGS/Anderson equal-oracle ratios:
  `0.999400/0.996944/0.997224`.
- L-BFGS and Anderson are slightly better at fixed iterations but much slower
  in the current Python implementation.
- Anderson is operationally cleaner; L-BFGS has many more nominal fallbacks.

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

After completion, independently verify all saved states as done for Phase 0.

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

1. Finish and validate the 29-scene L-BFGS/Anderson run.
2. Decide whether Anderson or L-BFGS merits promotion beyond an acceleration
   ablation. Current evidence favors Anderson operationally and Nesterov as the
   default efficiency tradeoff.
3. Build a unified Phase-1 one-factor matrix runner/analyzer. The design exists,
   but not every row has a one-click runner.
4. Run the remaining one-factor screen against the corrected plain denominator:
   consensus modes, no/Jacobi scaling, recovery modes, initial curvature,
   reset-DABA/persistent-DABA/DRS trust, local landmark refinement, line-search
   depth, final polishing, local linear solver, and partition construction.

### Method Development

5. Collect per-cluster trust-radius and model-ratio traces on 245, 3068, 1723,
   and a neutral scene.
6. Implement and test a trust hybrid: carry accepted radius on good progress;
   restore, shrink, or reset it after rollback.
7. Replace fixed curvature decay with a cautious adaptive policy if diagnostics
   support it.
8. Evaluate Schur-PCG versus local Nesterov at matched products/residuals.
9. Complete the cumulative forward-selection ladder over all 29 scenes.
10. Confirm selected plain, accelerated, and cumulative methods at K10 and K20.

### Paper and External Validity

11. Run controlled same-machine competitors where feasible: Ceres, STBA, and
    available MegBA/DABA configurations.
12. Add communication bytes, synchronization rounds, max-worker and aggregate
    RSS, and time-to-quality plots to the main comparison.
13. Run network latency/bandwidth and straggler experiments only after the
    single-host method is frozen.
14. Extend external validity with the SfM_Init-derived 1DSfM pipeline; do not
    label it a reproduction of DABA Table II.
15. Decide whether finite-local-solve diagnostics support an inexact-prox
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
  CONTINUATION_HANDOFF.md \
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