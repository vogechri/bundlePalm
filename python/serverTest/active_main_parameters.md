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
| `BUNDLE_PALM_CAMERA_DIAGONAL_FLOOR` | `1e-48` | explicit runner value; same as default | Relative eigenvalue floor for 9-by-9 camera metric blocks; directly swept and nonbinding through `1e-30` on the hard five. |
| `BUNDLE_PALM_CAMERA_DIAGONAL_MAXIMUM_GUARD` | `1e-32` | default | Guards a zero/tiny maximum camera diagonal before relative flooring; diagnostic probes recorded zero guarded blocks, but this guard was not independently swept. |
| `BUNDLE_PALM_CAMERA_PRECONDITIONER_DIAGONAL_FLOOR` | `1e-36` | default | Floors near-zero camera preconditioner entries; diagnostic probes recorded zero floor hits. |
| `BUNDLE_PALM_LANDMARK_PRECONDITIONER_FLOOR` | `1e-24` | default | Floors landmark preconditioner entries; diagnostic probes recorded zero floor hits. |
| `BUNDLE_PALM_MINIMUM_TRUST_REGION_RADIUS` | `1e-4` | default | Lower bound for persistent local trust-region radii. |
| `BUNDLE_PALM_INITIAL_TRUST_REGION_RADIUS` | `10` | default | Initial local trust-region radius. |
| `BUNDLE_PALM_MAXIMUM_TRUST_REGION_RADIUS` | `1e4` | promoted override; default `1e6` | Upper bound for local trust-region radii. |
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

The metric scale, camera update mode, and trust diagonal are materially active
in the selected trajectory. The camera metric floor is on the executed path but
was nonbinding in the tested range. The remaining floors and guard are active
numerical safeguards but were not observed to bind in instrumented runs.

### Numerical-Guard Evidence

| Parameter | Evidence | Conclusion |
|---|---|---|
| camera diagonal floor | Direct hard-five I200 sweep at `1e-54`, `1e-48`, `1e-42`, `1e-36`, and `1e-30` | Exact endpoint, rejection, fallback, and oracle-call no-op over this range. A separate 3068 diagnostic recorded zero floor hits at `1e-48`; larger floors such as `1e-12` can bind in older debug runs. |
| camera diagonal maximum guard | Instrumented camera-block probes | Zero guarded blocks were recorded. Observed minimum block maxima were vastly above the `1e-32` guard. No independent guard-value sweep was found. |
| camera preconditioner diagonal floor | Instrumented local-solve probes | Zero floor hits were recorded at `1e-36`; sampled minima were many orders larger (down to roughly `1e-18` in endpoint probes and about `7e-10` in the local grid). No current-method full-cohort sweep was run. |
| landmark preconditioner floor | Instrumented local-solve probes | Zero floor hits were recorded at `1e-24`; sampled minima were many orders larger (roughly `1e-11` to `1e-6` in representative probes). No independent floor-value sweep was found. |

Thus only `BUNDLE_PALM_CAMERA_DIAGONAL_FLOOR` has a direct recent sensitivity
sweep. The other three are classified as diagnostically nonbinding, not as
fully tuned parameters.

### Tower I50 Quantile Probe

Tower of London was monitored for K24/I50 with the current metric-75,
regularization-1e-4, maximum-radius-1e4 method. Each local solve emitted the 1%
quantile of the raw values presented to each safeguard. The table reports the
median emitted q01, which is a typical per-cluster/per-solve 1% threshold; the
q01 of emitted q01 values is included to show the lower tail across solves.

| Guard input | Median emitted q01 | Lower-tail emitted q01 | Existing value | Hits at existing value |
|---|---:|---:|---:|---:|
| relative camera diagonal | `1.23425e-9` | `6.85645e-11` | `1e-48` | 0 |
| raw camera-block maximum | `7.50972e6` | `2.43391e4` | `1e-32` | 0 guarded blocks |
| camera preconditioner diagonal | `0.3696225` | `0.106379` | `1e-36` | 0 |
| landmark preconditioner diagonal | `12.2632` | `0.531319` | `1e-24` | 0 |

The block-maximum guard is not independently effect-producing: it only changes
the `maxDiagonal` scale multiplied by the relative camera diagonal floor. With
that floor at `1e-48`, even a block guard near `7.5e6` still creates a negligible
absolute floor. Any effect probe for the maximum guard must therefore pair it
with a non-negligible relative camera floor and should be interpreted as an
interaction rather than an independent ablation.

Both the camera maximum guard and landmark preconditioner floor are restricted
to `[0,1]`. Their Tower median q01 values (`7.5e6` and `12.3`) are outside that
range. Consequently, the maximum guard cannot bind on this scene even at its
largest legal value; the landmark floor can only probe the lower tail using its
largest legal value `1`, not the measured median q01.

Matched Tower K24/I50 effect probes then set each legal threshold near the
measured q01 while keeping all other method parameters fixed:

| Variant | Best SSE | Ratio vs baseline | Rejections | Oracle calls |
|---|---:|---:|---:|---:|
| baseline guards | 1,978,770.409 | 1.000000 | 3 | 94 |
| camera diagonal floor `1.23425e-9` | 1,862,341.165 | **0.941161** | 2 | 95 |
| maximum guard `1` | 1,978,770.409 | 1.000000 | 3 | 94 |
| camera floor `1.23425e-9` + maximum guard `1` | 1,862,341.165 | 0.941161 | 2 | 95 |
| camera preconditioner floor `0.3696225` | 1,978,770.409 | 1.000000 | 3 | 94 |
| landmark preconditioner floor `1` | 1,978,770.269 | 1.000000 | 3 | 94 |

Only the relative camera diagonal floor materially changes this Tower I50
trajectory. The maximum guard adds nothing, confirming it remains nonbinding;
the camera and landmark preconditioner floors change no meaningful endpoint or
counter at their tested effect values. Because `1e-48` is far below the
measured relative-diagonal q01, making it still lower cannot create an effect:
the effect-producing direction is upward toward roughly `1e-9`. This is a
single-scene I50 finding and is not yet a promoted default.

### Notre Dame Camera-Floor Failure Diagnosis

An isolated K24/I200 matrix tested camera floors `1e-10`, `3e-10`, and `1e-9`
with maximum local trust radii `1e4` and `1e6`:

| Camera floor | Max radius `1e4`, original inverse | Max radius `1e6`, original inverse |
|---:|---:|---:|
| `1e-10` | non-finite local step | **5,083,124.295 SSE, completed** |
| `3e-10` | **5,095,656.566 SSE, completed** | non-finite local step |
| `1e-9` | non-finite local step | non-finite local step |

The apparent floor/trust-radius interaction was incidental. Failure-only
instrumentation traced the `1e-9`/max-`1e4` case through the following chain:

1. The physical camera transform is finite and nonsingular.
2. The camera step passed to that transform is already non-finite.
3. The first Nesterov inner iterate starts with a non-finite Schur right-hand
  side; this is not an accelerated-iteration divergence.
4. Camera inverse blocks and all gradient inputs are finite, but landmark block
  3804 produces a non-finite inverse.
5. That block is finite and positive definite, with entries around `1e-102` and
  eigenvalues `3.88e-108`, `9.22e-102`, and `9.45e-102`. Its direct 3x3
  determinant underflows to `3.38e-310` during Eigen's explicit inverse.

`BlockInverse` now retries only non-finite direct inverses by scaling the block
to unit magnitude, solving with LDLT, and scaling the inverse back. With the
baseline POBA floor still `0`, both disputed max-`1e4` cells complete I200:

| Camera floor | Repaired Notre Dame SSE |
|---:|---:|
| `1e-10` | **5,103,900.528** |
| `1e-9` | **5,297,874.778** |

An opt-in POBA floor of `1e-12` also completes the `1e-9` cell at
`5,128,504.537` SSE, but it changes the trajectory and remains a separate
untuned parameter. The camera floors themselves were not invalid.

### All-15 Camera-Floor Validation

The repaired hard-five screen promoted camera floors `3e-10` and `1e-8` to a
matched all-15 1DSfM K24/I200 evaluation. All 45 baseline/candidate cases
completed:

| Camera floor | Geomean SSE ratio | Wins/losses | Worst ratio | Rejections | Fallbacks | Oracle calls |
|---:|---:|---:|---:|---:|---:|---:|
| `1e-48` baseline | 1.000000 | baseline | 1.000000 | 232 | 161 | 5,626 |
| `3e-10` | **0.996823** | 6/9 | 1.050433 | 220 | 156 | 5,635 |
| `1e-8` | 1.010222 | 3/12 | 1.080252 | **217** | **155** | 5,632 |

Floor `3e-10` gives a modest `0.318%` geometric SSE reduction and slightly
fewer recoveries, but it is not uniform: NYC Library improves by `8.91%` and
Tower by `4.44%`, while Piazza regresses by `5.04%` and Roman Forum by `4.46%`.
Floor `1e-8` is rejected because it raises geometric SSE by `1.02%` and loses
on 12 of 15 scenes. Use `3e-10` as the best evaluated camera-floor candidate;
retain `1e-48` when minimizing worst-scene regression is more important than
the small aggregate gain.

### POBA Block Relative Floor Screen

With the camera floor fixed at the best evaluated candidate `3e-10`, a matched
hard-five K24/I200 screen tested `BUNDLE_PALM_POBA_BLOCK_RELATIVE_FLOOR` at
`0`, `1e-14`, `1e-12`, and `1e-10`. All 20 cases completed:

| POBA floor | Geomean SSE ratio | Wins/losses | Worst ratio | Rejections | Fallbacks |
|---:|---:|---:|---:|---:|---:|
| `0` baseline | **1.000000** | baseline | 1.000000 | 63 | 50 |
| `1e-14` | 1.010570 | 2/3 | 1.039451 | 63 | 53 |
| `1e-10` | 1.015578 | 1/4 | 1.043446 | **58** | **37** |
| `1e-12` | 1.018297 | 1/4 | 1.046047 | 60 | 64 |

No positive value passes the quality gate. Although `1e-10` reduces rejection
and fallback counts, it raises geometric SSE by `1.56%`; `1e-12` improves Notre
Dame slightly (`0.997111x`) but regresses four of five scenes. Keep the POBA
block relative floor at `0`. The scale-normalized `BlockInverse` fallback
provides the required numerical robustness without introducing this damping.

### Remaining One-Parameter Screens

Five further parameters were varied one at a time on the hard five at K24/I200
with camera floor `3e-10`. The best complete result from each screen was:

| Parameter | Tested values | Best value | Best ratio | Decision |
|---|---|---:|---:|---|
| Nesterov Schur Lipschitz | `0.75`, `0.85`, `0.9`, `1.0`, `1.1` | `0.9` | **1.000000** | keep default |
| camera trust diagonal scale | `3e-5`, `1e-4`, `3e-4`, `1e-3` | `1e-4` | **1.000000** | keep current |
| block curvature multiplier | `0.2`, `0.3`, `0.4`, `0.6`, `0.8` | `0.4` | **1.000000** | keep current |
| metric-proposal disagreement scale | `0.3`, `0.45`, `0.6`, `0.7`, `0.8`, `1.0` | `0.8` | **0.990073** | promote to all-15 gate |
| local acceptance ratio | `0.99`, `0.999`, `0.9999`, `1.0` | `0.9999` | **1.000000** | keep default |

Three aggressive arms did not complete the hard-five gate. Lipschitz `0.85`
produced an Alamo landmark block with minimum eigenvalue `8.94e-312`, whose
mathematical inverse exceeds finite double range. Curvature `0.8` exhausted
recovery on Tower at I71, and proposal scale `1.0` exhausted recovery on Tower
at I170. These are method/numerical failures rather than missing jobs.

Proposal scale `0.7` was checked separately after the initial grid. It completed
all five scenes but reached `1.013251x` geometric SSE with 2 wins and 3 losses;
Notre Dame was the largest regression at `1.062754x`. It is worse than both the
`0.6` baseline and `0.8`, so it was not promoted to all-15 validation.

Proposal disagreement `0.8` then completed all 15 1DSfM scenes at K24/I200:

| Proposal scale | Geomean SSE ratio | Wins/losses | Worst ratio | Rejections | Fallbacks | Oracle calls | Optimization s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.6` baseline | 1.000000 | baseline | 1.000000 | 220 | 156 | 5,635 | 571.320 |
| `0.8` | **0.983872** | 10/5 | 1.086993 | **151** | 188 | 5,708 | 607.407 |

Proposal scale `0.8` reduces geometric SSE by `1.61%`. The largest gains are
Piazza (`0.888247x`), Roman Forum (`0.917042x`), and Tower (`0.951888x`); the
largest regression is Madrid (`1.086993x`). It also takes `6.3%` more aggregate
optimization time. Promote `0.8` when aggregate I200 quality is primary; keep
`0.6` as the conservative setting when worst-scene regression and runtime are
more important.

Proposal `0.7` also completed the all-15 gate. It reaches `0.998027x`, 8 wins
and 7 losses, a `1.092370x` worst ratio, and `1.011030x` optimization time. It
is a mild runtime/quality compromise but is dominated by `0.8` for aggregate
quality and by `0.6` for robustness and runtime.

The existing per-iteration corrected-DRE grid over `0.6,0.7,0.8` was tested on
the hard five as an adaptive upper-bound experiment. It selected `0.6`, `0.7`,
and `0.8` on 664, 92, and 244 of 1,000 iterations, respectively, but reached
only `1.015055x` geometric SSE and took `171.876` optimization seconds versus
`141.070` for fixed `0.8`. Reject DRE-grid selection for these scales.

The next adaptive candidate should use the already-computed pre-damping metric
disagreement ratio with hysteresis: default to `0.8`, switch to stronger
damping `0.6` only when the ratio crosses a high threshold, and return to `0.8`
below a lower threshold. The observed distributions motivate initial high/low
pairs `(0.10, 0.03)` and `(0.20, 0.10)`. This requires extending the current
threshold mode, which switches between one damped scale and `1.0`, not directly
between `0.6` and `0.8`.

That two-scale hysteresis was implemented and tested on the hard five. The
`(low,high)=(0.03,0.10)` pair reached `1.009347x` and selected `0.6/0.8` on
113/887 iterations; `(0.10,0.20)` reached `1.054110x` and selected them on
85/915 iterations. Fixed `0.8` remains better at `0.990073x`. The first rule
improves Tower to `1,386,126.130` SSE but regresses NYC to `1,085,169.490`,
showing that the global disagreement ratio does not identify the useful damping
regime consistently across scenes. Do not promote either hysteresis rule.
Future adaptive work should combine disagreement with recovery/safeguard state
or use rotation/translation/intrinsics subspace energies.

### BAL Proposal-Disagreement Validation

Proposal scales `0.6`, `0.7`, and `0.8` were also run on all 29 BAL scenes at
K24/I90 with camera floor `3e-10`. BAL 3068 reproducibly exhausted recovery
before I90 for every scale (I30/I33/I44 respectively), so the matched I90
aggregate covers the other 28 scenes:

| Proposal scale | Geomean SSE ratio | Wins/losses | Worst ratio | Rejections | Fallbacks | Optimization-time ratio |
|---:|---:|---:|---:|---:|---:|---:|
| `0.6` | **1.000000** | baseline | 1.000000 | 16 | 24 | 1.000000 |
| `0.7` | 1.000262 | 12/16 | 1.002353 | 18 | **22** | **0.986408** |
| `0.8` | 1.000620 | 12/16 | 1.010597 | 22 | 23 | 1.006734 |

Unlike 1DSfM, BAL does not benefit from raising proposal disagreement to `0.8`.
Scale `0.7` is effectively quality-neutral and slightly faster, but `0.6`
retains the strongest matched I90 endpoint and best worst-scene behavior. Thus
`0.8` is a 1DSfM quality setting, not a universal BAL/1DSfM setting.

Focused BAL 3068 controls show that its early stops are unrelated to the
scale-normalized LDLT fallback: worker logs contain no non-finite inverse or
camera-update diagnostics, and every coordinator exits normally with
`terminationReason=recovery_exhausted`. Both the camera floor and proposal
scale are independently destabilizing on this scene:

The previous Metric-75 result of `3,263,766.254` used floor `1e-48` and maximum
radius `1e6`. A matched floor-`1e-48`, max-`1e4` control reaches
`3,270,255.309`, only `0.20%` worse. With floor `3e-10`, max-`1e4`, and enough
curvature headroom to complete I90, the endpoint is `3,909,069.271`, or
`19.53%` worse than the matched floor-`1e-48` control. Thus the raised camera
floor, not the trust-radius cap or LDLT fallback, causes the large regression.

| Camera floor | Proposal scale | Curvature cap | Result | SSE |
|---:|---:|---:|---|---:|
| `1e-48` | `0.6` | `64` | I90 completed | **3,270,255.309** |
| `1e-48` | `0.7` | `64` | recovery exhausted at I29 | 4,039,933.729 |
| `1e-48` | `0.8` | `64` | recovery exhausted at I27 | 3,954,872.151 |
| `3e-10` | `0.6` | `128` | recovery exhausted at I83 | 3,904,233.085 |
| `3e-10` | `0.6` | `256` | I90 completed | 3,909,069.271 |

Raising the curvature cap prevents the floor-`3e-10` early stop but does not
repair its poor basin. The universal BAL configuration should therefore retain
camera floor `1e-48` and proposal scale `0.6`; the `3e-10`/`0.8` quality pair is
specific to 1DSfM.

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
BUNDLE_PALM_MAXIMUM_TRUST_REGION_RADIUS=1e4
BUNDLE_PALM_THREADS_PER_CLUSTER=1
```

Together with the request configuration above and coordinator block
regularization `1e-4`, these reproduce the selected parameter family. Ports are
deployment details and may be changed without changing solver behavior.

## Maximum-Radius Study

Maximum trust radius `3e3` was initially promoted from the selected hard-five
study. On that cohort it reaches
`0.966527x` baseline SSE at I200 and `0.959595x` at I1000, with 4/5 wins at
both horizons. On the frozen ten-scene BAL cohort it reaches `0.999640x` SSE
with a worst ratio of `1.004562`. Across all 15 1DSfM scenes at I1000 it reaches
`0.980749x` the max-`1e6` baseline with 13 wins and 2 losses, reducing
rejections from 1,540 to 994 and fallbacks from 815 to 624. Yorkminster
(`1.082109x`) and NYC Library (`1.052318x`) remain explicit regressions.

The later full I200 comparison favors `1e4`: cap `3e3` is `1.017034x` on 14
matched 1DSfM scenes and reproducibly stops Piazza at I150, while `1e4`
completes all 15. On all 29 BAL scenes, `3e3/1e4` is `1.000501x`, effectively
tied. Therefore `1e4` is the current universal method value and `3e3` remains
an aggressive-cap ablation.