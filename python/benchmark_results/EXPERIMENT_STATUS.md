# Experiment Status

Last refreshed: 2026-07-27

## Live Status

**No experiment is currently running.**

The common DABA-versus-ours partition audit is complete for all five scenes at
K10/K20/K30. DABA's default mode completed 13/15 cases; scene 52 at K20/K30
requires its distinct `memory_efficient=true` mode. Across all 15 cases, DABA
partitions 6.57x faster and reduces camera replication, but induces 41.45% more
local observation work, 44.21x worse observation-load CV, and 316x more
degree-<5 camera incidences under the common local-problem accounting.

- [DABA partition comparison](daba_partition_comparison/report.md)

The official 1DSfM numerical archive has been downloaded and integrity-checked,
but it is not ready for a DABA Table II BA run. Its `gt_bundle.out` files are
comparison reconstructions whose camera/point graphs do not match the table;
the dense track graph still needs the undocumented initialization, filtering,
and triangulation path used to create DABA's inputs.

Decision: proceed with a separately named **SfM_Init-derived 1DSfM** benchmark
for paper external validity. Preserve the pinned original cleanup semantics,
add deterministic triangulation and explicit provenance, and start with Union
Square. These rows may compare methods on the same generated files but must not
be labeled DABA Table II reproduction.

The Union Square pilot now passes the end-to-end gate. The frozen conversion
produces 768 cameras, 82,614 points, and 265,464 observations with initial mean
pixel error 11.61 px and a 99.95 px maximum. Ceres reaches 2.266 px mean in
3.59 s; full-block DRS K10/30 reaches 2.349 px in 10.23 s with exact saved-state
evaluation. This validates external use of the generated dataset but is not a
runtime win or DABA Table II reproduction.

- [1DSfM provenance and readiness audit](1dsfm_provenance_audit.md)
- [Union Square pilot report](sfm_init_1dsfm/union_square/report.md)

The 29-scene K10/K20/K30 safeguarded outer Nesterov matrix completed all 174
cases. Every saved state independently reproduces its recorded pixel SSE.

| K | Control geomean best SSE | Accelerated | Paired wins/ties/losses | Stable endpoints accel/control |
|---:|---:|---:|---:|---:|
| 10 | 1,726,992 | **1,633,145** | 23/3/3 | 25/25 |
| 20 | 2,346,179 | **2,144,125** | 23/4/2 | 25/25 |
| 30 | 1,866,188 | **1,681,732** | 24/2/3 | 26/25 |

Acceleration wins 70/87 best-SSE comparisons, ties 9, and loses 8, with 26.6%
runtime overhead. It does not eliminate the baseline failure cohort: 646, 931,
1064, 1266, and 1723 remain the primary stability targets.

- [29-scene acceleration report](admm_linesearch_overnight_i30_k10_k20_k30/report.md)
- resumable command: `serverTest/run_admm_linesearch_overnight.sh`

The matched five-scene pixel-versus-DABA-ray objective ablation completed all
10 cases with full block consensus fixed.

| Optimized objective | Geomean mean px | Geomean pixel RMSE | Hard rejections |
|---|---:|---:|---:|
| Pixel | **0.6587** | **1.1433** | 12 |
| DABA ray | 0.6750 | 1.5473 | **4** |

Ray optimization is easier to stabilize but is 2.48% worse in aggregate mean
pixel error and 35.34% worse in pixel RMSE. Scene 245 has an `889.7 px` maximum
despite better median/p90/p95, showing that mismatch is concentrated in severe
pixel outliers. Ray 1723 also has 14 noninvertible observations.

- [Matched objective report](drs_objective_five_scene_k10_i30/report.md)

The same objective ablation is complete on the four previously untested ADMM
failure scenes 646, 931, 1064, and 1266. With mature full-block DRS fixed, ray
optimization reduces hard restorations from 11 to 9, but is 1.36% worse in
geomean mean pixel error and 2.07% worse in pixel RMSE. All eight cases finish
with finite trajectories and no ray inverse-projection failures.

- [Failure-cohort objective report](drs_objective_failure_cohort_k10_i30/report.md)

The five-scene K10/30 DRS consensus projection breadth gate completed all 20
cases. Full `9x9` blocks have the lowest pixel SSE on every scene.

| Projection | Geomean SSE | Relative to full | Hard rejections |
|---|---:|---:|---:|
| Arithmetic | 4,630,080 | +330.1% | 52 |
| Scalar | 1,209,725 | +12.36% | 20 |
| Diagonal | 1,148,814 | +6.70% | 15 |
| Full block | **1,076,627** | reference | **14** |

- [Five-scene projection report](drs_consensus_metric_five_scene_k10_i30/report.md)

The Ladybug-1723/K10/90 DRS consensus projection ablation completed.

| Projection metric | Best pixel SSE | Hard rejections | Final `Be` | Outcome |
|---|---:|---:|---:|---|
| Arithmetic | 124,050,155 | 19 | 0.5 | unstable; saved initial best |
| Scalar per copy | 866,306 | 14 | 0.5 | stable but plateaued |
| Diagonal | 805,805 | 8 | 0.0128 | stable |
| Full `9x9` block | **764,872** | **7** | **0.0064** | best |

- [Projection metric report](drs_consensus_metric_1723_k10_i90/report.md)
- [Paper-ready derivation](../block_metric_consensus_derivation.md)

The K10/K30 partition-stress comparison completed all 20 A1/A2 cases.

| Recovery | K | Geomean best SSE | Geomean endpoint SSE | Recoveries | Total time s |
|---|---:|---:|---:|---:|---:|
| A1: double ADMM penalty | 10 | 3,025,954 | 24,722,276 | 0 | 97.30 |
| A2: temporary proximal majorizer | 10 | 3,025,954 | 24,722,276 | 0 | 97.49 |
| A1: double ADMM penalty | 30 | 2,578,232 | 2,674,900 | 0 | 102.47 |
| A2: temporary proximal majorizer | 30 | 2,578,232 | 2,674,900 | 0 | 101.92 |

A1 and A2 are identical because the `1e6` one-step guard never fires. On
1723/K10, pixel SSE leaves the good basin at iteration 2 by `6.99e5` relative
to the best state, then consensus residuals converge toward zero around a bad
state; endpoint SSE is `7.85e11`. This is gradual basin loss, not a single
greater-than-`1e6` jump.

- [K10/K30 stress report](admm_recovery_partition_stress_k10_k30/report.md)
- [K10/K30 raw status](admm_recovery_partition_stress_k10_k30/status.tsv)
- [Stability diagnosis and DABA comparison](../global_safeguard_design.md)

The five-scene K20/30 global recovery ablation completed all 15 cases.

| Recovery | Geomean best SSE | Geomean endpoint SSE | Recoveries | Total time s | Gate |
|---|---:|---:|---:|---:|---|
| A1: double ADMM penalty | 3,110,876 | 3,518,342 | 1 | 99.47 | pass |
| A2: temporary proximal majorizer | 3,110,876 | 3,805,414 | 1 | 100.28 | pass |
| A3: persistent trust radius | 3,110,876 | 5,360,211 | 10 | 99.84 | fail |

A1 and A2 reproduced exactly the same best SSE and best iteration on all five
scenes. A2 activated only on 1723, remained positive for four recorded
iterations, and decayed back to zero. Its 1723 endpoint was `45,652,681` versus
A1's `30,842,287`. A3 accumulated ten 1723 recoveries and therefore fails the
predeclared fewer-than-five recovery gate.

- [Recovery design and partition plan](../global_safeguard_design.md)
- [Stage 1 report](admm_recovery_ablation_i30_k20/report.md)
- [Raw Stage 1 status](admm_recovery_ablation_i30_k20/status.tsv)

The repeated five-scene K20/30 guard comparison completed all 20 cases. One
guarded Nesterov scene-245 process hit a protobuf/NumPy buffer-ownership bug;
after changing reply conversion to an owning copy, its isolated rerun completed
with the same expected SSE and zero recoveries.

- [Complete guard/Ceres interpretation](admm_guard_ceres_comparison/analysis.md)
- [Machine-generated 19-row report](admm_guard_ceres_comparison/report.md)
- [Repaired guarded scene-245 row](admm_guard_245_rerun/nesterov_daba_tr_catastrophic_guard.jsonl)

The five-scene catastrophic-only recovery gate completed all 5 rows.

## Outer Recovery

Status: **passed, 5/5 rows**

The Nesterov + DABA trust-region path now restores the previous coordinator and
worker state only when candidate pixel SSE is nonfinite or grows by more than
`1e6` in one outer iteration. Recovery doubles the ADMM penalty. The policy
triggered once on 1723 and zero times on the other four scenes across 150 total
outer iterations.

| Dataset | Unguarded best SSE | Guarded best SSE | Unguarded endpoint | Guarded endpoint | Recoveries |
|---|---:|---:|---:|---:|---:|
| 1723 | 16,667,483 | 16,667,483 | 3.68e19 | 30,842,287 | 1 |
| 52 | 1,791,898 | 1,791,898 | 1,791,898 | 1,791,898 | 0 |
| 245 | 2,906,069 | 2,906,069 | 2,906,069 | 2,906,069 | 0 |
| 394 | 724,074 | 724,074 | 724,074 | 724,074 | 0 |
| 871 | 4,635,978 | 4,635,978 | 4,635,978 | 4,635,978 | 0 |

Best SSE and best iteration are identical to the unguarded path on all five
scenes. Runtime is also unchanged within run noise. The earlier `1.1x` and
`2x` guards were rejected because they suppressed a useful initial ADMM
transient and restarted on every 1723 iteration.

- [Five-scene recovery report](admm_catastrophic_guard_five_scene/report.md)
- [Raw recovery results](admm_catastrophic_guard_five_scene/nesterov_daba_tr_catastrophic_guard.jsonl)
- [Per-case status, time, and RSS](admm_catastrophic_guard_five_scene/status.tsv)

The five-scene solver/trust-region matrix completed all 25 rows. Every saved
physical state reproduces its recorded standard pixel SSE exactly.

## Current Main Result

### Five-scene K20 ADMM ablation

Status: **complete, 15/15 rows**

Protocol:

- datasets: 1723, 52, 245, 394, 871
- 20 clusters
- 30 outer iterations
- exactly one local Ceres LM iteration per worker and outer iteration
- one CPU thread per cluster solve
- standard Snavely pixel reprojection SSE

Readable report:

- [Five-scene K20 report](admm_five_scene_i30_k20/report.md)

Raw results:

- [Raw baseline](admm_five_scene_i30_k20/baseline.jsonl)
- [Raw Jacobi + alpha=1](admm_five_scene_i30_k20/diagnostic_jacobi_alpha1.jsonl)
- [Raw split-penalty adaptation](admm_five_scene_i30_k20/daba_split_penalty_adaptation.jsonl)
- [Per-case status, time, and RSS](admm_five_scene_i30_k20/status.tsv)

Summary:

| Dataset | Raw baseline best SSE | Jacobi best SSE | Split-adapt best SSE | Split endpoint / best |
|---|---:|---:|---:|---:|
| 1723 | 124,050,155 | 2,805,429 | 2,805,429 | 1.002 |
| 52 | 2,111,382 | 1,840,911 | 1,840,502 | 1.000 |
| 245 | 3,390,104 | 3,079,341 | 3,079,341 | 1.000 |
| 394 | 2,695,707 | 720,582 | 720,580 | 1.000 |
| 871 | 4,703,812 | 4,933,865 | 4,933,865 | 1.010 |

The 6+3 shared-adaptation representation was verified as an algebraic identity.
Independent extrinsic/intrinsic adaptation changed final penalty ratios but had
negligible effect on best pixel SSE.

Stability gate: **failed**. Ladybug-1723 still experiences a catastrophic
transient current-state excursion before recovering. The experiment was not
expanded to ten scenes.

## DRS Recovery

Status: **recovered and validated**

- [DRS recovery provenance and commands](drs1723_recovery.md)
- [Recovered 30-iteration result](drs1723_recovery_probe/results.jsonl)
- [Recovered physical state](drs1723_recovery_probe/state.npz)
- [Best prior K30 atomic result](ours_k30_atomic/results.jsonl)
- [Best prior K30 atomic state](ours_k30_atomic/states/1723_k30_i90.npz)

Recovered 30-iteration Ladybug-1723 result:

- best pixel SSE: 768,241
- best iteration: 26
- mean reprojection error: 0.76055 px
- native/common evaluator relative disagreement: 6.54e-9

The main regression was traced to explicit coordinator landmark buffers during
trial-cost evaluation. Historical trajectory behavior is restored with:

```text
BUNDLE_PALM_EXPLICIT_COST_LANDMARKS=0
BUNDLE_PALM_STRICT_TRIAL_SAFEGUARD=0
BUNDLE_PALM_REQUIRE_COMMON_COST_MATCH=1
```

## Supporting Diagnostics

- [K20 local-work comparison](admm_local_work_k20_i12/report.md)
- [K20 30-iteration recovery check](admm_recovery_k20_i30/report.md)
- [K20 stability sweep](admm_stability_k20_i12/report.md)
- [Jacobi scaling-cap sweep](admm_scaling_cap_k20_i12/report.md)
- [Initial-penalty sweep](admm_rho_scale_k20_i12/report.md)
- [Refined initial-penalty sweep](admm_rho_refine_k20_i12/report.md)
- [Split adaptation short comparison](admm_split_adaptation_k20_i12/report.md)

## Next Experiment

Status: **complete, 25/25 rows**

Results:

- [Readable solver/trust-region report](admm_solver_trust_i30_k20/report.md)
- [Per-case status, time, and RSS](admm_solver_trust_i30_k20/status.tsv)
- [Ceres reference trajectories](admm_solver_trust_i30_k20/local_ceres_reference.jsonl)
- [PCG + DABA TR trajectories](admm_solver_trust_i30_k20/pcg_daba_tr.jsonl)
- [Nesterov + DABA TR trajectories](admm_solver_trust_i30_k20/nesterov_daba_tr.jsonl)
- [PCG + DRS TR trajectories](admm_solver_trust_i30_k20/pcg_drs_tr.jsonl)
- [Nesterov + DRS TR trajectories](admm_solver_trust_i30_k20/nesterov_drs_tr.jsonl)

Local linear solver × trust-region policy:

| Variant | Linear solver | Trust-region policy | Status |
|---|---|---|---|
| PCG + DABA TR | custom Schur-PCG | DABA LM acceptance/radius | complete |
| Nesterov + DABA TR | Schur-Nesterov | DABA LM acceptance/radius | complete |
| PCG + DRS TR | custom Schur-PCG | existing DRS retry/radius | complete |
| Nesterov + DRS TR | Schur-Nesterov | existing DRS retry/radius | complete |
| Ceres reference | Ceres iterative Schur | Ceres internal LM | available as contextual reference |

Aggregate result: Ceres one-step iterative Schur plus Ceres LM is the only path
with endpoint/best <= 1.05 on all five scenes and has the best geometric-mean
pixel SSE. Custom solvers are competitive and faster on 52, 245, 394, and 871,
but all fail on 1723. DABA TR improves Nesterov on 1723 relative to DRS TR
(`16.7M` versus `52.3M` best SSE), but still diverges at the endpoint. The next
work must diagnose the custom damped quadratic step on 1723 rather than expand
the matrix.

### Solver/trust numbers

| Variant | Geomean best pixel SSE | Endpoint stable | Geomean time s |
|---|---:|---:|---:|
| Ceres Schur-PCG + Ceres LM | **2,241,156** | **5/5** | 19.02 |
| Nesterov + DABA TR | 3,110,876 | 4/5 | 13.95 |
| PCG + DRS TR | 3,376,614 | 4/5 | **13.49** |
| Nesterov + DRS TR | 3,935,580 | 4/5 | 14.10 |
| PCG + DABA TR | 4,729,922 | 4/5 | 14.22 |

Ladybug-1723:

| Variant | Best pixel SSE | Best iteration | Endpoint / best |
|---|---:|---:|---:|
| Ceres Schur-PCG + Ceres LM | **2,805,429** | 27 | 1.00 |
| Nesterov + DABA TR | 16,667,483 | 1 | 2.21e12 |
| PCG + DRS TR | 22,158,530 | 1 | 5.63e4 |
| Nesterov + DRS TR | 52,281,868 | 0 | 1.27e6 |
| PCG + DABA TR | 124,050,155 | -1 | 16.08 |

### Weak-camera diagnosis on 1723

The shared K20 partition contains the following camera-cluster incidences:

```text
Weak camera-cluster counts 1..19:
2 7 11 7 12 10 11 14 7 116 58 56 68 54 53 45 51 53 28
Additional camera copies: 6146
Maximum cameras in a cluster: 711
```

There are 27 incidences with only 1--4 landmarks (`2 + 7 + 11 + 7`). Each
camera has nine unknowns but at most `2m` scalar reprojection residuals for `m`
landmarks, so all 27 blocks are rank-deficient by row count before proximal and
trust-region regularization.

A one-outer-iteration local diagnostic over all 20 clusters found:

- 7,869 camera blocks;
- minimum camera diagonal / block maximum: `4.27467e-15`;
- configured relative camera-diagonal floor: `1e-48`;
- zero diagonal entries floored: `0`;
- custom PCG DABA gain-ratio range: `6.73` to `4.51e8`;
- custom Nesterov DABA gain-ratio range: `0.623` to `0.954`;
- first global pixel SSE: `3.03e11` (PCG) and `2.71e11` (Nesterov).

This strongly implicates weak/rank-deficient camera blocks in the custom-path
failure. The current `1e-48` floor is numerically inactive, and custom code
directly inverts full 9x9 block preconditioners. Ceres LM handles the same
partition much more robustly. Probe artifacts:

- [Weak-camera probe status](admm_weak_camera_probe/status.tsv)
- [PCG probe trajectory](admm_weak_camera_probe/pcg_daba_tr.jsonl)
- [Nesterov probe trajectory](admm_weak_camera_probe/nesterov_daba_tr.jsonl)

## Durable Research Log

- [Full experiment log](experiment_log.md)
- [ADMM/DRS implementation plan](../admm_drs_baseline_plan.md)
- [DABA-style ADMM notes](../towards_daba_admm.md)
