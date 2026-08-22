# K24 DRS/Schur Direction-Alignment Diagnostic

Date: 2026-08-21
Status: safeguarded I90 proposal passed; seven-scene transfer next
Source checkpoint: `d326283`

This is the restart contract for the behavior-neutral diagnostic following the
integrated K24/I200 quality ceiling. It does not alter DRS proposals or apply
mid-run corrections.

## Question

At late iterations, is ordinary DRS merely taking a smaller version of the
coordinated global Schur direction, or is it moving in a materially different
camera direction/model geometry?

The diagnostic compares, before ordinary iterations I90, I120, I160, and I200:

- reference: the frozen low-damping global Schur camera tangent;
- candidate: the nominal accepted DRS consensus camera tangent for that
  iteration.

## Frozen Configuration

Use the integrated mature K24/I200 configuration and final one-correction policy
from `K24_I200_QUALITY_PLAN.md`. Alignment uses the actual correction settings:

- camera/landmark damping `0.005859375`;
- `bsr_low_memory`, Jacobi PCG, tolerance `1e-6`, maximum 500 iterations;
- no Schur-model clipping or any other behavior-changing option.

The first breadth pass showed that BAL3068 reaches tolerance at I90 but hits
the 500-iteration diagnostic ceiling at I120/I160/I200. Recover only that scene
with a diagnostic-only 5000-iteration ceiling. The terminal correction remains
capped at 500, so this recovery cannot alter solver behavior.

The diagnostic must reproduce the reference I200 trajectory, correction
acceptance, endpoint cameras, and endpoint points exactly.

## Cohort

- Roman Forum and Trafalgar: original K1/mature sentinels;
- Montreal Notre Dame and Yorkminster: corrected-I200 losses;
- Piazza del Popolo: recovery exhaustion at I188;
- BAL52 and BAL3068: stable cross-family controls.

No scene is used to select a scalar setting.

## Telemetry

At every available checkpoint report:

- all-camera and shared-camera cosine;
- diagonal-weighted all/shared cosine;
- DRS/Schur tangent norm ratio;
- translation, rotation, and intrinsics cosine/norm ratios;
- gradient-action ratio;
- camera-only damped model reduction split into shared and unique camera
  components, with a shared-DRS/shared-Schur ratio;
- similarity-gauge fraction and quotient-space alignment/model action;
- metric-invariant coherence of the reflected copy votes entering each shared
  camera consensus projection;
- Schur PCG termination, iterations, residual, and solve time.

Piazza is expected to have no I200 checkpoint because its ordinary trajectory
ends at I188. This is reported, not repaired.

## Gate

The run is valid only when:

- all diagnostic rows reproduce their reference trajectories and final states;
- every available diagnostic Schur solve terminates successfully below `1e-6`;
- the diagnostic field is present at every checkpoint reached by the reference;
- no clipping or correction is applied at the diagnostic checkpoints.

This experiment is descriptive. It must name a falsifiable mechanism before any
new solver behavior is implemented. Do not tune damping, checkpoint times, or
subspace scales from these seven scenes.

## Crash Recovery

Artifacts live under:

```text
benchmark_results/k24_schur_alignment_i200/
  diagnostic/
  summary.json
  report.md
```

Use `OVERWRITE=0`; keep logs, memory traces, NPZ states, and absolute manifests
untracked. Commit the runner/analyzer/plan before launch.

## Immediate Next Action

The seven-scene diagnostic passes exact trajectory/state neutrality and all
available Schur solves converge below `1e-6`. Its late shared-camera weighted
cosine is only `0.022--0.050` in median, and median shared DRS model gain is
only `0.000007--0.000645x` the shared Schur gain. Unique-camera motion has a
nonpositive camera model in every row. BAL3068 additionally has
`1e17--1e18` shared DRS/Schur norm ratios and effectively zero cosine.

Similarity-gauge drift is falsified. Median DRS gauge fraction is
`1.58e-9` at I90 and falls to `1.93e-13` at I200. Quotient projection leaves
the median shared cosine (`0.0503 -> 0.0505` at I90,
`0.0486 -> 0.0488` at I200) and norm ratio unchanged. BAL3068 also has gauge
fractions below `1.1e-15`, so its `1e17--1e18` norm anomaly is not similarity
gauge motion.

Reflected-copy cancellation is real but not the discriminating mechanism.
Median global coherence is only `0.0199--0.0310`, but across all 27 rows its
Spearman correlation is only `0.190` with shared Schur cosine and `0.223` with
shared model gain; excluding anomalous BAL3068, both correlations are slightly
negative. BAL52 is the direct counterexample: it has lower coherence than most
1DSfM rows while retaining much stronger Schur alignment and model gain.

The Roman/BAL52 sentinel closes the copy-level mechanism. Raw-copy global
cosines are near zero in both. Exact metric-contribution cosines are also near
zero, with signed action balance `0.0068--0.1091` on Roman and
`0.0191--0.0602` on BAL52. Median best-copy cosines overlap. Projection improves
both directions, so neither erased coherent copy signal nor incorrect copy
weighting explains why projected BAL52 aligns much better than Roman. See
`benchmark_results/k24_schur_contribution_alignment_sentinel/report.md`.

The projected-camera sentinel shows broad misalignment rather than a small
outlier set. Roman has median camera cosine `0.21--0.28`, only `71--75%`
positive cameras, and signed action balance `0.42--0.80`. BAL52 has median
cosine `0.39--0.59`, `81--100%` positive cameras, and action balance
`0.96--1.00`. See
`benchmark_results/k24_schur_camera_distribution_sentinel/report.md`.

The behavior-neutral coupled-consensus oracle fails decisively. Its norm is
`337--1249x` Schur on Roman and `275--26512x` on BAL52, with strongly negative
camera-model gain everywhere. See
`benchmark_results/k24_coupled_consensus_oracle_sentinel/report.md`.

This rejects projection-only coupling, not cross-camera curvature. The local
copies were generated under the block metric, so projecting them under a new
coupled metric violates product-space DRS consistency. The next matched solver
gate must use one metric in both local prox and consensus. Restore the verified
transient Frobenius-majorizer worker dispatch currently blocked by the recovered
backend's explicit `NotImplementedError`, then compose its frozen threshold
`0.55`, scale `0.5`, I2--I10 policy with the mature C1/Nesterov Roman/BAL52
backbone. Do not replace DRS with Ceres or a global Schur step.

That restoration is complete and default-off neutral. The frozen majorizer is
`0.993173x` Roman at I30 but reverses to `1.007679x` at I90 with 10 versus 7
rejections. BAL52 declines the selector and is bitwise exact. Close this metric
proxy without tuning. See
`benchmark_results/schur_majorizer_mature_i90/report.md`.

Next compute, but do not apply, one Jacobi-preconditioned residual correction
of the nominal shared DRS tangent under the same global Schur system used by the
terminal correction. If one operator action materially improves Roman cosine
and model gain while preserving BAL52, implement it later as a safeguarded DRS
acceleration proposal. Failure closes one-step curvature transport and rules
out another approximate metric.

The one-step oracle passes. At I90/I120 Roman cosine improves from
`0.063/0.134` to `0.763/0.775` and captures `0.881/0.888` of Schur camera-model
gain. BAL52 improves from `0.555/0.412` to `0.929/0.950` and captures
`0.875/0.969`. Trajectory prefixes remain exact. See
`benchmark_results/k24_one_step_schur_oracle_sentinel/report.md`.

Next add one default-off I90 proposal. Evaluate precise worker SSE for ordinary
and one-step consensus candidates; select one-step only when lower, then rebuild
centers and residuals atomically with `drs_state_for_consensus`. Permit one
proposal maximum. Keep the full Schur solve diagnostic-only; the applied
proposal itself uses one matvec and block-Jacobi correction. Roman and BAL52 are
the first gate, with no scene routing.

The applied sentinel passes with fixed geometric SSE backtracking. Roman selects
scale `0.5`, reaches `0.888157x` control immediately and `0.797016x` delivered
at I120, with 8 versus 11 rejections. BAL52 selects `0.0625` and remains
effectively neutral (`0.9999995x`) with no rejection. See
`benchmark_results/k24_one_step_schur_proposal_i90_sentinel/report.md`.

Next run the unchanged I90 proposal through I120 on the frozen seven-scene
direction cohort. Report every selection scale, immediate and delivered ratio,
completion, and loss. Do not tune the eight geometric scales, damping, or
proposal time before this transfer.

That transfer passes. All five 1DSfM scenes improve at I120, with geometric
ratio `0.876147` and W/L `5/0`. BAL52 is effectively neutral; BAL3068 declines
the proposal. All seven complete. See
`benchmark_results/k24_one_step_schur_proposal_i90_cohort/report.md`.

Crash boundary: the first BAL3068 attempt exited WSL while the rejected coupled
consensus oracle was still enabled inside alignment telemetry. That oracle is
now explicit opt-in. Recovered BAL3068 completed alone under a 14 GiB process
ceiling at `8,358,544 KiB` coordinator RSS. Never enable the coupled oracle in
proposal breadth. Before all-15/all-29 transfer, add a resumable low-memory
runner/analyzer that serializes large BAL cases with an explicit ceiling.

The low-memory all-15 run initially exposed product-state incompatibility:
several live trajectories exploded after I90. Canonical restart at the selected
consensus fixes that failure. The stable all-15 result is `0.887673x` geometric
and `0.861948x` summed, W/T/L `12/1/2`; Gendarmenmarkt loses `1.058521x` and
Tower `1.011275x`. The strict no-loss gate fails, so do not promote or tune.
Proceed only with unchanged serial BAL safety transfer under the memory cap,
consistent with the repository's bounded-loss component policy. See
`benchmark_results/k24_one_step_schur_proposal_i90_breadth/report.md`.

The serial all-29 BAL safety transfer is complete. All 29 proposals decline,
yielding exact control trajectories and W/T/L `0/29/0`. BAL961 exposed one
remaining diagnostic-only memory dependency: proposal execution still invoked
the full reference PCG solve and failed locally at the 14 GiB ceiling. The
proposal checkpoint now skips that solve, gauge/copy telemetry, and global
model evaluation; it retains only Schur-system construction, one block-Jacobi
residual action, and eight fixed worker-SSE trials. Recovered BAL961 completed
at `7,187,668/8,902,156 KiB` coordinator/worker RSS. This closes BAL safety but
not promotion because BAL receives no gain and the 1DSfM no-loss gate fails.

The post-selection state-consistency gate is also complete. Both prior losses
showed immediate I90 gains followed by extra rejections. A selected-only I91
trust/curvature/acceleration rebase reduces all-15 delivered/control from
`0.887673x` to `0.884652x` geometrically and changes W/T/L from `12/1/2` to
`13/1/1`, with exact I1--I90 prefixes and unchanged proposal scales. Tower is
now `0.999139x`; Gendarmenmarkt remains `1.010662x`. BAL proposals all decline,
so this mechanism is inactive there; BAL52 is bitwise exact. Retain the
rebase-enhanced proposal as a research component, close rebase timing/radius
tuning, and do not promote while the strict no-loss gate still fails. See
`benchmark_results/k24_one_step_schur_proposal_i90_trust_rebase_breadth/report.md`.

Fixed stationary depth two is closed by a behavior-neutral sentinel. The
second unit Jacobi action worsens actual candidate SSE by `1.000710x` on
Gendarmenmarkt and `1.008820x` on Roman relative to depth one, while residual
norms expand `4.92x` and `238x`. BAL52 residual contracts `0.845x` but neither
depth passes the SSE floor. All trajectories and endpoint states are exact.
Do not apply or broaden depth two and do not tune stationary relaxation. See
`benchmark_results/k24_two_step_schur_residual_oracle_sentinel/report.md`.

Exact model line minimization does not rescue the second action. The
Gendarmenmarkt/Roman coefficients are `0.964/1.075`; frozen-model decrease
improves, but physical worker SSE remains `1.000677x/1.010264x` depth one.
BAL52 still declines. Trajectory/state neutrality is exact. Therefore the
second direction, not a materially wrong unit coefficient, fails transfer.
Close further fixed local Schur iteration and keep the one-action proposal.
See
`benchmark_results/k24_model_optimal_two_step_schur_oracle_sentinel/report.md`.

Relinearization at the selected depth-one state is also closed. The best fresh
Gendarmenmarkt action improves only `0.0455%`, below the fixed `0.1%` floor;
Roman has no descent through scale `1/128`. Full fresh steps regress
`2.62%/19.42%`, and BAL52 skips the rebuild after declining depth one. Exact
trajectory/state neutrality holds. Do not broaden or lower the floor. See
`benchmark_results/k24_relinearized_second_schur_oracle_sentinel/report.md`.

Continuation policy is frozen as well. Best-state-only delivery gives a safe
but weaker `0.934688x`. The exact ordinary/proposal endpoint race ceiling is
`0.884026x`, only `0.071%` beyond retained continuation, and removes only the
Gendarmenmarkt loss at the cost of duplicated I91--I120 and missing full DRS
snapshot infrastructure. Do not implement a shadow branch. See
`benchmark_results/k24_schur_proposal_continuation_policy_oracles/report.md`.

Full-camera residual correction is closed. It worsens the shared-only
candidate by `1.001933x` on Gendarmenmarkt and `1.127536x` on Roman, with
unique-inclusive correction norms `6.14e13/2.16e11`; BAL52 still declines.
Exact trajectory/state neutrality holds. Keep unique cameras locally owned and
restrict residual repair to shared cameras. See
`benchmark_results/k24_all_camera_schur_residual_oracle_sentinel/report.md`.

Consistent eliminated-landmark response passes the frozen transfer. The
existing three fixed-camera landmark steps select scale `1.0` on all five
1DSfM scenes and improve geometric delivered/control from `0.822388x` to
`0.691764x`, W/T/L `5/0/0`. BAL52/3068 decline exactly. This identifies the
missing Schur contract: camera trial evaluation and committed state must include
the marginalized landmark response. Proceed unchanged to all-15, then BAL29.
See `benchmark_results/k24_schur_proposal_landmark_response_cohort/report.md`.

All-15 transfer passes the strict gate unchanged: geometric/summed
delivered/control `0.790889x/0.760039x`, W/T/L `14/1/0`; 14 scenes select scale
`1.0` and Madrid declines exactly. Candidate/Ceres is `1.177568x`, with all
scenes complete. Proceed unchanged to serialized BAL29 under the existing
memory ceiling. See
`benchmark_results/k24_schur_proposal_landmark_response_breadth/report.md`.

The frozen one-terminal-correction composition passes the four-scene sentinel:
incremental correction/raw `0.963498x` on Roman/Trafalgar and `0.999059x` on
BAL52/3068, with all solves converged and accepted. Expand unchanged to breadth
as a separately labeled polish, not a DRS mechanism. See
`benchmark_results/k24_landmark_response_terminal_correction_sentinel/report.md`.

Polished breadth is complete with bounded low-memory transpose assembly.
Correction/handoff is `0.964507x` on all-15 and `0.999642x` on BAL29;
corrected/Ceres is `1.135772x` and `0.998681x`. All 15 1DSfM corrections accept;
BAL correction W/T/L is `26/3/0`, with exact no-op for nonconverged solves.
Peak BAL RSS is `8.306/12.051 GiB`. Retain one correction as separate polish;
the remaining 1DSfM gap is upstream trajectory/basin quality. See
`benchmark_results/k24_landmark_response_terminal_correction_breadth/report.md`.

Behavior-neutral accepted-landmark diagnostics at I30/I60/I89 show modest
ordinary-candidate descent but no primal safeguard flips on Roman, Trafalgar,
or BAL52; trajectories/states and worker round trips are exact. Stale landmark
scoring is not the DRS rejection cause. Keep legacy per-iteration refinement
off. The sole bounded timing follow-up is the previously frozen I60 checkpoint
with the joint camera/landmark proposal. See
`benchmark_results/k24_drs_candidate_landmark_response_accepted_diagnostic/report.md`.

The frozen alternate I60 checkpoint passes the four-scene safety gate and is
slightly aggregate-positive on Roman/Trafalgar (`0.761715x` versus I90
`0.762678x`); BAL52/3068 decline exactly. Run unchanged all-15 once and close
timing afterward. See
`benchmark_results/k24_landmark_response_proposal_i60_sentinel_actual/report.md`.

All-15 confirms I60 as the aggregate-better frozen checkpoint:
geometric/summed control `0.786533x/0.754474x`, W/T/L `14/1/0`, versus I90
`0.790889x/0.760039x`. Candidate/Ceres is `1.171081x`. Close timing and run
unchanged BAL29 once. See
`benchmark_results/k24_landmark_response_proposal_i60_breadth/report.md`.

I60 plus one terminal correction is mixed and `1.003848x` the I90-polished
Roman/Trafalgar pair; BAL is exactly unchanged. Close this composition without
breadth. Keep I60 for DRS and I90 for the separate polished scoreboard. See
`benchmark_results/k24_i60_landmark_response_terminal1_sentinel/report.md`.

The explicit two-proposal I60+I90 gate improves Roman/Trafalgar geomean from
`0.761715x` to `0.748524x`; BAL52/3068 decline both proposals exactly. Transfer
unchanged all-15 once and close proposal count/checkpoint additions. See
`benchmark_results/k24_landmark_response_proposal_i60_i90_sentinel/report.md`.

All-15 I60+I90 reaches `0.766084x` control but loses Gendarmenmarkt
`1.019557x`, W/T/L `13/1/1`; weak second gains create extra rejections. Keep as
bounded-loss evidence, do not tune the floor, and test only suppressing the
second trust rebase. See
`benchmark_results/k24_landmark_response_proposal_i60_i90_breadth/report.md`.

BAL29 closes timing: all 29 I60 proposals decline exactly, while all-15 is
`0.786533x` control and `1.171081x` Ceres with no losses. Promote I60 as the
common checkpoint and freeze all timing/scale/depth/floor/restart settings. See
`benchmark_results/k24_landmark_response_proposal_i60_breadth/report.md`.

The serial BAL29 gate passes unchanged: all 29 proposals decline exactly,
W/T/L `0/29/0`, with `6.873/9.689 GiB` maximum coordinator/worker RSS. Together
with all-15 `0.790889x`, W/T/L `14/1/0`, this promotes the common no-loss
landmark-response proposal component. Keep the frozen I90 timing, one shared
residual action, eight scales, three landmark steps, `1e-3` floor, atomic
camera/landmark commit, canonical restart, and I91 trust rebase. See
`benchmark_results/k24_schur_proposal_landmark_response_breadth/report.md`.
