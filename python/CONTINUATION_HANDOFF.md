# BundlePalm DRS Paper Handoff

Updated: 2026-08-19

## 2026-08-19 Plain-Resolvent Horizon Addendum

An additional all-15 long-horizon ablation is complete. It was run from a
clean detached `4db26e4` worktree because the main worktree contains unrelated
in-progress protocol/evaluator changes. Sixty-nine focused tests passed and a
Roman smoke reproduced the historical I90 event counts before breadth.

The frozen K24 pure-DRS tuple includes direct left-SE3 tangent equations,
exact tangent metric consistency, shared-only camera proximal semantics,
points-p95/Jacobi/preconditioning, and one finite Nesterov local step. C1, C5,
bootstrap, product-SO3, and final Schur correction are disabled.

Twelve of 15 scenes reach I1000. Piccadilly stops at I992, Tower at I77, and
Trafalgar at I135 when curvature recovery reaches its ceiling. Delivered
all-15 quality is `1.950010x` Ceres geometrically and `1.847092x` summed, with
0/15 wins and worst Tower `5.709418x`. The 12-scene common I1000 prefix is
`1.907980x` Ceres. Best-state restoration works correctly.

This is not the missing evaluation of the complete method. Existing all-15
direct-tangent/shared-only campaigns already evaluated the later
base-backbone/C1 architecture at I90 and reach `1.591438x--1.609367x` Ceres
without final Schur polishing. The preserved legacy K24/I200 pure-DRS
reference is `1.457017x`, and separately labeled polishing reaches
`1.063839x`. This addendum only closes further horizon extension of the early
fixed one-step plain resolvent. Full report:
`benchmark_results/k1_compatible_pure_drs_k24_i1000/report.md`.

The `1.950010x` result must never be described as the effect of adding the K1
corrections to the good base: it simultaneously removes C1 and swaps the mature
DRS-trust/curvature-0.4/metric-75 backbone for K1-local tuning. In matched
factorials with direct tangent and the mature backbone held fixed, shared-only
costs only `1.033321x--1.048656x` at I30 and `1.035714x` in the complete I90
transient-proposal pair.

The exact integrated successor already exists at I90. With direct tangent,
exact metric consistency, shared-only proximal semantics, local Nesterov, DRS
trust, curvature `0.4`, metric 75, and C1, the no-proposal version reaches
`1.596631x` Ceres on all-15 1DSfM and `1.000822x` on all-29 BAL. Transient
proposal damping reaches `1.609367x`/`0.999879x`; the diagnostic rebase reaches
`1.591438x`/`1.001609x`. These are `client_drs.py` results on the mature
`client_acc`-like backbone.

Do not merge them with the preserved pre-integration quality references:
legacy K24/I200 is `1.457017x` on 1DSfM and legacy K24/I90 is `0.998404x` on
BAL, but those raw artifacts predate direct tangent and shared-only controls.
The other remembered `1.4x` row is integrated I30 plus ten distributed Schur
corrections at `1.411625x`, which is explicitly polishing and lacks a matched
all-29 ten-correction BAL evaluation. The missing clean comparison is a longer
horizon of the integrated mature backbone, not another early K1-local preset.

An isolated bridge front end now preserves the baseline implementation:
`serverTest/client_drs_k1_bridge.py`, with a resumable two-scene runner and
focused tests. Fresh Roman/Trafalgar K24/I30 shows direct tangent alone is
`1.000000140x` the fresh legacy control. Direct tangent plus shared-only is
`0.810679x` geometrically, composed of Roman `1.058889x` and Trafalgar
`0.620650x`. The experiment confirms that the K1 and mature baseline already
share the Nesterov kernel. Future isolated solver work should target interior
solve forcing/accuracy under shared-only semantics, not duplicate or retune
the Nesterov loop. Report:
`benchmark_results/k1_inner_on_mature_base_k24_i30/report.md`.

The isolated fixed-L2 follow-up is negative: it is `1.063412x` shared-only L1
at `1.584348x` time, with Roman `1.140172x` and Trafalgar `0.991819x`. Do not
fork the C++ worker for unconditional extra interior steps. Keep the isolated
front end and require a measurable relative forcing or stationarity rule before
new inner-solver implementation.

The existing interior-defect diagnostic is bitwise trajectory-neutral but does
not separate these outcomes: Roman and Trafalgar have similar defect histories,
each activates six clusters under the old C5 threshold, and both decay to about
`0.0127` median by I30. It cannot by itself select additional local work.

The frozen eight-scene reliability cohort rejects the alternative normalization
`q=sqrt(interior defect/proximal displacement)>1` as a direct trigger for fixed
L2: precision is `0.429`, recall `0.750`, sign accuracy `0.375`, and maximum-q
rank correlation with benefit is `-0.071`. Fixed L2 also updates shared cameras,
so it is not a clean ground truth for an interior-only diagnostic. Do not wire q
to `local_steps=2`. A future true interior-only trial must hold shared cameras
fixed and provide atomic rollback on local proximal-objective failure. Report:
`benchmark_results/k1_inner_on_mature_base_k24_i30/forcing_reliability.md`.

The mature I90 endpoint oracle redirects that future work. One exact K1 step
gives Roman `0.699920x` and Trafalgar `0.982342x` endpoint SSE (`0.829193x`
geometric). Roman therefore has large coordinated camera descent available even
though fixed L2 hurts it; interior forcing is not the missing move. The next
clean diagnostic, one distributed global shared-camera Schur correction from
the same states, is complete. The frozen start damping `0.005859375` plus
geometric fallback improves all 15 1DSfM endpoints to `0.793908x` mature SSE
geometrically and `1.267579x` Ceres. On BAL29 it is safe but modest:
corrected/actual-prestate is `0.999518x`, W/T/L `26/3/0`; three nonconverged
cases no-op, and no nonconverged solve is accepted. Six large BAL checkpoints
were corrupt, so those endpoints were exactly rerun and corrected in-process;
the reloadable 23 were corrected directly. The common policy therefore passes
the cross-family safety gate without scene routing. Full report:
`benchmark_results/k1_mature_i90_endpoint_oracle_i1/report.md`.

The next mechanism question is late-iteration scheduling of this same
safeguarded distributed correction, not adaptive fixed L2 and not a repeated
Schur polishing tail.

The authoritative continuation protocol is now `K1_CARRYOVER_TEST_PLAN.md`.
It freezes the source checkpoint, cohorts, controls, one-correction safeguards,
coherent rebuild hypothesis, non-threshold late trigger, independent
shared-fixed interior trial, promotion gates, artifact layout, and per-phase
commit/update rules. Resume from its phase-status table, not from the newest
log directory. Phase 0 must first port and commit the detached zero-iteration
validation and Schur-damping runner controls; do not combine that protocol
commit with the coherent rebuild implementation.

Phase 2 is now closed. A same-executable I90 control and I90-horizon stop at
I60 match exactly through the first 60 trajectory rows. One frozen correction
plus canonical I30 restart improves the terminal-correction endpoint to
`0.933819x` on Roman/Trafalgar but regresses BAL52/3068 to `1.010695x`.
Therefore do not tune correction timing or implement the global stagnation
trigger. Keep the terminal correction and continue only with the independent
shared-fixed interior trial documented in the plan.

Phase 4 is also closed. The shared-fixed unique-camera/landmark trial has exact
rollback and shared-camera invariance, and improves the six-scene 1DSfM cohort
to `0.921785x`. It fails cross-family transfer: BAL is `1.071668x`, BAL52 is
`1.043313x`, and BAL3068 recovery-exhausts at I15 with `1.353560x` SSE. Do not
tune or broaden it. No additional K1 carryover phase remains; the retained
package is direct tangent, exact metric consistency, shared-only product-space
semantics, and one terminal safeguarded distributed Schur correction.

The active follow-up is no longer another K1 mechanism. It is the unchanged
terminal-correction transfer to the frozen K4 resource and K16 latency
endpoints. The sentinel gate passes on Roman/Trafalgar and BAL52/3068:
corrected/prestate is `0.822336x`/`0.775883x` on 1DSfM and
`0.976151x`/`0.973048x` on BAL at K4/K16. Resume from
`TERMINAL_CORRECTION_SCALING_PLAN.md`; many BAL NPZ states are corrupt, so
validate physical SSE and rerun invalid rows in-process.

That K4/K16 breadth gate is complete and passes. Correction/prestate is
`0.785461x`/`0.778120x` on all-15 1DSfM and
`0.997869x`/`0.997663x` on all-29 BAL for K4/K16. Accepted/no-op counts are
`15/0`, `15/0`, `26/3`, and `27/2`; no nonconverged solve is accepted. Raw
K4/K16 remain the DRS-only resource/latency endpoints. Publication tables now
include separately labeled one-terminal-correction rows. Full report:
`benchmark_results/terminal_correction_scaling_all/report.md`.

The active quality gate is now the unresolved integrated K24/I200 continuation,
frozen in `K24_I200_QUALITY_PLAN.md`. Both sentinel arms request I200 so their
first 90 rows must match exactly; one stops at I90 before the terminal
correction, while the other runs through I200 before the same correction. Start
with Roman, Trafalgar, BAL52, and BAL3068. Do not substitute the historical
pre-integration I200 artifact or the failed plain-resolvent I1000 preset.

The four-scene K24/I200 sentinel passes. Corrected-I200/corrected-I90 is
`0.948260x` on Roman/Trafalgar and `0.989516x` on BAL52/3068, with all four
scenes improving; corrected I200 is `1.072873x`/`0.976646x` Ceres on the two
pairs. Continue unchanged to all-15/all-29 breadth from
`K24_I200_QUALITY_PLAN.md`.

Full K24/I200 breadth is complete but not promoted. Failure-inclusive
corrected-I200/corrected-I90 is `0.959795x` on 1DSfM and `0.996820x` on BAL;
corrected/Ceres is `1.160916x` and `0.996933x`. Piazza recovery-exhausts at
I188, violating the completion gate. Montreal/Yorkminster are the only
corrected-I200 1DSfM losses. Do not tune the horizon or safeguards. Retain this
as a diagnostic quality ceiling and next collect behavior-neutral nominal-DRS
versus global-Schur direction alignment at I90/I120/I160/I200.

The active diagnostic is `K24_SCHUR_ALIGNMENT_PLAN.md`. It uses the actual
low-damping correction direction and must remain bitwise trajectory/state
neutral. The fixed cohort is Roman, Trafalgar, Montreal, Yorkminster, Piazza,
BAL52, and BAL3068; do not tune checkpoints or damping from its outcome.

The split result passes neutrality and linear convergence. Late shared DRS
motion has median weighted cosine `0.022--0.050` with Schur and only
`0.000007--0.000645x` its shared model gain. Every unique-camera component has
a nonpositive frozen Schur camera model; BAL3068 has `1e17--1e18` shared norm
ratios. Similarity-gauge projection does not change these quantities: median
DRS gauge fraction is `1.58e-9` at I90 and `1.93e-13` at I200, including tiny
fractions on BAL3068. Gauge drift is closed. The sole active mechanism test is
behavior-neutral coherence of the exact metric-weighted reflected copy votes
entering shared-camera consensus.

That coherence test passes neutrality and shows heavy cancellation, with
median global coherence `0.0199--0.0310`, but cancellation does not predict
quality: full-cohort Spearman correlations are only `0.190` with Schur cosine
and `0.223` with shared model gain, and BAL52 has low coherence with far better
alignment. Close aggregate cancellation as the primary mechanism. The next
telemetry-only test compares each reflected shared-camera copy tangent against
its corresponding global Schur tangent before projection.

The Roman/BAL52 sentinel closes that copy-level test. Raw and exact
metric-contribution cosines are near zero in both, signed action balance
overlaps, and projection improves both; see
`benchmark_results/k24_schur_contribution_alignment_sentinel/report.md`. The
sole active diagnostic is the projected shared-camera cosine/action
distribution, separating broad misalignment from domination by a small
high-energy camera subset.

The projected-camera sentinel identifies broad misalignment. Roman has median
camera cosine `0.21--0.28`, `71--75%` positive cameras, and signed action
balance `0.42--0.80`; BAL52 has `0.39--0.59`, `81--100%`, and `0.96--1.00`.
See `benchmark_results/k24_schur_camera_distribution_sentinel/report.md`.
Next compute a behavior-neutral factorized cross-camera consensus oracle from
the same reflected copies and distributed Schur factors. Apply nothing until
that oracle improves Roman while preserving BAL52; any promoted implementation
must keep the coupled metric consistent in our DRS local and consensus steps.

The projection-only oracle fails on both scenes: coupled/Schur norms are
`337--1249x` on Roman and `275--26512x` on BAL52, with strongly negative model
gain throughout. Exact trajectory/state neutrality still passes. Close this
oracle because it mixes block-metric local copies with a coupled coordinator
projection. Next restore the crash-lost observability/off-diagonal-majorizer
dispatch currently guarded by `NotImplementedError`, then run the frozen
threshold `0.55`, scale `0.5`, I2--I10 majorizer consistently in local prox and
consensus on the mature C1/Nesterov Roman/BAL52 pair.

The restored consistent majorizer is neutral when disabled and safe on BAL52,
but Roman changes from `0.993173x` control at I30 to `1.007679x` at I90, with
10 versus 7 rejections. Close it without tuning; see
`benchmark_results/schur_majorizer_mature_i90/report.md`. Next add read-only
telemetry for one Jacobi-preconditioned Schur residual correction of the shared
DRS tangent. It must improve Roman while preserving BAL52 before any proposal
is applied.

The one-step oracle passes: Roman I90/I120 cosine reaches `0.763/0.775` and
BAL52 `0.929/0.950`, with `0.875--0.969` of full Schur camera-model gain and
exact trajectory prefixes. See
`benchmark_results/k24_one_step_schur_oracle_sentinel/report.md`. Next implement
one default-off I90 proposal selected only by lower precise worker SSE, with
atomic `drs_state_for_consensus` rebuild and one proposal maximum. Keep the full
Schur direction diagnostic-only.

The applied I90 gate passes after fixed geometric SSE backtracking. Roman uses
scale `0.5` and reaches `0.797016x` delivered control at I120 with fewer
rejections; BAL52 uses `0.0625` and remains `0.9999995x`. See
`benchmark_results/k24_one_step_schur_proposal_i90_sentinel/report.md`. Next run
the exact policy on the frozen seven-scene direction cohort through I120. No
timing, damping, or scale changes before that transfer.

The seven-scene transfer passes. Five 1DSfM scenes reach `0.876147x` control
geometrically at I120, W/L `5/0`; BAL52 is neutral and BAL3068 declines. All
seven complete. Two abrupt WSL exits occurred because the rejected coupled
consensus oracle remained enabled as proposal telemetry on BAL3068. It is now
explicit opt-in. The recovered case completed alone with a 14 GiB process
ceiling and `8,358,544/4,603,516 KiB` coordinator/worker RSS. See
`benchmark_results/k24_one_step_schur_proposal_i90_cohort/report.md`. Next create
a low-memory resumable all-15/all-29 runner; serialize large BAL and keep the
coupled oracle disabled.

Canonical product-state restart stabilizes all-15 continuation. Delivered I120
is `0.887673x` control geometrically and `0.861948x` summed, W/T/L `12/1/2`;
Gendarmenmarkt is `1.058521x` and Tower `1.011275x`. Strict no-loss promotion
fails, so do not tune or promote. The next and only active run is unchanged
serial BAL safety transfer under the memory ceiling, allowed by the repository
bounded-loss component policy. See
`benchmark_results/k24_one_step_schur_proposal_i90_breadth/report.md`.

The all-29 BAL transfer is complete and behavior-exact: all 29 proposals
decline at the fixed `1e-3` floor, W/T/L `0/29/0`, with geometric and summed
delivered ratios `1.000000x`. BAL961 first failed locally at the 14 GiB ceiling
inside the unnecessary full reference Schur solve. Proposal checkpoints are
now independent of full alignment diagnostics; they build the distributed
Schur systems, apply one block-Jacobi residual correction, and evaluate the
eight fixed scales only. BAL961 then completed at
`7,187,668/8,902,156 KiB` coordinator/worker RSS. Safety transfer passes, but
do not promote: BAL has no selected updates and 1DSfM retains two losses.

The continuation-state follow-up is complete. Selected proposals now have an
opt-in I91 worker trust and coordinator curvature/acceleration rebase. It keeps
I1--I90 and proposal choices exact. All-15 1DSfM improves from `0.887673x` to
`0.884652x` control geometrically and from `0.861948x` to `0.860613x` summed;
W/T/L becomes `13/1/1`. Tower is repaired to `0.999139x`; Gendarmenmarkt is the
only remaining loss at `1.010662x`. All BAL29 proposals previously declined,
so this selected-only mechanism leaves that exact safety result unchanged;
BAL52 was reconfirmed bitwise. Retain as a bounded-loss proposal component,
not a promoted common default. Do not tune rebase timing or radius. See
`benchmark_results/k24_one_step_schur_proposal_i90_trust_rebase_breadth/report.md`.

The next behavior-neutral oracle rejects a second fixed unit block-Jacobi
residual correction. Two-step/one-step worker SSE is `1.000710x` on
Gendarmenmarkt and `1.008820x` on Roman; their second-step residual norms grow
`4.92x` and `238x`. BAL52 contracts `0.845x` but still declines. Trajectories
and endpoint states remain exact. Keep one residual action only; do not run
depth/relaxation sweeps. See
`benchmark_results/k24_two_step_schur_residual_oracle_sentinel/report.md`.

An exact frozen-Schur line minimization of the second correction also fails.
Gendarmenmarkt/Roman coefficients are near unit (`0.964/1.075`) and improve
the quadratic model, but physical candidate SSE is still
`1.000677x/1.010264x` depth one. BAL52 still declines. This closes coefficient
error and further fixed local Schur iteration; retain one action only. See
`benchmark_results/k24_model_optimal_two_step_schur_oracle_sentinel/report.md`.

A fresh Schur rebuild at the selected depth-one state also fails. Gendarmenmarkt
improves only `0.0455%`, below the frozen `0.1%` floor; Roman has no descent at
any scale, and full fresh steps regress `2.62%/19.42%`. BAL52 skips the rebuild.
All trajectories/states are exact. Close stale linearization and do not lower
the floor. See
`benchmark_results/k24_relinearized_second_schur_oracle_sentinel/report.md`.

Offline continuation-policy oracles are also closed. Best-state-only proposal
delivery is safe but only `0.934688x` control. An exact ordinary/proposal
endpoint race would reach `0.884026x`, just `0.071%` beyond the retained result,
by falling back only on Gendarmenmarkt; it costs at least `1.25x` outer work and
requires absent full DRS snapshots. Do not rebuild a shadow race. See
`benchmark_results/k24_schur_proposal_continuation_policy_oracles/report.md`.

The all-camera residual oracle is rejected. Versus shared-only depth one it is
`1.001933x` on Gendarmenmarkt and `1.127536x` on Roman, with correction norms
`6.14e13/2.16e11`; BAL52 still declines. Exact trajectories/states confirm
telemetry neutrality. Unique cameras remain locally owned; do not tune this
route. See
`benchmark_results/k24_all_camera_schur_residual_oracle_sentinel/report.md`.

Three-step eliminated-landmark response passes the frozen seven-scene gate.
Five 1DSfM scenes improve from `0.822388x` camera-only to `0.691764x` control,
W/T/L `5/0/0`, with scale `1.0` selected throughout. BAL52/3068 decline and
remain exact. The applied path atomically commits selected cameras plus refined
landmarks, canonically restarts, and rebases trust at I91. Next run unchanged
all-15 breadth, then BAL29. See
`benchmark_results/k24_schur_proposal_landmark_response_cohort/report.md`.

All-15 breadth passes unchanged: `0.790889x` geometric and `0.760039x` summed
delivered/control, W/T/L `14/1/0`, with 14 scale-`1.0` selections and exact
Madrid decline. Candidate/Ceres is `1.177568x`; all scenes complete below
`1.82 GiB` coordinator/worker RSS. The only active gate is unchanged serial
BAL29 under the 14 GiB cap. See
`benchmark_results/k24_schur_proposal_landmark_response_breadth/report.md`.

BAL29 is complete and exact: all 29 proposals decline, W/T/L `0/29/0`, with
maximum coordinator/worker RSS `6.873/9.689 GiB`. The combined all-15/all-29
no-loss gate therefore passes. Promote the K24 I90 landmark-response proposal
component unchanged: one shared-camera residual action, eight scales, three
rollback-safe landmark steps, exact refined-SSE selection, atomic camera plus
landmark commit, canonical restart, and selected-only I91 trust rebase. This is
our distributed DRS acceleration, not Ceres or a full Schur solve. See
`benchmark_results/k24_schur_proposal_landmark_response_breadth/report.md`.

The frozen one-correction polish passes Roman/Trafalgar/BAL52/BAL3068 after the
promoted I120 state. Incremental correction/raw is `0.963498x` on 1DSfM and
`0.999059x` on BAL; all four converge and accept. Expand unchanged to
all-15/all-29 as a separately labeled polished endpoint. See
`benchmark_results/k24_landmark_response_terminal_correction_sentinel/report.md`.

Polished breadth is complete. One correction gives `0.964507x` the promoted
1DSfM handoff and `1.135772x` Ceres geometrically (`1.044180x` summed), with all
15 accepted. BAL is `0.999642x` handoff and `0.998681x` Ceres geometrically
(`0.994776x` summed), correction W/T/L `26/3/0`; three nonconverged solves no-op
exactly. Bounded low-memory transpose assembly keeps peak BAL RSS at
`8.306/12.051 GiB`. Keep this separately labeled polish. The remaining gap is
1DSfM trajectory/basin quality, not terminal descent. See
`benchmark_results/k24_landmark_response_terminal_correction_breadth/report.md`.

Only `all15_corrected/` is authoritative. The sibling `all15/` directory is a
quarantined setup error with the wrong trust cap and recovery controls.

## 2026-08-14 Offline Freeze Boundary

This is the current continuation boundary for several days of offline work.
The local and remote branch were verified at `8954873` on
`user/chvogel/notSamePerformanceNew2c`. That checkpoint includes the final
product-SO3 decision in `DRS_RESEARCH_OBJECTIVE.md`. No solver source change is
pending. This handoff is the only intentional tracked edit after that
checkpoint.

The research method is frozen unless a new mechanism is derived from existing
accepted-step and geometric evidence. Do not use the offline period for another
scalar, scene-specific, duration, selector, trust-radius, metric-scale, C5, or
correction-tail sweep. The productive offline tasks are manuscript
consolidation, artifact validation, table/figure generation, and tests.

Current method decisions:

1. Keep the distributed K>1 product-space DRS architecture. Landmark tracks
  have one owner; only duplicated cameras participate in proximal consensus.
2. Keep direct left-SE3 tangent normal equations and exact tangent-metric
  transforms as the common camera model. Ceres and BAE are references, not
  replacements for the distributed method.
3. Keep one global policy across 1DSfM and BAL. The frozen Stage-C stack is C1
  safeguarded acceleration plus C5 adaptive local work, with C5 start I5,
  thresholds `0.35/0.20`, window/dwell `3/3`, and maximum depth 2.
4. Keep K4 as the resource endpoint and K16 as the latency endpoint. These are
  deployment budgets, not scene routing. Do not promote K24 as the final
  scaling endpoint.
5. Keep product-SO3 default-off. Its final C1-only breadth is strong on BAL29
  (`0.996776x` left-SE3, 23/29 wins, worst `1.001555x`) but fails cross-family
  promotion on all-15 1DSfM (`1.039726x`, worst Piazza `1.496952x`). It is a
  BAL-oriented ablation, not the common method.
6. Keep the startup distributed Schur bootstrap as an aggregate-positive
  diagnostic/Pareto component, not a default. It improves fresh all-15 raw
  equal-budget quality to `0.968770x` at `1.031338x` runtime, but is
  `1.003172x` the stronger preserved weekend workflow and does not persist
  under unchanged long BAL DRS.

The final SO3+C1 report is
`benchmark_results/so3_c1_confirmation_k24_i30/report.md`. All 15 1DSfM and 29
BAL rows complete 30 iterations; maximum delivered local linear residuals are
`0.0099999581` and `0.0099999362`, respectively.

When a statement here conflicts with an older historical section below, this
freeze boundary and `DRS_RESEARCH_OBJECTIVE.md` are authoritative.

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
aggregate quality on all 15 1DSfM scenes (`0.993529x` SSE). That exact K1
package uses raw scene coordinates, no camera scaling, and no landmark
preconditioning; the normalized/Jacobi package is the separate maintained K>1
core and must not be attributed to this K1 artifact.

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

The K4/K16 timing-repeat gate is also closed. Three measured sentinel repeats
after one warm-up have `7/8` bitwise-identical SSE cases, maximum relative SSE
spread `2.956e-09`, identical rejection/oracle counts, and maximum optimization
CV `1.75%`. Repeated K16/K4 optimization ratios are `0.5158x` on 1DSfM and
`0.4190x` on BAL. Keep both endpoints frozen. Report:
`benchmark_results/stage_c_scaling_repeats_k4_16_i30/report.md`.

The final K1/K4/K16 versus Ceres/BAE publication comparison is generated from
frozen artifacts. It uses separate all-15 1DSfM, all-29 BAL, and six-scene BAE
panels, explicitly marks unavailable coverage, and never compares CPU/GPU time
as a speedup. The corrected table includes preserved best base DRS: current
K4/K16 are faster but worse in endpoint quality on both families. K1 remains a
local diagnostic; K4/K16 are distributed speed endpoints, not base-quality
replacements. Report:
`benchmark_results/stage_c_publication_comparison/report.md`.

A new base-backbone hybrid gate is complete. Restoring Nesterov, persistent DRS
trust, curvature `0.4` recovery/decay, and metric `75`, then adding C1 plus
shared-only proposal damping gives `0.992977x` base-I30 on all-15 1DSfM and
`0.997428x` on all-29 BAL with full completion. C5 adds no robust 1DSfM value on
this backbone. Do not yet promote the hybrid: permanent and two transient I90
continuations all regress both families; permanent damping also exhausts Tower.
Next work must address late continuation, not retune scalar thresholds. Reports:
`benchmark_results/stage_c_base_structure_factorial_k24_i30/report.md` and
`benchmark_results/stage_c_transient_shared_proposal_c1_k24_i90/report.md`.

Denominator clarification: tuned C1+C5 at `1.129336x` base on 1DSfM and
`0.995443x` on BAL compares against best-I30 checkpoints extracted from the
preserved legacy long trajectories. Those checkpoints are `1.896006x` and
`1.014321x` Ceres; tuned C1+C5 is `2.141229x` and `1.009698x` Ceres. This is
different from `C1+C5/plain = 0.847890x/0.983275x`, whose denominator is the
fresh Stage-C plain I30 arm with the same Schur-PCG/DABA-trust configuration.

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

### P0: Offline Manuscript Consolidation

1. Make the manuscript internally consistent with the current Stage-C labels,
   cohorts, and frozen policy. Update `paper_plan.md`,
   `paper_method_evaluation_spine.md`, and `stage_c_long_horizon_results.tex`;
   those files contain useful structure but some provisional decisions.
2. Build final tables directly from authoritative JSONL/summary artifacts. Each
   row must state family, scene count, K, outer budget, objective, camera mode,
   local solver, and whether the number is a fresh control, preserved artifact,
   or cross-hardware reference.
  A preliminary per-scene pixel-RMSE table is generated by
  `serverTest/build_per_scene_reprojection_comparison.py` at
  `benchmark_results/stage_c_publication_comparison/per_scene_reprojection.md`.
  Its JSON companion retains SSE, observation counts, mean error, and ratios.
3. Generate the main plots from existing data: per-scene SSE ratios, objective
   versus optimization time, K4/K16 time-quality tradeoff, communication bytes,
   maximum-worker/aggregate memory, and accepted/rejected outer trials. Do not
   launch replacement runs merely to simplify a plot.
4. Audit every headline number against the independent pixel-SSE evaluator and
   the reproducibility manifest. Regenerate
   `benchmark_results/stage_c_reproducibility_manifest.json` only if an
   authoritative artifact path or set changes.
5. Reduce the long mechanism history to a clean main narrative. Put failed or
   diagnostic mechanisms in an appendix/table rather than presenting the paper
   as a chronological search log.

### P1: Reproducibility Package

6. Run the maintained test suite and worker build before the next source
   checkpoint. The last full camera-parameterization validation passed 224
   tests and the worker built successfully.
7. Record one canonical command/configuration for plain, C1, C5, C1+C5, K4,
   K16, product-SO3, and startup bootstrap. Verify that every paper table can be
   traced to one report and one machine-readable artifact.
8. Archive raw JSONL, logs, states, timing, and memory records outside ordinary
   Git. Compact reports, analyzers, manifests, and manuscript inputs belong in
   Git; multi-gigabyte run trees do not.
9. Add a short limitations/reproducibility note covering the post-crash source
   boundary: historical checkpoint portfolios and full factorized-C3 execution
   are artifact-backed but not maintained runnable paths.

### P2: Experiments Still Worth Doing Before Submission

10. Run controlled same-machine competitors where runnable and legally
    available: Ceres is already the quality reference; prioritize DABA, STBA,
    and MegBA only when their objective, initialization, and hardware accounting
    can be made explicit. Never report CPU/GPU or unlike-machine ratios as
    speedups.
11. Complete network latency/bandwidth and straggler sweeps for the frozen K4
    and K16 methods. Report rounds, payload bytes, barrier wait, local compute,
    and time-to-target. This is the most important missing systems evidence for
    the synchronization-tradeoff thesis.
12. Add representative repeated wall-time measurements beyond the completed
    sentinel repeat gate if required by the target venue. Endpoint quality does
    not need rerunning when the saved deterministic artifacts already answer the
    question.
13. Optionally test the already implemented bootstrap trust-state rebase as one
    matched, frozen persistence gate. Do this only if method development resumes:
    write the hypothesis and transfer criterion first, change no other control,
    and do not open another trust sweep.
14. Re-run the literature/forward-citation search after network access returns,
    especially for DABA, PenBA, MegBA, camera consensus, lazy/asynchronous BA,
    and inexact nonconvex DRS.
15. Pursue an inexact-prox convergence result only if its assumptions can be
    connected to recorded local stationarity/residual quantities. Otherwise,
    state theory for the exact fixed-metric idealization and label finite local
    solves, refreshed metrics, and safeguarded acceleration as empirical.

No new solver mechanism is currently authorized. A future mechanism must first
explain an observed accepted-step or basin transition and specify a cheap
falsifying gate. The existing evidence closes broad K1 tuning, C3 variants,
C4 tolerance/solver switching, C5 timing/threshold sweeps, continuation,
mid-run Schur reset, repartitioning, solver switching, product-SO3 tuning,
camera-center coordinates, and repeated correction-tail tuning.

## What To Include In The Paper

### Main Claim And Scope

The paper should ask when independent nonlinear local BA solves plus
camera-only outer consensus are preferable to synchronized global linear-solver
collectives. The defensible claim is a communication, synchronization, memory,
and quality tradeoff for distributed K>1 BA. It is not a claim that this CPU
implementation universally beats Ceres, BAE, DABA, or GPU Schur solvers in wall
time.

Use the original Snavely pixel SSE as the primary objective. State that all
promotion decisions use one global policy across both 1DSfM and BAL, and that
Ceres/BAE are references rather than substituted inner implementations.

### Method Sections

1. Landmark-owned product-space decomposition and why landmarks never require
   consensus.
2. Shared-only camera proximal semantics and the full regularized `9x9`
   block-metric projection, including the metric-consistency derivation.
3. Direct left-SE3 perturbations, tangent order, exact physical/scaled tangent
   maps, points-p95 normalization, Jacobi equilibration, and independent
   physical pixel-SSE evaluation.
4. Plain variable-metric DRS followed by C1 safeguarded Themelis acceleration:
   accelerated proposal, DRE/physical safeguards, nominal fallback, restart,
   accepted-state recovery, and best-state restoration.
5. C5 adaptive local nonlinear work and its frozen common policy. Explain that
   C5 is interaction-sensitive and was selected on development then confirmed
   unchanged, not tuned per scene.
6. Finite local Schur solvers as an ablation. Nesterov is the named baseline;
   Schur-PCG is a faster but nonuniform C4 level, not a universal replacement.
7. Partitioning and K as systems controls over work balance, duplicated cameras,
   communication, memory, and synchronization.

### Main Experimental Evidence

Include these results, with cohort and budget labels visible:

- K1 diagnostic: the portable BAE-style left-SE3 path reaches `0.993529x`
  Ceres geometrically over all 15 1DSfM scenes. Explicitly separate this local
  correctness diagnostic from the normalized/Jacobi distributed K>1 method.
- Stage-C breadth: tuned C1+C5/plain is `0.847890x` on all-15 1DSfM and
  `0.983275x` on all-29 BAL; C1+C5/C1 is `0.980053x` and `0.999486x`.
- Honest quality reference: preserved best base DRS remains better than frozen
  K4/K16 endpoints. On 1DSfM, K4/K16 are `1.408725x`/`1.343358x` base SSE at
  `0.280585x`/`0.163860x` base optimization time. On BAL they are
  `1.008920x`/`1.011713x` base SSE at `0.738821x`/`0.376767x` time.
- Scaling repeat evidence: K16/K4 optimization ratios are `0.5158x` on 1DSfM
  and `0.4190x` on BAL, maximum optimization-time CV is `1.75%`, and endpoint
  variation is negligible. Pair this with traffic, worker CPU, and memory.
- C1/C2/C4/C5 ablations. Present C1 as the robust quality component, C5 as a
  selective work-policy interaction, coordinate equilibration as necessary
  numerical infrastructure, and PCG/Nesterov as a finite-work basin tradeoff.
- Product-SO3 as a negative cross-family/positive BAL ablation: BAL29 improves
  to `0.996776x` left-SE3 with a `1.001555x` worst case, while all-15 1DSfM
  regresses to `1.039726x` with a `1.496952x` tail. This supports retaining
  left-SE3 globally and demonstrates that camera coordinates change basins.
- Startup Schur bootstrap as a secondary Pareto/diagnostic result, not part of
  the default algorithm. Show its short-horizon aggregate gain, low-overhead
  all-15 equal-budget result, BSR40/BSR50 Pareto points, and failed long-BAL
  persistence. This is evidence about basin initialization, not a replacement
  for DRS consensus.
- Determinism, completion, residual, safeguard, and objective-agreement checks.
  Include the robust Huber implementation as validated infrastructure but a
  negative transfer result, not a promoted headline method.

### Figures And Tables

The minimum main-paper set should be:

1. Method diagram: landmark ownership, duplicated cameras, independent local
   nonlinear solves, one camera metric/state exchange, and coordinator
   projection.
2. Pseudocode for plain DRS plus the C1 accelerated/fallback path and C5 local
   depth update.
3. All-15/all-29 component table for plain, C1, C5, and C1+C5.
4. K4/K16 quality-time-traffic-memory table and objective-versus-time curves.
5. Per-scene ratio plot exposing tails rather than only aggregate geomeans.
6. Reference comparison table with separate 1DSfM, BAL, and limited BAE panels;
  unavailable coverage and unlike hardware must be visibly marked. The
  preliminary table already supplies Ceres/ours/DABA for all 15 1DSfM scenes,
  BAE-CG for its six verified scenes, and Ceres/ours for all 29 BAL scenes.
7. Communication/synchronization operating-regime plot once network experiments
   exist.

Put the extensive rejected-mechanism matrix, C3 artifact-only evidence,
product-SO3 details, camera-center rejection, continuation/repartition/solver
switches, and bootstrap variants in appendices or supplementary material. The
main paper should explain decisions, not narrate every experiment.

### Required Limitations And Non-Claims

- Do not claim the first distributed, decentralized, accelerated,
  asynchronous, or variable-metric consensus BA method.
- Do not claim K4/K16 improve the best endpoint quality; they are speed/resource
  operating points.
- Do not claim Nesterov universally dominates PCG or product-SO3 universally
  improves camera optimization.
- Do not claim convergence of changing metrics, finite GN/LM solves, or
  heuristic acceleration from an exact fixed-metric DRS theorem.
- Do not claim fewer variables imply fewer bytes; full metrics and rejected
  outer trials must be counted.
- Do not label SfM_Init-derived 1DSfM results as a reproduction of DABA Table II.

## What Must Be Checked In

Current branch: `user/chvogel/notSamePerformanceNew2c`

Remote: `origin https://github.com/vogechri/bundlePalm.git`

Checkpoint at the offline boundary:

- local and remote HEAD: `8954873`;
- no pending solver source modification;
- `DRS_RESEARCH_OBJECTIVE.md` and `results_base.json` are included in that
  checkpoint;
- this updated `CONTINUATION_HANDOFF.md` is the intentional post-checkpoint
  tracked change;
- the following compact artifacts are currently untracked:
  `serverTest/build_per_scene_reprojection_comparison.py`,
  `benchmark_results/stage_c_publication_comparison/per_scene_reprojection.md`,
  `benchmark_results/stage_c_publication_comparison/per_scene_reprojection.json`,
  and `benchmark_results/so3_c1_confirmation_k24_i30/report.md`.

Do not blindly run `git add .`. The working tree contains datasets, raw run
trees, PDFs, build products, virtual environments, and binary libraries. Review
the ignore status and stage only compact semantic artifacts. A suitable review
sequence is:

```bash
git status --short --untracked-files=all
git diff -- CONTINUATION_HANDOFF.md
git add \
  CONTINUATION_HANDOFF.md \
  serverTest/build_per_scene_reprojection_comparison.py \
  benchmark_results/stage_c_publication_comparison/per_scene_reprojection.md \
  benchmark_results/stage_c_publication_comparison/per_scene_reprojection.json \
  benchmark_results/so3_c1_confirmation_k24_i30/report.md
git diff --cached --check
git diff --cached
```

The user must explicitly request a commit/push; this handoff update does not
create one.

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
  serverTest/test_drs_safeguards.py \
  serverTest/test_camera_update_parameterizations.py \
  serverTest/test_se3_right_update.py

serverTest/.venv/bin/python \
  serverTest/build_per_scene_reprojection_comparison.py

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
current branch and staged/untracked handoff files. Regenerate the per-scene
reprojection table, validate its source artifacts, then continue the P0
manuscript tasks. Do not restart completed solver sweeps or edit solver behavior
without first deriving a new falsifiable mechanism from existing evidence.
```