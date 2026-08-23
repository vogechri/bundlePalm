# DRS Research Objective and Restart Contract

Updated: 2026-08-20

> Recovery note (2026-08-11): an unclean WSL/VS Code restart rolled open source
> files back to an earlier coherent editor snapshot after the completed C5
> campaigns. The maintained C1/C2/C4/C5 paper path is restored, passes 128
> tests, and the worker builds. Exact checkpoint/handoff and full factorized-C3
> coordinator execution are historical artifact-backed results but are not
> currently runnable without explicit reconstruction. Do not assume those
> experimental paths are live.

This is the authoritative restart file for the current BundlePalm research.
Read this file first after a VS Code, WSL, terminal, or context failure. Do not
resume from the latest log, experiment directory, or chat summary alone.

Compact current scoreboard: `benchmark_results/drs_ceres_current_scoreboard.md`.

Latest C3 breadth report:
`benchmark_results/factorized_fixed_all15_i5_report.md`.

Large-BAL C3 transfer report:
`benchmark_results/factorized_fixed_large_bal_i5_report.md`.

C1+C3 sentinel factorial:
`benchmark_results/c1_c3_sentinel_factorial_i5_report.md`.

C5+C3 sentinel factorial:
`benchmark_results/c5_c3_sentinel_factorial_i5_report.md`.

C1 all-15 and large-BAL breadth report:
`benchmark_results/c1_themelis_all15_large_bal_i5_report.md`.

WSL incident diagnosis and safe-run contract:
`WSL_CRASH_ANALYSIS_2026-08-10.md`.

Frozen K1 carryover and crash-resume test protocol:
`K1_CARRYOVER_TEST_PLAN.md`.

Active terminal-correction K4/K16 transfer protocol:
`TERMINAL_CORRECTION_SCALING_PLAN.md`.

Active integrated K24/I200 quality protocol:
`K24_I200_QUALITY_PLAN.md`.

Queued unified 1DSfM/BAL comparison against Ceres left-SE3:
`CERES_SE3_LEFT_BENCHMARK_PLAN.md`. Generate the artifact-only report before
authorizing any fresh solver rerun.

Active late-direction diagnostic protocol:
`K24_SCHUR_ALIGNMENT_PLAN.md`.

## 1. North Star

Develop one genuinely distributed product-space DRS method for bundle
adjustment that is competitive in endpoint quality on both 1DSfM and BAL while
preserving the K>1 architecture.

The intended contribution is our DRS consensus and consensus-aware local inner
optimization route. Do not replace our method with Ceres. Ceres is a reference
solver and diagnostic oracle. K=1 is a correctness and local-solver diagnostic;
the target method must work at K>1.

Primary quality objective: standard Snavely pixel sum-squared error (SSE).
Speed is secondary until endpoint quality is competitive.

## 1A. Active K1-to-Core Decision (2026-08-11)

K1 investigation is complete enough to stop broad K1 tuning. In the maintained
worker/coordinator implementation, the deterministic portable BAE-style K1
package reaches `0.993529x` Ceres SSE geometrically over all 15 1DSfM scenes.
It is not scene-identical to Ceres, but it answers the aggregate local-solver
quality question. Tightening Schur-PCG from `1e-2` to `1e-4` did not recover the
BAE-style Nesterov trajectory on Roman/Trafalgar, so the remaining difference
is not generic linear-solve accuracy.

Mandatory K1 carryovers already integrated into the core method are:

1. direct left-SE3 tangent normal-equation assembly;
2. exact `T^T M T` proximal metric consistency;
3. shared-only product-space DRS, giving the correct zero-consensus K1 limit;
4. points-p95 normalization, Jacobi camera scaling, and landmark
   preconditioning, which are compatible with the portable K1 package.

Supplementary plain-resolvent horizon ablation (2026-08-19): a clean `4db26e4`
checkpoint ran all 15 1DSfM scenes at K24 with direct left-SE3 tangent
equations, exact tangent metrics, shared-only proximal semantics,
points-p95/Jacobi/preconditioning, one finite Nesterov local step, and no C1,
C5, bootstrap, or Schur polishing. Twelve scenes reached I1000. Piccadilly
exhausted recovery at I992, Tower at I77, and Trafalgar at I135; all three hit
the curvature ceiling 64. Best-state restoration is correct.

The failure-inclusive delivered endpoint is `1.950010x` Ceres geometrically,
`1.847092x` summed, W/L `0/15`, worst Tower `5.709418x`. On the 12 scenes that
reach I1000, the common-prefix ratio is `1.907980x` Ceres. The corrected method
is `1.367264x` the old legacy I1000 endpoint and loses all 15 direct
comparisons. The old I1000 artifact is not method-equivalent because it lacks
direct tangent assembly and shared-only semantics.

Keep the K1 carryovers as mathematical and implementation correctness
requirements, but close horizon extension of this early fixed plain resolvent.
Do not mistake this row for the best current architecture. Existing all-15
direct-tangent/shared-only I90 campaigns already cover the later base backbone
with C1: `1.591438x` Ceres with a diagnostic trust/curvature rebase,
`1.596631x` without proposal damping, and `1.609367x` with transient proposal
damping. The preserved K24/I200 pure-DRS reference remains `1.457017x` Ceres,
and separately labeled polished workflows reach `1.063839x`. Continue from the
later backbone and publication presets. See
`benchmark_results/k1_compatible_pure_drs_k24_i1000/report.md`.

Do not attribute the `1.950010x` row to bringing K1 corrections into the good
base. It also disables C1 and replaces the mature DRS-trust/curvature-0.4/
metric-75 package with the K1-local DABA-trust/curvature-0.00625/metric-25
package. Matched all-15 structural factorials keep direct tangent and the
mature backbone fixed: shared-only/all-camera is `1.033321x` without proposal
damping and `1.048656x` with it at I30. The matched transient-proposal I90 pair
is `1.035714x`, with both arms complete. Thus shared-only has a modest 3--5%
cost under this backbone; the large cross-artifact regression is confounding,
not evidence that direct tangent or product-space semantics catastrophically
hurt the method.

### K1-to-Backbone Lineage Map (2026-08-19)

Two result lineages must remain separate:

1. The preserved mature `client_drs.py` quality backbone predates the K1
   algebra/product-space integration. Its K24/I200 all-15 1DSfM endpoint is
   `1.457017x` Ceres and its established K24/I90 all-29 BAL endpoint is
   `0.998404x`. The raw 1DSfM row has
   `directTangentNormalEquations=false` and no shared-only field; the BAL row
   also predates those controls. These are strong historical quality
   references, not evidence for the integrated K1 formulation.
2. The K1 bridge introduced direct tangent assembly, exact tangent metric
   consistency, and shared-only product-space semantics step by step, then
   recomposed them with the mature Nesterov/DRS-trust/curvature-0.4/metric-75/C1
   backbone. The complete no-final-Schur K24/I90 successor is
   `1.596631x` Ceres on all-15 1DSfM and `1.000822x` on all-29 BAL with no
   proposal damping. Transient proposal damping gives `1.609367x` and
   `0.999879x`; a diagnostic I30 trust/curvature rebase gives `1.591438x` and
   `1.001609x`. All three use one global policy and complete both cohorts,
   except the permanent-proposal arm, which is not the named successor.

The remembered approximately `1.4x` 1DSfM number refers either to the legacy
K24/I200 pure-DRS reference (`1.457017x`) or to the later integrated I30 plus
ten distributed Schur corrections (`1.411625x`). The latter is polishing and
has no matched all-29 ten-correction BAL row. The common all-family bounded
three-correction workflow is `1.626944x` Ceres on 1DSfM and `1.011023x` on BAL.
All of these authoritative runs use `client_drs.py`; `client_acc.py` is the
historical predecessor, not a separate final result source.

Therefore the actual missing apples-to-apples quality gate is not another K1
bridge or the early plain I1000 run. It is an all-15/all-29 longer-horizon
evaluation of the integrated mature backbone, with direct tangent and
shared-only semantics retained, against the preserved legacy quality
references. Do not claim that gate has already been completed at I200.

That gate is now active under `K24_I200_QUALITY_PLAN.md`. The first experiment
uses one I200 horizon for both matched arms: one stops ordinary DRS at I90 and
then applies the frozen terminal correction; the other completes I200 and
applies the same correction. Their first 90 trajectory rows must match exactly.
Roman, Trafalgar, BAL52, and BAL3068 form the frozen sentinel; no duration or
policy sweep is permitted.

Integrated K24/I200 sentinel result (2026-08-21): the two I200-horizon arms
match exactly through I90 and the continuation gate passes. Raw I200/raw I90 is
`0.943915x` on Roman/Trafalgar and `0.989118x` on BAL52/3068.
Corrected-I200/corrected-I90 is `0.948260x` and `0.989516x`, W/L `2/0` in both
families. Corrected I200 reaches `1.072873x` Ceres on the 1DSfM pair and
`0.976646x` on the BAL pair. The frozen next action is unchanged all-15/all-29
breadth; do not tune duration or correction policy. See
`benchmark_results/k24_i200_terminal_correction_sentinel/report.md`.

Integrated K24/I200 breadth result (2026-08-21): failure-inclusive corrected
I200 improves corrected I90 to `0.959795x` on all-15 1DSfM and `0.996820x` on
all-29 BAL, reaching `1.160916x` and `0.996933x` Ceres. BAL improves on all 29
scenes; 1DSfM W/L is `13/2`, with Montreal `1.025943x` and Yorkminster
`1.066442x`. Piazza recovery-exhausts at I188 with curvature ceiling 64, so the
frozen completion gate fails despite its delivered endpoint improving
`0.997056x` corrected I90. No accepted correction is nonconverged; maximum
residual is `9.967e-7`. Do not tune duration, curvature ceiling, or damping.
Retain I200+one correction as a diagnostic quality ceiling, not the promoted
common method. Next use behavior-neutral DRS/Schur direction-alignment
telemetry at fixed I90/I120/I160/I200 checkpoints. See
`benchmark_results/k24_i200_terminal_correction_breadth/report.md`.

That diagnostic is frozen in `K24_SCHUR_ALIGNMENT_PLAN.md`. It compares the
nominal accepted DRS consensus tangent with the actual low-damping terminal
Schur tangent at I90/I120/I160/I200 on Roman, Trafalgar, Montreal, Yorkminster,
Piazza, BAL52, and BAL3068. Clipping is disabled, and the run must reproduce
the I200 reference trajectories and endpoint states exactly before any
mechanism is inferred.

The split diagnostic passes those gates. Shared DRS camera motion is not a
useful scaled Schur direction: median diagonal-weighted cosine is
`0.022--0.050`, and median shared model gain is only
`0.000007--0.000645x` the shared Schur gain. Unique-camera motion has a
nonpositive frozen Schur camera model in every row, while BAL3068 also shows an
astronomical shared norm mismatch. Similarity-gauge drift is now falsified:
median DRS gauge fraction falls from `1.58e-9` at I90 to `1.93e-13` at I200,
and quotient projection does not materially change cosine, norm, or model
action, including on BAL3068. The active falsifiable mechanism is cancellation
among metric-weighted reflected copy votes. That follow-up finds real heavy
cancellation (`0.0199--0.0310` median global coherence), but closes it as the
primary explanation: coherence has only `0.190` Spearman correlation with
shared Schur cosine and `0.223` with shared model gain, and BAL52 is a strong
low-coherence/high-alignment counterexample. Next measure individual reflected
copy-to-Schur alignment before projection to locate whether global signal is
lost by aggregation or absent from local resolvents.

The Roman/BAL52 copy sentinel closes both alternatives at copy level. Raw and
exact metric-contribution global cosines are near zero in both scenes, signed
contribution balance overlaps, and best-copy cosine does not explain the much
stronger BAL52 projected direction. Projection improves both. The active test
is now the per-camera distribution of the projected shared tangent: determine
whether a small high-energy camera subset or broad camera-wise misalignment
causes the low global cosine.

That per-camera test identifies broad missing coupling. Roman has median camera
cosine `0.21--0.28`, only `71--75%` positive cameras, and signed action balance
`0.42--0.80`; BAL52 reaches `0.39--0.59`, `81--100%`, and `0.96--1.00`.
The next gate is a behavior-neutral factorized cross-camera consensus oracle on
the same reflected copies. It tests the curvature mechanism before any solver
change. If successful, integrate coupled curvature into our DRS metric on both
sides of the proximal/consensus contract; do not substitute a Ceres step or
switch wholesale to the historical Schur-PCG preset.

That projection-only oracle is rejected. Roman coupled/Schur norms are
`337--1249x`; BAL52 reaches `275--26512x`; every coupled camera-model gain is
strongly negative. The experiment remains exactly trajectory/state neutral.
This does not reject cross-camera curvature: it proves the local and consensus
metrics cannot be mixed. The next gate is the frozen transient Frobenius Schur
majorizer, consistently active in both local prox and consensus, composed with
the mature C1/Nesterov backbone. The recovered worker client currently blocks
that path with an explicit `NotImplementedError`; restore the verified
historical dispatch before running Roman/BAL52. See
`benchmark_results/k24_coupled_consensus_oracle_sentinel/report.md`.

The consistent transient Frobenius-majorizer path is restored and tested with
its frozen `0.55`/`0.5`/I10 policy. Roman reaches `0.993173x` control at I30 but
reverses to `1.007679x` at I90 with more rejections. BAL52 declines the selector
and remains bitwise exact. Close this proxy without retuning; report:
`benchmark_results/schur_majorizer_mature_i90/report.md`. The active mechanism
test is now one behavior-neutral Jacobi-preconditioned global Schur residual
correction of the nominal shared DRS tangent, not a full solve or replacement
method.

That oracle passes on Roman/BAL52 at I90/I120. Roman cosine becomes
`0.763/0.775` with `0.881/0.888` of Schur camera-model gain; BAL52 becomes
`0.929/0.950` with `0.875/0.969`. Prefix behavior remains exact. Promote only
the next gate, not the mechanism yet: one default-off I90 proposal, selected by
precise worker SSE against ordinary consensus and committed via atomic DRS
state rebuild. The proposal is one distributed matvec plus block-Jacobi
correction, not Ceres or a full Schur solve. Report:
`benchmark_results/k24_one_step_schur_oracle_sentinel/report.md`.

The safeguarded applied gate also passes. Fixed geometric SSE backtracking
selects scale `0.5` on Roman, giving `0.888157x` immediate and `0.797016x`
delivered I120 control with fewer rejections. BAL52 selects `0.0625` and is
effectively neutral at `0.9999995x`. This remains our DRS: one distributed
Schur matvec/Jacobi acceleration proposal, actual-SSE selection, and atomic
product-state rebuild; it is not a full global solve. Next transfer unchanged
to the frozen seven-scene cohort before breadth. Report:
`benchmark_results/k24_one_step_schur_proposal_i90_sentinel/report.md`.

The frozen seven-scene transfer passes: all five 1DSfM scenes improve at I120,
geometric ratio `0.876147`, W/L `5/0`; BAL52 is effectively neutral and BAL3068
declines the proposal. All seven complete. The initial BAL3068 attempt crashed
WSL while a rejected high-memory coupled-consensus oracle was still enabled as
telemetry. That oracle is now explicit opt-in; recovered BAL3068 completed alone
under a 14 GiB process ceiling. See
`benchmark_results/k24_one_step_schur_proposal_i90_cohort/report.md`. Next build
a resumable low-memory all-15/all-29 runner that serializes large BAL cases and
never enables the coupled oracle.

The stabilized all-15 run uses canonical product-state restart after selection.
It reaches `0.887673x` control geometrically and `0.861948x` summed, W/T/L
`12/1/2`. Gendarmenmarkt (`1.058521x`) and Tower (`1.011275x`) fail the strict
no-loss promotion gate. Do not tune from these tails. Under the existing
bounded-loss component policy, run the unchanged serial BAL safety transfer
under the memory ceiling before deciding whether to retain this as a research
component. Report:
`benchmark_results/k24_one_step_schur_proposal_i90_breadth/report.md`.

The unchanged all-29 BAL transfer is complete: every proposal declines under
the fixed `1e-3` floor, so all 29 trajectories and delivered states tie control
with zero losses. BAL961 initially hit the 14 GiB ceiling because proposal
execution unnecessarily ran the full reference PCG diagnostic. The proposal is
now independent of alignment diagnostics and performs only Schur-system build,
one block-Jacobi residual action, and eight worker-SSE trials. Recovered BAL961
completed at `7,187,668/8,902,156 KiB` coordinator/worker RSS with the reference
solve skipped. Cross-family safety therefore passes, but promotion remains
closed: BAL gains nothing and the all-15 1DSfM result still has two losses.

Post-proposal continuation diagnosis is complete. The two 1DSfM losses both
improve immediately at I90 but accumulate extra rejections afterward. A
default-off, selected-only I91 trust/curvature/acceleration rebase preserves
the exact I1--I90 prefix and proposal decisions. On all 15 1DSfM scenes it
improves delivered/control from `0.887673x` to `0.884652x` geometrically and
from `0.861948x` to `0.860613x` summed, with W/T/L improving from `12/1/2` to
`13/1/1`. Tower becomes `0.999139x`; Gendarmenmarkt improves from `1.037742x`
to a bounded `1.010662x` loss. Existing BAL29 proposals all decline, so the
selected-only rebase cannot alter them; BAL52 was explicitly confirmed exact.
Retain the rebase as part of the research proposal component, but do not
promote it as the common default or tune timing/radius from Gendarmenmarkt.
Report: `benchmark_results/k24_one_step_schur_proposal_i90_trust_rebase_breadth/report.md`.

A behavior-neutral depth-two follow-up closes fixed stationary Jacobi
iteration. Relative to the retained one-step candidate, two unit residual
corrections worsen worker SSE by `1.000710x` on Gendarmenmarkt and `1.008820x`
on Roman. The second-step residual expands `4.92x` and `238x`, respectively.
BAL52 contracts `0.845x` but still declines both candidates. Every trajectory
and endpoint state is exact because depth two is telemetry-only. Do not apply
or broaden fixed-depth Jacobi and do not sweep depth/relaxation. One action is
a useful direction repair; stationary repetition is unstable on the real
restricted Schur systems. Report:
`benchmark_results/k24_two_step_schur_residual_oracle_sentinel/report.md`.

Exact Schur-quadratic line minimization of the second correction also fails.
Its coefficients are already near unit on Gendarmenmarkt/Roman
(`0.964/1.075`) and increase frozen-model decrease as designed, yet physical
worker SSE remains `1.000677x/1.010264x` the retained depth-one candidate.
BAL52 chooses coefficient `3.356` but still has no admissible candidate. All
trajectories and states remain exact. This falsifies wrong second-step length:
the second direction itself does not transfer from frozen Schur model to
physical SSE. Close further fixed local Schur iteration and retain one action.
Report:
`benchmark_results/k24_model_optimal_two_step_schur_oracle_sentinel/report.md`.

Relinearizing after the selected depth-one candidate does not rescue a second
action. A fresh Schur build gives Gendarmenmarkt only `0.0455%` incremental
worker-SSE decrease at scale `0.125`, below the frozen `0.1%` floor; Roman has
no descent even at `1/128`. Full fresh steps regress `2.62%/19.42%`. BAL52
skips the rebuild because depth one declines. Trajectories and endpoint states
are exact. Close stale linearization as the cause of depth-two failure; do not
broaden or lower the floor. Report:
`benchmark_results/k24_relinearized_second_schur_oracle_sentinel/report.md`.

Continuation-policy oracles do not justify another implementation branch.
Keeping proposals as best-state checkpoints while ordinary DRS continues is
no-loss but weak (`0.934688x`, W/T/L `8/7/0`). An exact ordinary/proposal
endpoint race reaches `0.884026x`, W/T/L `13/2/0`, only `0.071%` better than
the retained `0.884652x` component, by falling back on Gendarmenmarkt alone.
It requires duplicating I91--I120 (`1.25x` outer work) and full DRS snapshots,
which current maintained source does not have. Do not reconstruct a shadow
race or replace continuation with best-only delivery for this ceiling. Report:
`benchmark_results/k24_schur_proposal_continuation_policy_oracles/report.md`.

An all-camera residual oracle rejects globally correcting unique-camera motion.
Relative to shared-only depth one, candidate SSE is `1.001933x` on
Gendarmenmarkt and `1.127536x` on Roman; Roman has no admissible all-camera
scale. The correction norm reaches `6.14e13/2.16e11` on the 1DSfM pair. BAL52
still declines. Trajectory/state neutrality is exact. Keep unique cameras
locally owned and restrict the proposal correction to duplicated cameras; do
not scale-tune this route. Report:
`benchmark_results/k24_all_camera_schur_residual_oracle_sentinel/report.md`.

The eliminated-landmark response is the first strong successor mechanism.
Using the existing three rollback-safe fixed-camera landmark steps changes the
preferred I90 scale to `1.0` and passes the frozen seven-scene transfer. Five
1DSfM scenes improve from `0.822388x` camera-only to `0.691764x` control
geometrically, W/T/L `5/0/0`; Montreal/Piazza/Roman/Trafalgar/Yorkminster are
`0.571810/0.580406/0.649632/0.895394/0.820583x`. BAL52 and BAL3068 both decline
and remain trajectory/endpoint exact. Selected cameras and refined landmarks
commit atomically, then use canonical restart and the retained I91 trust
rebase. Proceed unchanged to all-15 breadth before BAL29. Report:
`benchmark_results/k24_schur_proposal_landmark_response_cohort/report.md`.

Unchanged all-15 breadth passes the strict no-loss gate. Delivered/control is
`0.790889x` geometrically and `0.760039x` summed, W/T/L `14/1/0`; all 14
accepted scenes select scale `1.0`, while Madrid declines exactly. The previous
camera-only rebase component was `0.884652x`, W/T/L `13/1/1`. Candidate/Ceres
is now `1.177568x`. All scenes complete with maximum coordinator/worker RSS
`1.815/1.727 GiB`. Proceed unchanged to serial BAL29 under the 14 GiB cap; no
scale, depth, timing, floor, or restart changes. Report:
`benchmark_results/k24_schur_proposal_landmark_response_breadth/report.md`.

The unchanged serial BAL29 transfer is complete and exact. All 29 landmark-
response proposals decline under the fixed `1e-3` floor, giving W/T/L
`0/29/0`, geometric/summed delivered ratios equal to `1.0` within tie
tolerance, and maximum coordinator/worker RSS `6.873/9.689 GiB`. Combined with
the all-15 `0.790889x`, W/T/L `14/1/0` result, this passes the common
cross-family no-loss gate. Promote the three-step landmark-response proposal
as the retained K24 I90 DRS acceleration component: shared-camera residual
action, eight fixed scales, exact refined-landmark SSE selection, atomic camera
and landmark commit, canonical restart, and selected-only I91 trust rebase.
It remains our distributed DRS, not Ceres and not a full Schur solve. Report:
`benchmark_results/k24_schur_proposal_landmark_response_breadth/report.md`.

One frozen terminal distributed Schur correction composes safely with the
promoted I120 state. Correction/raw is `0.963498x` on Roman/Trafalgar and
`0.999059x` on BAL52/3068; corrected/Ceres is
`1.122559/0.926137/0.970115/0.989642x`. All four solves converge and accept.
Expand the exact one-correction policy unchanged to all-15/all-29 as a
separately labeled polished endpoint, not another DRS mechanism. Report:
`benchmark_results/k24_landmark_response_terminal_correction_sentinel/report.md`.

The polished all-15/all-29 breadth is complete after bounding transposed
`bsr_low_memory` assembly in 16K-block chunks. On 1DSfM, one correction improves
the promoted handoff by `0.964507x` geometrically and reaches `1.135772x` Ceres
geometrically (`1.044180x` summed); all 15 corrections converge and accept.
On BAL, correction/handoff is `0.999642x`, corrected/Ceres is `0.998681x`
geometric and `0.994776x` summed, with W/T/L `26/3/0`; the three nonconverged
solves no-op exactly. Peak BAL coordinator/worker RSS is `8.306/12.051 GiB`.
Retain this as the separately labeled one-correction polished endpoint. The
remaining research gap is 1DSfM basin/trajectory quality (`1.135772x` Ceres),
not blocked terminal descent or BAL safety. Report:
`benchmark_results/k24_landmark_response_terminal_correction_breadth/report.md`.

Ownership-correct ordinary-candidate diagnostics close stale landmark scoring
as the source of DRS rejection behavior. Three fixed-camera landmark steps give
refined/unrefined SSE `0.9877--0.9956` on Roman and `0.9587--0.9883` on
Trafalgar at I30/I60/I89, but flip no primal safeguard decisions; BAL52 is
`0.9995--0.9996` and also flips none. Worker round-trip error and complete
trajectories/states are exact. Keep the inconsistent legacy per-iteration
consensus-refinement path off. The one bounded next test is the previously
frozen alternate I60 checkpoint with the new joint camera/landmark proposal;
do not sweep timing. Report:
`benchmark_results/k24_drs_candidate_landmark_response_accepted_diagnostic/report.md`.

The sole bounded alternate checkpoint, I60, is safe and slightly
aggregate-positive for the joint camera/landmark proposal. Roman is
`0.652918x` control versus I90 `0.649632x`; Trafalgar is `0.888641x` versus
`0.895394x`, giving pair geomeans `0.761715x/0.762678x`. BAL52/3068 decline
exactly. Transfer I60 unchanged to all-15 once; do not test another checkpoint.
Report: `benchmark_results/k24_landmark_response_proposal_i60_sentinel_actual/report.md`.

Unchanged all-15 I60 transfer passes: geometric/summed delivered/control
`0.786533x/0.754474x`, W/T/L `14/1/0`, versus I90
`0.790889x/0.760039x`. All 14 accepted scenes select scale `1.0`; Madrid
declines exactly. Candidate/Ceres improves from `1.177568x` at I90 to
`1.171081x` at I60. Effects remain mixed but bounded, so close checkpoint
comparison and run I60 unchanged on serial BAL29 once. Do not test another
timing. Report:
`benchmark_results/k24_landmark_response_proposal_i60_breadth/report.md`.

The final serial BAL29 timing gate passes unchanged: all 29 I60 proposals
decline exactly, W/T/L `0/29/0`, with peak coordinator/worker RSS
`6.859/9.823 GiB`. Combined with all-15 `0.786533x` control and `1.171081x`
Ceres, W/T/L `14/1/0`, this promotes I60 over I90 as the common checkpoint.
Freeze timing now: one I60 shared-camera residual action, eight fixed scales,
three rollback-safe landmark steps, `1e-3` floor, atomic camera/landmark commit,
canonical restart, and selected-only I61 trust rebase. Do not test another
checkpoint. Report:
`benchmark_results/k24_landmark_response_proposal_i60_breadth/report.md`.

The I60-plus-one-terminal-correction composition is mixed and closes without
breadth expansion. Roman/Trafalgar correction improves the I60 handoff by
`0.968427x`, but the corrected pair is `1.003848x` the established I90-polished
pair: Trafalgar improves and Roman regresses. BAL52/3068 are exactly unchanged.
Retain I60 for the promoted DRS endpoint and I90 for the separately labeled
one-correction polished scoreboard; do not checkpoint-tune polishing. Report:
`benchmark_results/k24_i60_landmark_response_terminal1_sentinel/report.md`.

One additional joint proposal at I90 after the promoted I60 restart passes the
frozen four-scene gate. Roman/Trafalgar geomean improves from `0.761715x` to
`0.748524x` control: Roman improves `0.652918x -> 0.628975x`, while Trafalgar
has a bounded `0.888641x -> 0.890795x` regression. BAL52/3068 decline both
proposals exactly. Transfer the explicit default-off two-proposal policy
unchanged to all-15 once; do not add another checkpoint or proposal-count
sweep. Report:
`benchmark_results/k24_landmark_response_proposal_i60_i90_sentinel/report.md`.

All-15 I60+I90 is aggregate-strong but fails the no-loss gate. It reaches
`0.766084x` geometric and `0.732090x` summed control versus I60-only
`0.786533x/0.754474x`, but W/T/L becomes `13/1/1`: Gendarmenmarkt regresses to
`1.019557x` from `0.957620x`. Its weak `0.2475%` second-checkpoint gain is
followed by rejections increasing `2 -> 8`; Alamo shows the same weaker pattern.
Retain as a bounded-loss candidate, not the common default. Do not tune the SSE
floor. Next test only the state mechanism: keep the proven first trust rebase
but suppress the second. Report:
`benchmark_results/k24_landmark_response_proposal_i60_i90_breadth/report.md`.

Suppressing the second trust rebase does not repair the two-proposal tail.
Gendarmenmarkt remains a loss (`1.016089x` control versus `1.019557x` with both
rebases and `0.957620x` I60-only); Roman/Trafalgar also slightly worsen versus
both-rebase. The three-scene geomean is `0.831329x` first-only,
`0.829738x` both-rebase, and `0.822103x` I60-only. BAL52/3068 remain exact.
Thus repeated proposal state, not the second trust rebase, causes the tail.
Close proposal-count/checkpoint tuning and retain I60-only as common default.
Report:
`benchmark_results/k24_landmark_response_i60_i90_first_rebase_sentinel/report.md`.

Unified full-1DSfM proposal evaluation confirms the progression: ordinary
control `1.488916x` Ceres, camera-only I90 `1.321671x`, camera-only plus rebase
`1.317172x`, joint I90 `1.177568x`, promoted joint I60 `1.171081x`, and
bounded-loss I60+I90 `1.140635x`. Joint camera/landmark response is the dominant
mechanism. I60 remains the no-loss common default; Yorkminster and Tower are
the absolute quality tails, while Madrid is the sole declined 1DSfM proposal.
See `benchmark_results/k24_proposal_full_1dsfm_evaluation/report.md`.

A bounded mechanistic sensitivity on Yorkminster, Tower, and Madrid compares
half/base/double Schur damping and one/three/five landmark steps. Three steps
remain best. Half damping (`0.0029296875`) improves both tails and wins the
three-scene aggregate, then validates unchanged on the other 12 scenes:
all-15 delivered/control `0.777854x` geometric and `0.748190x` summed versus
base `0.786533x/0.754474x`; W/T/L half/base `11/1/3`. Regressions are bounded
to Gendarmenmarkt `+0.79%`, Alamo `+0.36%`, Trafalgar `+0.21%`. Madrid moves
from `0.0672%` to `0.0964%` immediate decrease but remains below the frozen
`0.1%` floor and declines. Do not continue a damping/depth sweep. Run half
damping unchanged on BAL29 before promotion. See
`benchmark_results/k24_landmark_response_parameter_sensitivity/report.md` and
`benchmark_results/k24_landmark_response_damping_half_validation/report.md`.

The final bounded decomposition shows landmark damping drives the gain:
camera-half alone is slightly worse than base, landmark-half improves base, and
quarter damping (`0.00146484375` for both terms) improves the development tails
again. Frozen held-out validation gives all-15 delivered/control `0.774687x`
geometric and `0.746302x` summed, candidate/Ceres `1.153445x`, W/T/L versus
control `14/1/0`; quarter/half is `0.995929x`, W/T/L `12/1/2`. Tower and
Yorkminster improve to `0.873032x/0.792652x` control. Madrid remains declined
and moves away from the floor, proving this is accepted-tail quality rather
than marginal activation. BAL29 remains bitwise exact with all proposals
declined. Promote quarter damping as the common global proposal setting and
close damping/depth sensitivity; do not halve again. See
`benchmark_results/k24_landmark_response_damping_quarter_validation/report.md`
and `benchmark_results/k24_landmark_response_damping_quarter_breadth/report.md`.
The canonical `run_k24_one_step_schur_proposal_breadth.sh` defaults now encode
the complete promoted preset: I60, quarter camera/landmark damping, three
landmark steps, atomic landmark response, and selected-only trust rebase.

A behavior-neutral I60 full-Schur oracle on Tower and Yorkminster reproduces
all 120 ordinary-control trajectory rows exactly. Nominal shared-camera DRS has
weighted cosine `0.154/0.305` and only `0.031/0.016x` Schur norm. The promoted
one-action proposal raises cosine to `0.585/0.536`, but reaches only
`0.382/0.437x` Schur norm and `0.623/0.728x` shared-Schur camera-model gain.
The accepted proposal then improves another `0.847/0.761x` through I120, so the
remaining tails are not caused by restart or continuation. The next mechanism
must improve broad cross-camera direction quality; do not reopen damping,
landmark depth, fixed second-Jacobi action, or correction-tail sweeps. See
`benchmark_results/k24_quarter_i60_tail_alignment_oracle_clean/report.md`.

Two preconditioned conjugate directions provide that missing mechanism. Across
all 15 1DSfM scenes, median shared-camera cosine/norm/model-gain recovery rises
from `0.592/0.496/0.917` for one block-Jacobi action to
`0.728/0.668/0.992`. With unchanged eight scales, three landmark steps, floor,
atomic commit, restart, and trust rebase, delivered/quarter is `0.985801x`,
delivered/control `0.763688x`, and delivered/Ceres `1.137067x`; W/T/L versus
quarter is `13/0/2`, with bounded Alamo/Gendarmenmarkt regressions
`1.000488x/1.003041x`. All 29 BAL proposals decline and reproduce quarter/base
exactly. The proposal-only `krylov2` path is trajectory- and state-exact to the
diagnostic applied path on all15 and all29 BAL scenes and skips the converged
Schur reference. On BAL29 it runs in `0.899325x` diagnostic elapsed time with
peak coordinator/worker RSS `6.859/10.684 GiB`. Promote `krylov2` in the
canonical proposal preset; retain lower-level `jacobi` for ablation. See
`benchmark_results/k24_quarter_i60_krylov_applied_sentinel/report.md`.

A rollback-safe physical ceiling compares the converged shared-camera Schur
tangent with Krylov2 on Gendarmenmarkt, Madrid, Tower, and Yorkminster. Full
Schur/Krylov is `0.994705x` geometrically but W/T/L `1/0/3`: it improves Tower
to `0.960723x`, regresses Gendarmenmarkt/Yorkminster to
`1.010814x/1.006836x`, and declines Madrid. All 120 ordinary trajectories and
worker-state roundtrips are exact. Therefore more linear convergence is not a
transferable proposal mechanism. Keep bounded Krylov2 and move to
basin/nonlinear-model diagnosis; do not add Krylov depth or apply full Schur.
See `benchmark_results/k24_quarter_i60_full_schur_model_fidelity/report.md`.

One frozen orthogonal interaction retests the previously closed I60+I90 count
with the new Krylov2 direction. All15 improves `0.979939x` over I60-only and
reaches `0.748367x` ordinary control / `1.114256x` Ceres, W/T/L versus I60
`11/0/4`. Tower/Yorkminster improve to `0.939289x/0.937256x` I60, while
Gendarmenmarkt/Madrid regress to `1.015436x/1.023649x`; Madrid is
`1.014837x` ordinary control. BAL29 remains bitwise exact with both proposals
declined. Retain I60+I90 as a frozen bounded-loss portfolio component, not the
common preset. Do not retune checkpoint, count, floor, scale, damping, or
Krylov depth. See
`benchmark_results/k24_quarter_i60_i90_krylov_interaction_v3/report.md`.

Continuation-aware intervention costing repairs that bounded-loss component
without changing the first proposal. The I60 Krylov2 proposal retains its
`0.1%` floor; the I90 repeat must clear a fixed `1%` immediate refined-SSE
margin. All15 selects 9/15 repeats and reaches `0.978063x` I60-only,
`0.746935x` ordinary control, and `1.112123x` Ceres, with W/T/L versus I60
`9/6/0` at `1e-8` numerical tie tolerance and versus control `15/0/0`. BAL29
remains bitwise exact with both proposals declined. Promote guarded I60+I90 as
the common canonical preset; retain I60-only as the conservative reference and
unguarded repetition as an ablation. The `1%` margin is a frozen intervention
cost, not a new floor sweep. Do not tune margin, checkpoint, count, scale,
damping, landmark depth, or Krylov depth. See
`benchmark_results/k24_quarter_i60_i90_krylov_repeat_guard/report.md`.

The missing copied-baseline interaction factorial is complete and must not be
conflated with the older C1--C5 publication matrix. Direct left-SE3 and exact
tangent-metric consistency are fixed; the 2x2 varies only shared-only
product-space semantics and guarded I60+I90 Krylov proposals on the copied
mature DRS policy. All15 1DSfM gives shared/direct `0.992877x`, proposal/direct
`0.787326x`, combined/direct `0.762902x`, proposal under shared `0.768375x`,
shared under proposal `0.968978x`, and beneficial interaction `0.975930x`.
Combined W/T/L versus direct is `14/0/1`; Madrid is the sole loss
(`1.049375x`). Shared-only alone recovery-exhausts Tower at I112, while the
combined arm completes it. On BAL29, proposal under shared is nearly neutral
but favorable (`0.999953x`, one selected proposal on BAL1723), interaction is
`0.999955x`, and combined/direct is `1.000131x`; the small cost comes from the
mandatory shared-only factor, not the proposal. Thus the K1-derived additions
have clear mutual conditional benefit on 1DSfM, mild positive interaction on
BAL, and no evidence that guarded Krylov merely compensates for shared-only.
Retain shared-only as a correctness invariant and guarded Krylov as the quality
mechanism. This is already full all15/all29 breadth; do not retune or rerun the
factorial. See `benchmark_results/k1_carryover_joint_factorial/report.md`.

Fresh isolated bridge gate (2026-08-19):
`serverTest/client_drs_k1_bridge.py` preserves `client_drs.py` and pins the
mature legacy backbone while exposing ordered `legacy`, `direct`, and
`direct_shared` arms. On fresh matched Roman/Trafalgar K24/I30, direct tangent
alone is `1.000000140x` legacy and reproduces rejection counts. It is therefore
algebraically required but behavior-neutral on this mature sentinel. Adding
shared-only semantics is material: Roman is `1.058889x`, Trafalgar is
`0.620650x`, and the pair is `0.810679x` legacy. This confirms that the worker
already uses the same Nesterov kernel as K1; copying that linear solver is not
the missing mechanism. The relevant inner-optimization surface is accurate,
safe minimization of unique cameras and landmarks under shared-only proximal
coupling. See `benchmark_results/k1_inner_on_mature_base_k24_i30/report.md`.

A fixed two-local-step arm rejects the simplest interior-accuracy repair. It is
`1.063412x` shared-only L1 geometrically at `1.584348x` time: Trafalgar changes
only to `0.991819x` L1 while Roman regresses to `1.140172x`. Do not extract a
fixed-depth C++ interior solver or broadly sweep local-step count. A future
inner policy requires a checkable relative forcing or stationarity diagnostic,
not unconditional extra local solves.

The existing normalized interior-defect diagnostic is behavior-neutral on the
fresh bridge: all SSE rows and saved cameras/points are bitwise equal with it
enabled. It is not predictive enough by itself. Roman and Trafalgar have
similar median defect histories, each activates six clusters under the old C5
window-three threshold, and both decay to about `0.0127` median by I30 despite
opposite L2 effects. Do not revive that defect-threshold controller as the new
solver policy.

Broader forcing reliability (2026-08-19): behavior-neutral L1 diagnostics and
fixed-L2 controls completed on six development 1DSfM scenes plus BAL1490/3068.
For `q=sqrt(interior defect/proximal displacement)>1`, TP/FP/FN/TN is
`3/4/1/0`, precision `0.429`, recall `0.750`, sign accuracy `0.375`, and
maximum-q Spearman correlation with L2 benefit `-0.071`. Reject q as a
standalone adaptive-depth trigger. Full L2 is not a clean interior-only
intervention because it also re-updates shared cameras, so this does not reject
relative inexact-prox theory; it rejects wiring the current q directly to the
existing `local_steps=2` path. See
`benchmark_results/k1_inner_on_mature_base_k24_i30/forcing_reliability.md`.

Mature endpoint oracle: one exact K1/no-proximal step from the integrated
K24/I90 endpoint reaches `0.699920x` on Roman and `0.982342x` on Trafalgar,
`0.829193x` geometrically. The saved-state initialization agrees within
`9e-12` relative SSE. Since fixed L2 hurts Roman while the coordinated K1 step
finds 30% descent, Roman is not blocked by insufficient interior-only work; it
needs coordinated camera descent. Trafalgar is nearly stationary after its
large shared-only gain.

The matched distributed endpoint correction is now complete. One frozen
safeguarded shared-camera Schur correction, starting camera/landmark damping at
`0.005859375` and doubling only after rejection, improves all 15 1DSfM mature
endpoints to `0.793908x` geometrically and `0.770181x` by summed SSE. The
corrected 1DSfM endpoint is `1.267579x` Ceres geometrically. Unchanged BAL29
transfer is safe but small: corrected/actual-prestate is `0.999518x`
geometrically and `0.999652x` by summed SSE, with W/T/L `26/3/0`; three
nonconverged cases safely no-op. Of 26 accepted BAL solves, 24 use the initial
damping and two accept fallback damping `0.09375`; every accepted solve has
positive model gain and relative residual below `1e-6`. No scene-specific
routing is used. Six large BAL checkpoint files are corrupt and were recovered
by exact I90 reruns with the correction applied in-process. See
`benchmark_results/k1_mature_i90_endpoint_oracle_i1/report.md`.

This passes the cross-family safety gate and identifies coordinated global
camera descent, not interior local depth, as the useful mechanism. The next
mechanism question is how to invoke the same safeguarded distributed move
during late DRS without turning it into a repeated centralized polishing tail.
The phased implementation, development/held-out gates, artifact layout, and
crash-resume rules are frozen in `K1_CARRYOVER_TEST_PLAN.md`. Its immediate
Phase 0 is to port and commit the detached zero-iteration validation and runner
damping controls before implementing coherent correction rebuild behavior.

Late-correction restart gate (2026-08-21): a fresh same-executable I90 control
and an I90-horizon run stopped after 60 ordinary iterations have identical
first-60 trajectory rows. Applying the frozen correction at that point and
canonically restarting DRS for I30 gives `0.933819x` the terminal correction on
Roman/Trafalgar but `1.010695x` on BAL52/3068. Although all four restart arms
descend from the corrected I60 state, the common cross-family gate fails. Close
late scheduling without timing/trigger tuning; retain the terminal one-step
correction. The active next test is only the rollback-safe shared-fixed
interior trial. See
`benchmark_results/k1_carryover_late_correction/phase2_restart_oracle_i90_stop60/report.md`.

Shared-fixed interior gate (2026-08-21): one default-off worker trial zeros all
shared-camera directions, jointly updates only unique cameras and landmarks,
and backtracks on exact local data cost. Forced rejection reproduces Roman I2
trajectory, cameras, points, and endpoint bitwise; accepted and rejected trials
move shared cameras by exactly zero. On the frozen 6+5 development cohort it
improves 1DSfM to `0.921785x` control at `1.133425x` optimization time, but BAL
regresses to `1.071668x`; BAL52 is `1.043313x`, and BAL3068 recovery-exhausts at
I15 with `1.353560x` delivered SSE. Reject without tuning or breadth expansion.
This closes further K1 carryover: retain the already integrated algebra,
shared-only semantics, and one terminal distributed Schur correction. See
`benchmark_results/k1_carryover_late_correction/phase4_interior_development/report.md`.

Terminal-correction scaling sentinel (2026-08-21): the frozen K24 correction
policy transfers unchanged to the K4/K16 Stage-C endpoints. On
Roman/Trafalgar, corrected/prestate is `0.822336x` at K4 and `0.775883x` at
K16. On BAL52/3068 it is `0.976151x` and `0.973048x`; all eight corrections
accept at initial damping and converge below `1e-6`. BAL3068 required exact
in-process I30 reruns because its saved K4/K16 states are corrupt. The active
gate is full all-15/all-29 transfer with physical-SSE state validation and
in-process recovery, frozen in `TERMINAL_CORRECTION_SCALING_PLAN.md`.

Terminal-correction K4/K16 breadth (2026-08-21): the unchanged policy passes
all four complete cohorts. K4 correction/prestate is `0.785461x` on all-15
1DSfM and `0.997869x` on all-29 BAL, with accepted/no-op `15/0` and `26/3`.
K16 is `0.778120x` and `0.997663x`, with `15/0` and `27/2`. Corrected/Ceres is
`1.612187x`/`1.005041x` at K4 and `1.523013x`/`1.007594x` at K16. No accepted
solve is nonconverged; maximum residual is `9.971e-7`. Correction time totals
`17.394/16.097s` on 1DSfM and `608.228/521.372s` on BAL for K4/K16. Keep raw
K4/K16 as DRS-only resource/latency endpoints; add separately labeled terminal
correction variants. See
`benchmark_results/terminal_correction_scaling_all/report.md`.

Copied-baseline combined-stack K4/K16 transfer (2026-08-23): the frozen
shared-only plus guarded I60+I90 Krylov stack completed all 22 development
`(scene,K)` rows with no recovery exhaustion. Relative to the copied direct
baseline, geometric SSE is `0.714243x`/`0.788556x` on six 1DSfM scenes and
`1.001972x`/`0.997714x` on five BAL scenes at K4/K16. K4 wins all six 1DSfM
scenes; K16 wins five but Gendarmenmarkt is `1.022191x` control, exceeding the
frozen `1.02x` per-scene tail bound. The development gate therefore fails.
Close this common K4/K16 transfer without held-out/all15/all29 expansion or
K-specific retuning. This does not reverse the accepted K24 combined stack,
the mandatory algebraic carryovers, or the separately accepted terminal
correction scaling result. See
`benchmark_results/k1_carryover_scaling_k4_k16/development_report.md`.

Do not implement an adaptive forcing policy from these eight outcomes. The only
remaining targeted experiment in this direction is a default-off true
interior-only trial that holds shared cameras fixed, rolls back unless the local
proximal objective decreases, and proves rejected trials are state-neutral.
That worker change is higher risk and requires an explicit snapshot/rollback
test before a solver gate.

Do not broaden this two-scene gate as another shared-only breadth campaign;
all-15 structural factorials already exist. Any new solver code should be
implemented behind the isolated bridge front end and change only interior
forcing/accuracy, with the fresh `legacy` and `direct` arms retained as
controls.

Do not reopen broad K1 parameter, trust-radius, tolerance, or basin sweeps.
Do not replace `main.cpp`/`client_drs.py` with Ceres or BAE. The active task is
to consolidate the later base-backbone and publication evidence before any new
solver behavior. Do not extend this early fixed one-step preset beyond I1000
or tune its scalar trust/curvature limits.

The K2 reset/preserve/checkpoint portfolio campaign is closed historical
evidence, not the active implementation direction. It showed that the BAE
package can create valuable early distributed states, but useful duration and
trust continuation are scene-dependent. The exact staged safety portfolio
reaches `0.954453x` baseline SSE over all 15 at about `2.246814x` optimization
work; this is a diagnostic oracle, not a practical inner solver. Its checkpoint
implementation did not survive into the maintained post-crash source and must
not be reconstructed or extended unless explicitly requested.

The 2026-08-11 C4 telemetry smoke also established a narrower recovery
boundary: the maintained source runs the recorded shared-only direct-tangent
K2 configuration, but does not reproduce the historical Roman trajectory.
Keep that historical JSONL immutable as evidence. Before the matched C4
comparison, freeze a separately named executable post-crash baseline; do not
attribute its trajectory to the historical source state.

### Frozen Next Gate

Use the maintained C4 surface in `main.cpp` and `client_drs.py`:

1. freeze a named current K>1 baseline and its exact configuration/artifacts;
2. compare Schur-PCG and BAE-style finite Nesterov on the same direct-tangent,
   damped, preconditioned local system with residual, Hessian-vector work,
   model decrease, trust acceptance, and nonlinear step telemetry;
3. test the justified inner policy **inner-only** at K2, then K24;
4. keep outer acceleration off for this transfer gate;
5. only after the inner-only result, test isolated C1/C5 interactions.

K2 gate result (2026-08-11): on the separately named post-crash Roman and
Trafalgar `K2/I30` control, Schur-PCG reaches `0.9165x` and `0.7992x` the
matched finite-Nesterov SSE while taking `0.8391x` and `0.8324x` its
optimization time. This passes the two-scene safety/usefulness gate without
tolerance tuning.

K24 transfer result (2026-08-11): with the same frozen configuration,
Schur-PCG reaches `0.9789x` and `0.9042x` the Nesterov SSE while taking
`0.8732x` and `0.8672x` its optimization time. Rejection counts are identical
and PCG uses materially fewer inner iterations. C4 therefore passes the K24
transfer gate. The active next check is unchanged six-scene K24 breadth before
isolated C1/C5 composition; do not tune the inner tolerance from these scenes.

Six-scene breadth result (2026-08-11): PCG wins SSE on `3/6`, runtime on
`6/6`, summed SSE at `0.9745x`, geometric-mean per-scene SSE at `0.9948x`, and
geometric-mean optimization time at `0.8979x` Nesterov. Piccadilly regresses to
`1.0913x` SSE, so this is not a uniform quality promotion. The active decision
gate is one unchanged all-15 K24 comparison. Do not tune from the six scenes
and do not introduce a scene-dependent solver switch.

All-15 C4 decision (2026-08-11): PCG reaches `0.9851x` summed SSE but
`1.0055x` geometric-mean per-scene SSE, with `8/15` quality wins and a
`1.1953x` NYC Library regression. It reaches `0.9317x` geometric-mean
optimization time and wins runtime on `12/15`. Therefore PCG is retained as an
independently switchable speed-oriented C4 level, not promoted as the universal
inner solver. Nesterov remains the named unchanged default trajectory. The
inner-only gate is closed. The retained Stage-C C1/C5 publication factorial is
a separate PCG/regularization family; do not reinterpret its ratios as a
Nesterov composition. No tolerance tuning or scene-dependent solver selection
is permitted from this matrix.

Fresh Stage-C cross-family confirmation (2026-08-11): current-source K24/I30
plain and C1+C5 rows complete on all 15 1DSfM and all 29 BAL scenes, with no
recovery exhaustion. Relative to fresh plain DRS, C1+C5 reaches `0.870061x`
geometric SSE on 1DSfM (13/15 wins) and `0.983638x` on BAL (28/29 wins), at
`1.801873x` and `2.030313x` optimization time. Relative to left-SE3 Ceres it
reaches `2.197218x` and `1.010071x`.

The complete global 2x2 factorial establishes the cumulative order. C1 is the
accepted first rung (`0.865147x` plain SSE on 1DSfM and `0.983780x` on BAL).
Current C5 alone is `1.015807x` and `1.000057x`; adding it to C1 gives
`1.005680x` C1 on 1DSfM and `0.999855x` on BAL. Both C1 and C5 remain in the
final architecture and C1+C5 is the intended combined stack. The active
research task is global C5 threshold/work-policy tuning to make its incremental
gain over C1 robust across both families. Never use per-scene settings. The
frozen one-factor tuning order and acceptance gates are in
`benchmark_results/stage_c_global_c5_tuning_plan.md`.

Global C5 tuning result (2026-08-11): scalar threshold, window, dwell, and
maximum-depth sweeps did not pass both development families. Delaying C5 until
I5 did. Freeze one common policy everywhere: start I5, high/low `0.35/0.20`,
window/dwell `3/3`, maximum depth 2. It was selected on six development 1DSfM
plus five BAL sentinels, then run unchanged on nine held-out 1DSfM and all-29
BAL. Tuned C1+C5/C1 is `0.980053x` all-15 1DSfM and `0.999486x` all-29 BAL;
C1+C5/plain is `0.847890x` and `0.983275x`, W/T/L `14/0/1` and `28/0/1`.
Promote tuned C1+C5 as the final Stage-C stack, with C1 and C5 retained as
independent switches. No scene-specific settings are used.

Deterministic repeat gate (2026-08-11): after one warm-up, three tuned C1+C5
repeats on Roman, Trafalgar, BAL52, and BAL3068 are endpoint-bitwise identical,
including rejection and oracle counts. Optimization-time CV is `1.1%--2.4%`
and overall-time CV is `0.9%--2.5%`. The tuned L2 stack is frozen. The active
next gate is an identical Huber objective contract across distributed worker,
coordinator safeguards/evaluator, and Ceres reference; evaluation-only Huber
scoring is not sufficient.

Huber gate result (2026-08-11): the shared delta-`0.5` contract is implemented
and validated. The custom worker uses explicit observation-level IRLS weights
and exact robust trust costs; worker/coordinator and Ceres/evaluator objectives
agree numerically. Raw-start Huber I30 remains far behind Ceres (`7.8002x`
1DSfM, `1.2674x` BAL on the sentinel pairs). L2-to-Huber continuation gains
only `0.997919x` over its start on 1DSfM and `0.990092x` on BAL while adding 30
outer iterations, and remains `2.5698x`/`1.1679x` Ceres. Retain robust support,
but do not broaden either current policy. Tuned L2 C1+C5 remains promoted.
See `benchmark_results/stage_c_huber_sentinel_report.md`.

K>1 scaling gate (2026-08-11): the frozen tuned L2 C1+C5 stack was swept at
K2/K4/K8/K16/K24 on Roman, Trafalgar, BAL52, and BAL3068, then K4 and K16 were
confirmed unchanged on all 15 1DSfM and all 29 BAL scenes. K16/K4 geometric SSE
is `0.953599x` on 1DSfM and `1.002768x` on BAL, while optimization time is
`0.584x`/`0.510x`. K16 raises worker CPU to `1.456x`/`1.568x` and traffic to
`1.927x`/`1.985x`. Retain K4 as the global resource endpoint and K16 as the
global latency endpoint. These are two deployment budgets, not per-scene
settings. K24 is not promoted. See
`benchmark_results/stage_c_scaling_confirmation_k4_16_i30/report.md`.

K4/K16 repeat gate (2026-08-11): after one excluded warm-up, three measured
repeats on Roman, Trafalgar, BAL52, and BAL3068 are SSE-bitwise identical in
`7/8` `(scene,K)` cases; the sole Trafalgar K16 variation is `2.956e-09`
relative. Rejection/oracle counts and traffic are identical throughout, and
maximum optimization-time CV is `1.75%`. Mean K16/K4 optimization ratios are
`0.5158x` on 1DSfM and `0.4190x` on BAL with ratio CV below `0.5%`. Keep K4 and
K16 frozen as the global resource and latency endpoints. See
`benchmark_results/stage_c_scaling_repeats_k4_16_i30/report.md`.

Publication comparison correction (2026-08-12): the cohort-explicit final table
now explicitly restores the best base DRS rows. On all-15 1DSfM,
the best BAE-style K1/I90 diagnostic is `0.993529x` Ceres and the secondary
Schur-PCG K1/I80 diagnostic is `1.022987x`. The preserved K24/I200 base is
`1.457017x` Ceres; frozen K4/K16 I30 are `2.052536x`/`1.957296x`, or
`1.408725x`/`1.343358x` base SSE with `0.280585x`/`0.163860x` base optimization
time. On all-29 BAL, preserved K24/I90 is `0.998404x` Ceres; K4/K16 are
`1.008920x`/`1.011713x` its SSE at `0.738821x`/`0.376767x` its optimization
time. Thus C1+C5 is a valid matched-I30 improvement, but K4/K16 are speed
endpoints, not quality replacements for best base DRS. Verified BAE is only a
six-scene RTX 5090 inset; no cross-hardware speedup is claimed. See
`benchmark_results/stage_c_publication_comparison/report.md`.

Base-backbone hybrid gate (2026-08-12): matched base-I30 analysis showed tuned
C1+C5 was `1.129336x` base on all-15 1DSfM but `0.995443x` on all-29 BAL. The
numerator is the tuned Stage-C K24/I30 C1+C5 endpoint. The denominator is the
per-scene best SSE within the first 30 iterations of the preserved legacy
K24/I200 1DSfM or K24/I90 BAL trajectory, not Ceres and not the fresh Stage-C
plain arm. Base-I30 is `1.896006x`/`1.014321x` Ceres and tuned C1+C5 is
`2.141229x`/`1.009698x`; hence `2.141229/1.896006=1.129336` and
`1.009698/1.014321=0.995443`. A
K24/I30 factorial restored local Nesterov, persistent DRS trust, curvature
`0.4` recovery/decay, and camera metric `75`. C1 was the transferable component;
C5 alone was neutral. The structural 2x2 then isolated proposal damping and
shared-only proximal semantics. With C1 fixed, shared-only proposal damping
applied only to duplicated cameras reaches `0.992977x` base-I30 on 1DSfM and
`0.997428x` on BAL, completes all `15+29` scenes, and preserves the maintained
product-space architecture. This is the first cross-family equal-I30 hybrid
win. It is not promoted as a final endpoint: permanent I90 reaches
`1.017187x`/`1.001950x` base with Tower recovery exhaustion; proposal cutoff
I30 reaches `1.022809x`/`1.001478x`; cutting both proposal damping and C1 at
I30 reaches `1.136224x`/`1.009261x`. The remaining problem is late-trajectory
continuation, not initial component compatibility. See
`benchmark_results/stage_c_base_structure_factorial_k24_i30/report.md` and
`benchmark_results/stage_c_shared_proposal_c1_k24_i90/report.md`.

Outer-acceleration and global-Schur gate (2026-08-12): on matched Roman,
Trafalgar, BAL52, and BAL3068 K24/I30 sentinels, safeguarded Themelis Nesterov
is the best tested outer accelerator and completes all four cases. One final
distributed global Schur correction from its best checkpoint improves all
four; the reductions are `5.075%`, `3.223%`, `0.016%`, and `0.048%`.
Allowing up to three accepted corrections with the same global damping and
acceptance policy gives another `12.021%`/`5.244%` after correction one on
Roman/Trafalgar, while BAL52/BAL3068 stop after correction two for only
`0.00071%`/`0.00836%` additional gain. The up-to-three endpoint is
`0.875121x` the uncorrected I30 state on the 1DSfM pair and `0.999636x` on the
BAL pair. Retain this as separately labeled polishing, not a DRS-core gain.
The four-case operator gate showed that `bsr_low_memory` preserves every
acceptance and termination decision while reducing BAL3068 correction time
from `87.90s` to `23.18s`; correction-three endpoint drift versus Python is
`+0.190%` on Roman and `+0.043%` on Trafalgar. Freeze `bsr_low_memory`, a
three-correction cap, and the predeclared `1e-3` relative-progress stop for
breadth confirmation. See
`benchmark_results/stage_c_themelis_final_schur3_k24_i30/report.md` and
`benchmark_results/stage_c_themelis_final_schur3_bsr_low_memory_k24_i30/report.md`.

Final-Schur breadth confirmation (2026-08-12): the frozen policy completed all
15 1DSfM and all 29 BAL scenes with no scene-specific settings. All 15 1DSfM
scenes accept all three corrections and reach `0.864158x` their best I30
handoffs, `0.858090x` base-I30, and `1.626944x` Ceres. This is a strong
polishing result but remains `1.116626x` the established I200 base. On BAL,
27/29 scenes stop after one correction and only BAL135/BAL142 accept a second;
the endpoint is `0.999320x` its I30 handoff, `0.996749x` base-I30,
`1.011023x` Ceres, and `1.012640x` the established I90 base. Peak coordinator
RSS is `12.134 GiB` on BAL961. Retain bounded global Schur as separately
labeled 1DSfM-oriented polishing; it does not replace either long-horizon base
and its BAL gain is too small for the added memory. Do not retune from this
confirmation cohort. See
`benchmark_results/stage_c_themelis_final_schur3_bsr_low_memory_stop1e3_all15_all29_k24_i30/report.md`.

Deeper-Schur and I100 diagnosis (2026-08-12): extending the frozen all-15
1DSfM I30 handoff from three to ten `bsr_low_memory` corrections resolves the
apparent quality gap. All 15 scenes accept all ten corrections. Correction
eight crosses the established I200 base (`0.996373x`), and correction ten
reaches `0.968846x` base-I200, `0.749791x` its I30 handoff, and `1.411625x`
Ceres. I30 DRS plus the tail takes `188.491s`, `0.329301x` the established
I200 base optimization time (`572.399s`), making this a genuine Pareto
improvement. Every accepted CG solve converges and physical SSE decreases on
every scene. Median damped model gain falls below one after correction seven,
indicating declining model calibration, but not a broken correction because
independent SSE acceptance remains positive. The three-correction gate was
under-polished; the ten-correction cap is still active everywhere, so this is
a quality point rather than convergence. See
`benchmark_results/stage_c_themelis_final_schur10_bsr_low_memory_stop1e3_all15_k24_i30/report.md`.

The separate I100 acceleration diagnostic uses Madrid, Montreal, Roman, Tower,
and Trafalgar with bitwise-identical prefixes through I100 and proposal damping
retained. Permanent acceleration reaches `1.052528x` their established I200
base; an isolated acceleration-only reset at I100 is aggregate-neutral at
`1.051481x`. Disabling acceleration after I100 is already `1.02014x` permanent
at the fully matched I150 checkpoint and later exhausts recovery on Trafalgar.
Keep acceleration active; stale momentum history is not the cause of the late
gap. Do not broaden the isolated reset. See
`benchmark_results/stage_c_i100_acceleration_diagnostics_k24_i200/report.md`.

Frozen Schur-budget frontier (2026-08-12): with damping, CG tolerance,
operator, safeguards, and the `1e-3` progress stop unchanged, a five-scene
development run remains productive through correction 20. The cap was frozen
before all-15 confirmation. The all-15 cap-20 run reproduces the cap-10 I30
handoffs and first ten corrections exactly, completes every scene, and reaches
`0.673164x` its I30 handoff, `0.869832x` the established I200 base, and
`1.267361x` Ceres. Combined I30 DRS plus Schur work is `385.234s`, or
`0.673017x` base-I200 optimization time. Seven scenes stop naturally after
13--20 corrections; eight remain cap-limited. Retain two global Pareto presets:
cap 10 gives `0.968846x` base-I200 at `0.333964x` its time, while cap 20 gives
`0.869832x` at `0.673017x`. Neither dominates. Do not tune damping, linear
tolerance, or the progress threshold from this cohort. See
`benchmark_results/schur20_all15/report.md`.

Equal-time DRS/Schur allocation (2026-08-12): a frozen five-scene gate followed
by unchanged all-15 confirmation compares I60+10 Schur, I30+15 Schur, and
I5+17 Schur at approximately 300 seconds total optimization work. Shared
prefixes are bitwise identical and safeguard annealing is fixed at I30. All 45
all-15 runs complete without recovery exhaustion. Against the nearest
equal-time base-DRS checkpoints, I60+10 reaches `0.819300x` SSE at `299.233s`,
I30+15 reaches `0.852585x` at `301.016s`, and I5+17 reaches `0.951988x` at
`289.837s`; time mismatch is at most `0.24%`. I60+10 is `0.961553x` I30+15
and `0.858391x` I5+17, winning 10/15 pairwise in both comparisons. Retain
I60+10 as the balanced-budget preset. Very early handoff leaves a poorer basin
and makes Schur work more expensive. This does not replace the lower-budget
I30+10 fast preset or higher-budget I30+20 quality preset. See
`benchmark_results/equal_time_all15/report.md`.

With the later behavior-exact BSR systems optimizations, a fresh I60+10 run
takes `289.498s` total and reaches `0.817178x` the nearest equal-time base-I101
reference (`289.730s`), preserving the balanced-preset decision.
The named preset runner repeats every endpoint and Schur decision bitwise; its
second measured total is `283.324s`, with `2.53%` per-scene time-ratio CV.

Quality-budget allocation (2026-08-12): a frozen five-scene I60+20 frontier
selected I60+16 as the allocation nearest the existing I30+20 quality budget.
Unchanged all-15 confirmation completes every scene and reaches `0.834064x`
base-I200 and `1.215246x` Ceres in `381.521s`. Relative to I30+20, I60+16 is
`0.958879x` geometric SSE and `0.955439x` summed SSE at `0.990360x` measured
optimization time, with 10/15 wins. Promote I60+16 as the quality preset; retain
I30+20 only as the frozen earlier-handoff frontier reference. See
`benchmark_results/i60_s16_all15/report.md`.

I60 performance and tail diagnostics (2026-08-12): current-source Roman and
Trafalgar local profiles show assembly (`0.609s/1.341s`) exceeds Nesterov
(`0.253s/1.006s`) on the local critical path; repeated $W^T/W$ actions dominate
the Nesterov kernel. A default-off one-pass dual-coordinate assembly reduces
local critical time `7.0%/4.6%` and all-15 total work `6.3%`, but changes every
Schur decision sequence and regresses all-15 geometric SSE `0.584%` with a
`4.40%` worst tail. The code was removed; worker speedups must preserve the
metric and tangent systems exactly by construction. Extending unchanged
I60+16 to I60+20 gives a separate higher-budget point: `0.813073x` base-I200
and `1.184661x` Ceres in `430.320s`, improving I60+16 by `0.974832x` SSE at
`1.130382x` time. Keep I60+16 as the named quality preset. Tail geometry is
not uniform: Tower has `12.8%` low-8 camera-center mode energy versus base and
improves another `11.67%` from corrections 17--20, while Madrid has only
`0.045%` low-8 energy and improves `0.93%`. Treat Tower as under-polished and
Madrid as a separate non-low-mode basin problem. See
`benchmark_results/i60_performance_tail_diagnostics/report.md`.

Conflict-free camera/landmark adjacency traversal preserves I10 trajectories
bitwise but slows Roman/Trafalgar optimization `3.3%/2.4%`; iterative $W^T$
time rises `14%/42%` because camera-sorted edge storage has better
camera-vector locality. That experiment was also removed. Do not revisit edge
reordering without a cache/layout design that preserves the current locality.

Four-scene polishing ceiling and Madrid diagnosis (2026-08-12): unchanged
I60+30 on NYC, Piazza, Roman, and Tower reaches `0.706069x` their base-I200 SSE
and `1.130525x` Ceres. Corrections 21--30 improve cap 20 by `0.923502x` in
`17.274s`; Piazza stops at 30 and Roman at 23, while NYC and Tower remain
cap-limited. Keep this targeted ceiling separate from global presets. Madrid's
I60+16 endpoint is `1.251524x` base, `1.314266x` Ceres, and `1.150461x`
BAE-style K1. Its first eight camera-center graph modes explain below `0.1%`
of the gap to every reference, and its heavy residual concentration is similar
to all three references. Cameras 177/171/173 repeatedly dominate
observation-normalized excess, but camera/point state swaps are catastrophic and
weak radial/translation coordinates are ill-conditioned. Treat Madrid as a
coupled non-low-mode camera/point basin centered on those tracks, not as a
global prior, scalar subspace, or polishing-budget problem. See
`benchmark_results/i60_s30_ceiling_four/report.md` and
`benchmark_results/madrid_tail_diagnostics/report.md`.

Tower-only continuation stops naturally at correction 53 with `0.701186x`
base-I200 and `1.316613x` Ceres after `12.151s` Schur work. Its last accepted
gain is `0.0715%`. Tower is no longer blocked by correction budget; its
remaining Ceres gap is also a basin ceiling.

COLMAP weak-mode transfer (2026-08-12): the fixed K24/I60+16 quality preset
does not transfer to Graham Hall's low graph-mode 20-pixel perturbation. It
reaches `1.002113x` the historical additive I30 SSE and retains `0.831443` of
the injected mode versus `0.459892` for additive I30. An isolated restart from
the old I30 state leaves the handoff SSE unchanged; Schur then improves SSE by
only `0.0159%` and changes mode retention by `1.000013x`. Do not broaden this
screen. Pixel-only global Schur cannot identify the nearly unobservable
deformation; independent geometric evidence would be required. See
`benchmark_results/colmap_graph_mode_schur_pilot_quality/report.md`.

Pose-prior selector compression (2026-08-13): complete artifact replay confirms
that correction nine is the first loss-free physical-SSE branch decision, at
`1.900388x` estimated balanced-workflow time. A development-selected I5 rule
requiring a 5% prior lead reaches `0.993792x` raw on the six-scene development
split without losses, but frozen held-out evaluation is W/T/L `4/3/2`:
Notre Dame and Yorkminster reverse despite strong I5 leads. Reject the early
branch race and do not retune on held-out outcomes. A new basin mechanism must
be robust without selection or use independently justified information beyond
pixel SSE. See `benchmark_results/pose_prior_early_selector_replay/report.md`.

Repeated initial distributed Schur (2026-08-13): the bootstrap now supports a
default-off bounded correction count; one correction remains the default. The
global cap-three low-memory policy is strong at I5 on large BAL (`0.928729x`
preserved raw geometrically), with strict rejection of nonconverged solves.
Matched current-source I30 on active 1490/3068 is `0.958353x` cap one but costs
`1.791232x` time and has a bounded 1490 loss. Composed with the named I60+16
workflow, Roman improves to `0.962247x` control but Trafalgar regresses to
`1.081335x`; geometric SSE is `1.020055x` at `1.008093x` time. Retain the
mechanism only as short-handoff diagnostic infrastructure. Do not broaden or
tune correction count. See
`benchmark_results/initial_schur_repeat_gate/report.md`.

Schur performance and orthogonal quality gates (2026-08-12): complete
per-attempt phase telemetry is live. On all-15 I60+10, coordinator numeric BSR
accumulation takes `28.755s` and CG `23.106s`, while symbolic graph construction
takes only `0.314s`. A pattern-fingerprinted symbolic BSR cache preserves every
accepted candidate and endpoint bitwise on all 15, with 153 hits/15 builds and
reduces Schur time from `105.612s` to `102.351s` (`0.969122x`). Promote this
behavior-exact cache. Duplicate-safe vectorized numeric accumulation is also
all-15 bitwise exact, cuts numeric assembly from `28.755s` to `14.261s`, and
reduces total Schur time further to `94.206s` (`0.920428x` cached). Promote both
systems changes. A default-off seven-mode projection-preserving similarity
preconditioner cuts Trafalgar CG iterations `507 -> 238` and Schur time
`37.749s -> 31.211s`, but regresses Roman/Trafalgar SSE by `0.764%/1.902%`;
matched `1e-8` tolerance does not repair the path (`1.034446x` tight-Jacobi SSE
on Trafalgar and only seven accepted corrections). Gauge staging followed by
Jacobi polishing to the original `1e-6` residual also fails: Roman is slower,
while Trafalgar is `1.012483x` promoted-Jacobi SSE. Remove that mode and close
preconditioner search for now; retain plain gauge only as a conditioning
diagnostic. The frozen removable relative-pose
prior improves fixed-policy all-15 geometric SSE to `0.989513x` control but
worsens summed SSE to `1.001376x`, with Trafalgar `+5.062%`; do not integrate it
unconditionally. The historical I60 pixel-SSE selector plus Schur10 reaches
`0.981565x` control but has a Notre Dame post-polishing reversal (`+0.517%`).
Selection after correction nine is loss-free but costs about `1.9x` the
balanced workflow, so it is an oracle rather than a practical policy. Retain
the prior as an orthogonal basin proposal and require a cheaper post-Schur-safe
global selector before promotion. See
`benchmark_results/schur_performance_quality_gates/report.md`.

Loss-factorial clarification (2026-08-11): fallback safety does not imply
endpoint dominance over a separately run plain trajectory. Roman's C1 and C5
main effects both lose but interact beneficially; Yorkminster's main effects
both lose and interact adversely; BAL135 is a C1-only loss because C5 is
inactive. Under a prefix-identical I30 safeguard schedule continued to I60,
Yorkminster C1 becomes `0.908028x` plain while C5 and C1+C5 remain `1.015475x`
and `1.029215x`. Retain the innovations, keep the I30 tail warning, and do not
introduce scene-dependent switching. These are global tuning diagnostics, not
innovation vetoes. Full evidence is in
`benchmark_results/stage_c_loss_factorials_k24_i30_report.md`.

The goal is consensus-aware finite local work, not convergence of each local
objective and not another multi-branch basin portfolio. Ceres/BAE remain
diagnostic references. The historical baseline artifact remains unchanged;
the separately named post-crash executable baseline is the control for new C4
runs.

## 2. Non-Negotiable Method Constraints

1. Keep camera and landmark optimization distributed across clusters.
2. Keep shared-only product-space DRS: only duplicated cameras participate in
   the DRS proximal metric and consensus projection.
3. Preserve independent physical global-SSE safeguards.
4. Preserve the converged-PCG acceptance gate for global Schur corrections.
5. Use direct left-SE3 tangent normal-equation assembly where enabled; the old
   post-Hessian transform is numerically invalid.
6. Do not claim a DRS improvement from a centralized or repeated-Schur tail.
7. Do not promote a policy that only works on 1DSfM or only on BAL.
8. Do not weaken safeguards to make a run complete.
9. Do not use BAL-49 to select, tune, reject, or promote research mechanisms.
   It is permitted only as a fast protocol/build smoke test.

## 2A. Publication Integration Contract

`admm_drs_baseline_plan.md` remains the publication ablation plan. The current
quality work must feed back into its Stage-C stack rather than create a second
method. The maintained implementation is `serverTest/client_drs.py`, with
plain DRS and every claimed innovation available as explicit runnable modes.

The intended contribution stack is:

1. **C1 safeguarded fast DRS:** Themelis direction, merit line search, plain-DRS
   fallback, restart, exact best-state restoration, and finite/physical-SSE
   safeguards. The K24/I5 breadth gate gives all-15 1DSfM `0.873033x` matched
   block SSE, W/T/L `7/8/0`, at `1.019837x` optimization time and `1.138767x`
   prox-oracle count. BAL 1490/1778/3068 are `0.892284/0.876836/0.600108x`,
   geometric mean `0.777231x`, at `1.137850x` overall time. Promote C1 as the
   current cross-family cumulative base.
   The matched K24/I30 gate confirms long-horizon 1DSfM transfer: all-15 SSE
   `0.850595x`, W/T/L `13/0/2`, optimization `1.541091x`, proximal oracles
   `1.673801x`, and true worker CPU `1.726258x`; worst is Trafalgar
   `1.045075x`. Large-BAL endpoint gains largely decay by I30: BAL
   1490/1778/3068 are `0.994416/0.977676/1.002193x`, geometric mean
   `0.991375x`, at `1.753382x` true worker CPU. Retain C1 as a publication
   quality component and cumulative base, but do not describe it as a universal
   practical speedup or a meaningful long-horizon BAL improvement.
2. **C2 coordinate equilibration:** no scaling versus initial Jacobi today;
   Ruiz and block-coordinate transforms remain planned and cannot yet be
   claimed.
3. **C3 variable-metric consensus:** full 9x9 shared-camera blocks, exact metric
   projection, direct left-SE3 tangent assembly, and the transient
   observability-selected Schur majorizer. Exact sparse cross-camera consensus
   projection is now implemented as infrastructure, but remains unreachable
   unless the same packed sparse metric is used by worker prox and coordinator
   consensus. The coherent opt-in path now exists with Schur-PCG. Fixed
   Frobenius edge stabilization gives Roman K24/I5 `0.554578x` the matched
   block-diagonal SSE, at `4.2949x` runtime and `14.5142x` received bytes.
   Materialized camera-pair blocks exhaust memory on Trafalgar K24/I1. The
   corrected matrix-free factorized implementation now runs on Trafalgar and
   uses the identical metric in worker prox, consensus, residuals, and DRE.
   With startup observability selection and exact bucketed stabilization, its
   fixed-factor all-15 K24/I5 ratio is `0.745081x` the matched block control,
   W/T/L `11/2/2`; 13/15 scenes select coupling, while Ellis Island and Montreal
   Notre Dame select raw and tie bitwise. Gendarmenmarkt and Tower regress by
   `15.09%` and `9.52%`. BAL 52/245 select raw and reproduce I1--I5 SSE bitwise.
   Retain as a strong C3 quality-for-cost component; all-15 costs are
   `2.506342x` runtime, `4.022384x` received bytes, and at most 1.382 GiB
   coordinator RSS. The original Roman/Trafalgar/Vienna ratio remains
   `0.527617x` at `3.533440x` runtime and `5.458339x` received bytes.
   Cached fixed-metric block-Jacobi inverses and previous-consensus PCG warm
   starts preserve the exact system and strict tolerance. On Trafalgar I5 they
   reduce the compiled fixed-factor overall time from `37.049s` to `17.590s`
   and projection time from `20.460s` to `9.419s`, with endpoint variation
   inside measured same-code worker repeatability.
   Large-BAL transfer is negative: BAL 1490/1778 select coupling but end at
   `1.830125x/1.759679x` matched block SSE, while BAL 3068 selects raw and ties
   bitwise. Both selected scenes improve at I1 (`0.642929x/0.652024x`) and cross
   above raw at I2. Do not promote fixed C3 as a cross-family default or retune
   its threshold on this gate; test frozen composition with C1/C5.
4. **C4 local linear solver:** matched Schur-PCG versus synchronization-reduced
   Nesterov, with residual, model-decrease, and work telemetry.
5. **C5 inexact local work:** fixed local depths, persistent trust state, and
   outer-residual-driven adaptive depth.
6. **C6 optional globalization:** the independently safeguarded distributed
   Schur bootstrap. This is an initialization layer, not a DRS-iteration gain.

The first coherent C1+C3 sentinel factorial is mixed. Roman C1 is neutral alone
but improves C3 from `0.551981x` to `0.404427x` block SSE. BAL 1490 C1 alone is
`0.892284x`, while C1+C3 remains poor at `1.618721x`; its interaction ratio
`0.991261` is nearly multiplicative and does not repair fixed C3. Retain C1+C3
integration, but do not broaden it as a cross-family candidate before C5 is
tested. The maintained source suite now has 144 passing tests.

The C5+C3 I5 sentinel is also mixed but does not repair BAL. Standalone adaptive
depth is inactive and trajectory-identical on Roman/BAL1490. With Roman C3 it
raises nine clusters to depth 2 on I4--I5 and improves C3 from `0.551981x` to
`0.541142x`. BAL 1490 never raises depth and remains exactly `1.830125x` raw.
Retain coherent execution, move C5 evaluation to longer horizons, and broaden
isolated C1 next. The worker builds and 145 maintained source tests pass.

The matched K24/I30 C5 breadth gate is now complete. Isolated C5 reaches
all-15 1DSfM SSE `0.952889x`, W/T/L `13/0/2`, with no extra proximal oracle
calls and `1.468372x` true worker CPU. C1+C5 reaches `0.786739x` plain SSE and
wins all 15 scenes; relative to C1 it is `0.924929x`, W/T/L `10/0/5`, at
`1.473737x` additional worker CPU. C5 activates on every 1DSfM scene. On large
BAL it is inactive on 1490/1778 and slightly harmful on 3068 with C1:
C1+C5/C1 is `1.004691x`. Promote C5 only as an isolated/cumulative 1DSfM
quality contribution, not a cross-family default, and do not retune its
thresholds.

The final publishable preset is not automatically all features enabled. It is
the conservative cumulative subset that improves a reproducible BAL and
1DSfM Pareto frontier. Every claimed component requires both an isolated
ablation and a cumulative ablation; unsuccessful components remain runnable
for reproducibility but disabled in the final preset. Repeated final Schur
corrections remain a separately labeled polishing budget.

## 3. Never Mix These Scoreboards

Every result must be labeled with exactly one budget class.

### A. DRS-Only Research

K24 shared-only DRS with no final Schur correction tail. This scoreboard tests
whether the consensus and local resolvent are improving.

Current metric-only research preset:

```text
BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS=1
CAMERA_UPDATE=se3_left
SHARED_ONLY_CAMERA_PROXIMAL=1
GLOBAL_SCHUR_MAJORIZER_OBSERVABILITY_THRESHOLD=0.55
GLOBAL_SCHUR_MAJORIZER_UNTIL=10
SCHUR_MAJORIZER_SCALE=0.5
SCHUR_MODEL_CONSENSUS_CLIPPING=0
```

The first displayed iteration uses the raw metric and makes one
coordinator-global observability decision. If selected, all clusters use the
half-strength Schur block majorizer on displayed iterations 2 through 10, then
all clusters return to the raw metric.

Verified all-15 result:

- `0.837765x` fresh raw DRS SSE;
- W/T/L `13/1/1`;
- `2.236813x` Ceres;
- this is useful DRS progress, not a competitive final endpoint.

### B. Polished 1DSfM Quality Reference

This is the weekend quality workflow, not DRS-only:

```text
K24 shared-only DRS I5
+ 50 forced-budget BSR Schur corrections
+ up to 10 Python confirmation corrections
```

Authoritative artifact:

`benchmark_results/1dsfm_k24_i5_practical_all15/plain_drs_block_full_se3_left_lip0.00625_curvature_persistent_tr_schur_d3_s1.0_n50_confirm10.jsonl`

Verified result:

- `1.063839x` Ceres geometric-mean SSE;
- 3/15 Ceres wins;
- `3020.4` summed seconds.

This result is intact and remains the 1DSfM quality reference until an
apples-to-apples end-to-end workflow beats it.

### C. BAL Quality Reference

The established K24/I90 BAL workflow remains the default BAL control:

- `0.998404x` Ceres over all 29 scenes;
- approximately `1349` summed seconds.

Do not replace this with I5 plus repeated Schur polishing. That transfer was
`1.015350x` Ceres and much slower.

### D. Ceres

Ceres is the common endpoint reference. It is not our implementation target and
must not be substituted into the distributed method.

## 4. Current Usable DRS Result

The usable consensus contribution is the transient Schur-majorized DRS metric.
It is a real DRS change: the same worker metric is used in the local proximal
solve and full metric consensus projection.

Construction:

1. Build the local landmark-marginalized camera Schur blocks.
2. For each dropped off-diagonal camera block `H_ij`, add
   `scale * ||H_ij||_F I` to both incident diagonal blocks.
3. Use one coordinator-global startup observability decision so all workers use
   the same metric mode.
4. Use it transiently, then restore raw DRS globally.

Validated behavior:

- six development scenes at I90: `0.810537x` raw, 6/6 wins, all complete;
- held-out nine: `0.856423x` raw, 7 wins, one tie, one loss;
- all-15: `0.837765x` raw, W/T/L `13/1/1`;
- BAL 52, 245, 1490, 1778, and 3068 select raw and reproduce matched SSE
  trajectories bitwise.

Status: retain as an opt-in DRS research preset. It is not yet the quality
workflow and not yet a default.

### Component Registry and Re-Test Policy

Mixed ideas are archived as frozen components, not repeatedly tuned and not
discarded because one scene loses. Re-test a component only after an orthogonal
change modifies the interaction surface. Use a frozen 2x2 comparison whenever
practical: baseline, component alone, new mechanism alone, and their
composition. This reveals whether per-scene effects cancel, reinforce, or move.

Current retained components:

| Component | Frozen setting | Quality effect | Time effect | Status |
|---|---|---:|---:|---|
| Safeguarded C1 | `themelis_nesterov`, grid `0,1`, fallback, restart 3, relative safeguards | all-15 1DSfM `0.873033x`, W/T/L `7/8/0`; large BAL3 `0.777231x`, 3/3 wins | all-15 `1.019837x` optimization, `1.138767x` oracles; BAL3 `1.137850x` overall | Promoted cross-family cumulative base |
| Transient Frobenius majorizer | threshold `0.55`, scale `0.5`, I2--I10 | all-15 DRS-only `0.837765x` raw, W/T/L `13/1/1` | `1.039617x` optimization | Strong DRS component |
| I15 raw equilibration | majorizer I2--I10, raw I11--I15, then 50+10 | Roman/Trafalgar `0.983919x` raw | `1.041510x` overall | Mixed component |
| Startup clipping + majorizer | I1 clipping, majorizer I2--I10 | all-15 DRS-only `0.824880x` raw, W/T/L `13/1/1` | `1.085588x` optimization | Mixed component; Madrid loses |
| Factorized cross-camera Schur metric | startup threshold `0.55`, stabilization `1`, 32 buckets, fixed factors, Schur-PCG | all-15 I5 `0.745081x` matched block, W/T/L `11/2/2`; BAL 52/245 and 3068 bitwise raw; BAL 1490/1778 regress to `1.830125x/1.759679x` | all-15 `2.506342x` time, `4.022384x` bytes; selected large BAL is much costlier | Strong 1DSfM component, failed standalone large-BAL transfer; frozen C1/C5 composition only |
| Initial distributed Schur bootstrap | one safeguarded d3 Schur correction before DRS | all-15 I5 `0.724048x` raw, 15/15 wins | `1.840345x` overall | Strong cross-family component |
| Bootstrap + transient majorizer | bootstrap, then majorizer I2--I5 | all-15 I5 `0.640499x` raw, W/T/L `14/0/1` | `1.997332x` overall | Strong portfolio candidate; Piazza +3.95% |
| Bootstrap + BSR40 | bootstrap I5, 40 accepted BSR corrections, no Python confirmation | `1.054381x` weekend quality; `1.121692x` Ceres | `1238.8s`, `0.410x` weekend | Fast diagnostic preset |
| Bootstrap + BSR50 | bootstrap I5, 50 accepted BSR corrections, no Python confirmation | `1.021438x` weekend quality; `1.086646x` Ceres | `1556.2s`, `0.515x` weekend | Preferred speed-quality preset |

Do not retune these settings in isolation. Preserve their per-scene ratio
vectors and retry them unchanged when a genuinely orthogonal change is ready.

An overall improvement may be acceptable even with bounded scene regressions.
Judge speed and quality jointly:

1. report geometric-mean SSE and runtime ratios;
2. report W/T/L, worst regression, completion, and safeguards;
3. identify whether the candidate is Pareto-improving or offers a reasonable
   explicit quality-for-time tradeoff;
4. compare the composed effect with the product of individual effects to detect
   synergy or cancellation;
5. never hide catastrophic tails, recovery exhaustion, or invalid solves inside
   an aggregate.

## 5. Separate Experimental Features

### Five-Idea Core Integration Queue

These mechanisms extend the same `client_drs.py` implementation. Pure startup
selection policy belongs in `serverTest/drs_startup.py`; worker RPCs and all DRS
state mutation remain owned by the core coordinator.

1. **Worker trust rebase after bootstrap:** implemented opt-in. The corrected
   dispatch audit showed the initialization RPC does perform a local nonlinear
   solve. However, matched BAL-49 K2/I1 and Roman K24/I5 tests found every
   pre-rebase radius already equal to the DABA initial cap `100`. Roman final
   SSE ratio rebase/control is `1.000000000151`, with identical reject and
   linear-iteration sequences. Retain as explicit consistency infrastructure;
   reject stale radius as the current basin-loss explanation.
2. **Short two-branch basin race:** first bounded implementation is opt-in via
   `--initial-shared-schur-basin-race-iterations`. It permits finite ordinary
   safeguard crossings, never bypasses the catastrophic ceiling, tracks
   independent SSE, and restores geometry, product-space state, landmarks, and
   captured worker radii. Roman race-3 selected bootstrap (`71.055M`) over its
   exploratory endpoint (`2029.976M`) and ended I5 at `69.683M`; the ordinary
   bootstrap path ended at `26.572M`. Retain the machinery, not this policy.
3. **Full cross-camera Schur metric:** not implemented. Next large research
   mechanism after startup policy is stable; requires coupled local proximal
   operators and coupled consensus transport, not diagonal majorization.
4. **Collective trust trial/commit:** implemented default-off without a new
   protocol. The coordinator compares nominal local solves with one globally
   safeguarded shared-radius alternative and carries only the selected trial's
   per-worker radii forward. On Roman/Trafalgar K24/I5, no shared alternative
   wins; SSE is unchanged to about `1e-9` while time rises `1.074614x`. Reject
   as a quality mechanism and retain only as trial-state infrastructure. See
   `benchmark_results/collective_trust_trial_gate/report.md`.
5. **Early nonlinear distributed bootstrap:** not implemented. It should reuse
   the race/commit infrastructure and remain separately budgeted from final
   polishing; do not start it until the short-race failure is understood.

### Ordered SSE Research Queue (2026-08-13)

Test these mechanisms one at a time with one global policy and freeze each
result before advancing. The numbering preserves the original proposal list;
testing began with item 2 at the user's request.

1. **Refreshed cross-camera C3:** tested and rejected. The historical
   matrix-free `A - B V^-1 B^T` path was restored with threshold `0.55`,
   stabilization `1`, and 32 buckets, then refreshed on each solve. A fixed
   BAL1490 I1 sentinel reproduces historical SSE to `5.18e-8` relative.
   Refreshed Roman I3 is `1.013905x` fixed C3; refreshed BAL1490 I3 is
   `0.997970x` fixed but still `1.526003x` raw. Stale factors do not explain
   the post-I1 BAL crossing. Keep C3 default-off and do not retune. See
   `benchmark_results/refreshed_c3_gate/report.md`.
2. **C2 block-coordinate Ruiz equilibration:** diagonal symmetric Ruiz from
   full initial `9x9` camera Hessian blocks is now tested and rejected: Roman,
   Trafalgar, and BAL1778 reproduce Jacobi SSE trajectories to about `1e-9`
   while adding runtime. Full matrix-valued `9x9` inverse-square-root transforms
   from raw initial pixel Hessians are also tested and rejected: every Roman,
   Trafalgar, and BAL1778 I3
   candidate is rejected, with final/control ratios `3.864440x`, `3.018749x`,
   and `92.449861x`. A faithful old-style transform derived after the first
   local solve from aggregated worker proximal metrics improves Roman to
   `0.973334x` Jacobi, but Trafalgar and BAL1778 are `1.395777x` and
   `87.637697x`. The requested worker-derived diagonal `diag(U_all)^{-1/2}`
   variant reproduces current Jacobi trajectories on Roman/Trafalgar to about
   `1e-9` and is `1.000083x` on BAL1778 with identical linear work. Historical
   source audit confirms `block_jacobi_gmean` used the dense map and unit
   diagonal scaling; diagonal Jacobi was separate. Close C2 for current direct
   left-SE3 DRS without scalar tuning. A restricted translation-z/focal `2x2`
   block is supported by high worker-metric correlation and improves
   Roman/Trafalgar I3 by `0.299%/0.130%`, but BAL1778 rejects all candidates
   and is `92.442181x` diagonal I3. Correlation cannot select it because BAL1778
   has the strongest coupling. Preserve the old additive-coordinate
   result as distinct evidence. See
   `benchmark_results/c2_diagonal_ruiz_gate/report.md`,
   `benchmark_results/c2_full_block_gate/report.md`, and
   `benchmark_results/c2_worker_metric_block_gate/report.md`, and
   `benchmark_results/c2_worker_diagonal_gate/report.md`, and
   `benchmark_results/c2_z_focal_block_gate/report.md`.
3. **Model-ratio nonlinear startup LM:** tested and rejected as a quality
   addition. With cap three, gain-ratio damping and bounded retries improve
   K24/I1 versus geometric damping on Roman/Trafalgar/BAL1490 by
   `0.830477x/0.865090x/0.467159x` (geomean `0.694946x`). BAL1490 recovers its
   nonconverged third correction by retrying damping `0.333333` at `0.666667`.
   The advantage does not persist: BAL1490 I30 is `1.007092x` at `1.366498x`
   time; Roman/Trafalgar I60+16 are `1.043756x/0.935121x`, geomean `0.987946x`,
   at `1.393951x` time. Keep the isolated startup policy default-off. See
   `benchmark_results/startup_model_ratio_gate/report.md`.
   Two follow-ups are also closed. A conservative `1/2` damping floor exactly
   reproduces geometric Roman/Trafalgar startup and is `1.680291x` the
   one-third BAL1490 SSE after two failed retries. Fixed-horizon dual-branch
   commit cannot predict the final winner: BAL1490 crosses at I9, Trafalgar
   changes ordering twice, and Roman geometric wins only after correction 12
   of final polishing. Full-workflow branch racing is possible but too costly
   for a production policy.
4. **Transient external pose-factor continuation:** closed under the current
   cross-family data contract. Roman/Trafalgar retain 94,250/861,509 mapped
   epipolar pose edges, and the existing eight-mode weight-`0.1` removable
   prior already implements and gates this proposal on all 15 1DSfM scenes.
   Fixed use is unsafe (`0.989513x` geometric but `1.001376x` summed SSE,
   Trafalgar `1.050620x`), while the first loss-free selector costs about
   `1.900388x`. BAL1490 contains no independent relative-pose measurements or
   sidecar. Deriving factors from its initial cameras would be an initialization
   anchor, not external evidence; reconstructing them from pixels would reuse
   the evaluated observations. Do not implement a family-dependent policy.
   Reopen only with a common persisted pose-graph contract. See
   `benchmark_results/external_pose_factor_audit/report.md`.

The four-item queue is complete. Numerical equilibration, refreshed C3, and
model-ratio startup fail persistence or cross-family safety; external pose
continuation lacks a common independent-information contract and its existing
1DSfM implementation is not safe as a fixed policy.

### Cluster-Count Continuation (2026-08-13)

K4/I3 to K24 continuation is now supported through explicit canonical-frame
state transfer. Existing raw-frame `--initial-state` behavior remains the
default; staged states must declare `--initial-state-frame canonical`. The
K4 endpoint and K24 initial SSE match exactly in every gate scene. Dynamic DRS,
trust, scaling, acceleration, and partition state intentionally restart.

The maintained Nesterov local solver overflows a Roman K4 tangent step before
trust rejection, under both DRS and DABA trust. A globally matched Schur-PCG,
block-Jacobi, DABA policy is finite and accepted. With the frozen K4/I3 split,
equal-budget K24/I30 ratios are Roman `0.744715`, Trafalgar `0.945498`, and
BAL1490 `0.997938`; geometric SSE is `0.889034x` at `1.345145x` time. This is a
strong fast-horizon frontier.

The same schedule at I60+16 is not quality-safe. Pre-Schur ratios are
`0.725244/1.009605/1.002656`; final Roman/Trafalgar/BAL1490 ratios are
`0.953985/1.038197/1.002877`, geometric `0.997753x`, at `1.012516x` time.
Retain continuation default-off for I30 diagnostics; do not promote it to the
named quality workflow or tune K/split on these scenes. See
`benchmark_results/cluster_continuation_gate/report.md`.

Frozen breadth transfer confirms a strong but unsafe I30 frontier. On all 15
1DSfM scenes, K4/I3 to K24/I27 reaches geometric/summed SSE
`0.943771x/0.950010x`, W/T/L `11/0/4`, at `1.297602x` geometric time; Montreal
is the worst loss at `1.058938x`. On BAL1490/1778/3068 the ratios are
`0.997938/0.998288/1.011140`, geometric `1.002437x`, so the large-BAL aggregate
is slightly negative. Combined 18-scene SSE is `0.953305x` geometric and
`0.955586x` summed, W/T/L `13/0/5`, at `1.313318x` time. Every canonical
handoff is exact and every arm completes its budget. Retain the resumable
breadth runner and fixed schedule default-off; do not promote or tune from
these outcomes. See
`benchmark_results/cluster_continuation_breadth_i30/report.md`.

Time-matched and BAL29 tails resolve the continuation interpretation. On the
18-scene breadth cohort, staged I30 versus direct K24/I40 is `0.997143x`
geometric but `1.003723x` summed SSE at `1.025159x` time; all three large BAL
scenes lose. On the complete BAL29 corpus, staged/direct ratios evolve from
`0.999377x` at equal I30 (`1.334163x` time), to `1.003477x` versus direct I40
(`1.070811x` time), to `0.998902x` at equal I90, and finally `0.999876x` after
equal I200+16 (`1.070250x` time). The separation contracts toward one and both
arms usually still improve at their final outer iteration. This is finite-time
path dependence, not evidence of a materially better converged minimum. Retain
continuation only as a default-off basin diagnostic; do not promote it for BAL
or globally. See `benchmark_results/cluster_continuation_bal29_i30_i40/report.md`.

### Mid-Run Global Schur Rebase (2026-08-14)

A default-off I10 global Schur rebase now materializes the worker's accepted
landmarks, applies one converged strict physical-SSE-decreasing correction,
resets product-space centers, and rebases worker trust without changing the
accepted geometry. Rejected corrections restore the original accepted state.

The four-scene sentinel is strong (`0.939472x` final I30 SSE at `1.164859x`
time), but breadth and time controls reject promotion. On all 15 1DSfM scenes,
candidate/direct-I30 is `0.977835x` geometric and `0.948698x` summed, W/T/L
`12/0/3`, at `1.822800x` time. Against the nearer direct I40 control it is
`1.030822x` geometric and `1.001725x` summed, W/T/L `3/0/12`, while still
costing `1.419161x` I40 time. BAL29 is harmful already versus I30:
`1.017020x` geometric, `1.013890x` summed, W/T/L `2/2/25`, at `2.040605x`
time; versus direct I60 it is `1.025366x`, W/T/L `1/0/28`.

The correction itself is effective, but subsequent DRS commonly stalls or
rejects after the product-space reset, and direct K24 catches up more
efficiently. Retain only as a diagnostic for global-step opportunity; do not
tune trigger iteration/damping or promote. See
`benchmark_results/mid_schur_rebase_i10_breadth/report.md`.

### Fixed-K Repartition Continuation (2026-08-14)

K24/I15 default-partition geometry was resumed for K24/I15 with the balanced
`landmark_scalable_stable` partition. DABA Louvain was excluded because its
sentinel residual max/mean load is `2.76--2.97x`; stable remains within the
required ±1% balance while changing 5.0% of Roman and 19.2% of BAL3068
landmark ownership after optimal relabeling.

The alternate partition has real 1DSfM path signal versus a same-partition
restart (`0.968615x` geometric, `0.983186x` summed, W/T/L `10/0/5`), but restart
itself is `1.036495x` uninterrupted I30. The complete stable/direct result is
`1.003965x` geometric and `1.016969x` summed, W/T/L `5/0/10`, at `2.019627x`
time. BAL partition effect is negligible (`0.999554x` stable/restart); complete
stable/direct is `0.999457x` at `1.954289x` time. Against direct I60, staged
1DSfM/BAL SSE is `1.117241x/1.007659x`, with W/T/L `1/0/14` and `2/0/27`.

Reject repartition continuation. Retain only as a path-diversity diagnostic;
do not tune checkpoint time or route partitions by scene. See
`benchmark_results/repartition_continuation_i15_i15_breadth/report.md`.

### Phased Local Solver Portfolio (2026-08-14)

The all-15 C4 trajectories show Schur-PCG ahead at I10 but Nesterov ahead
geometrically at I30, motivating one homogeneous PCG-I1--I10 then
Nesterov-I11--I30 path. A request-scoped global solver switch was added
default-off without resetting trust or accepted state.

The four-scene sentinel rejects the schedule. Phased/pure-Nesterov SSE is
Roman `1.258234`, Trafalgar `1.438000`, NYC Library `1.119719`, and Piccadilly
`1.060541`; geometric `1.210707x`. Phased/pure-PCG is geometric `1.066060x`,
with only Piccadilly winning. Nesterov cannot undo the inherited PCG
product-space trajectory, and Trafalgar stalls across the switch.

Do not expand or tune the switch. Dual per-iteration racing would require
complete worker trust/accepted-state snapshots, homogeneous global branch
selection, and roughly doubled local work; the cheaper transfer gate gives no
justification for that complexity. See
`benchmark_results/phased_local_solver_pcg10_nesterov_i30_gate/report.md`.

### Bounded Parameter Sensitivity (2026-08-14)

The promoted K24/I30 C1+C5 stack was tested one factor at a time around its
current control: block regularization `5e-5`, acceleration restart-after `3`,
maximum acceleration-step ratio `10`, quartic relative-safeguard annealing,
and DRE allowance `0.01`. The previously hardcoded annealing exponent and
reference iteration are now request-scoped, default-preserving controls;
acceleration step-cap hits are reported without changing proposals.

The four-scene Roman/Trafalgar/BAL1490/BAL3068 safety gate tested block
regularization `2.5e-5/1e-4`, restart-after `1/5`, cap `3`, annealing exponents
`2/8`, and DRE allowances `0.005/0.02`. Only `1e-4` passed both families:
`0.937564x` control on the 1DSfM pair and `0.991877x` on BAL, 4/4 wins.
Restart `5`, exponent changes, and the lower DRE allowance were mostly
trajectory-neutral; restart `1`, cap `3`, and `2.5e-5` were cross-family
negative.

Frozen `1e-4` then failed the established six-1DSfM plus five-BAL development
gate. It is `0.988214x` control on 1DSfM but `1.005534x` on BAL, W/T/L `2/0/3`
there, and costs `1.042784x` BAL optimization time. Retain `5e-5`; do not
interpolate or expand to held-out/all-29. The tested safeguard constants are
now sufficiently insensitive around the promoted stack to close scalar
wiggling. See `benchmark_results/drs_parameter_sensitivity_k24_i30_gate/report.md`
and `benchmark_results/drs_block_regularization_1em4_development_k24_i30/report.md`.

### Dual-Preserving Mid-Schur Transport (2026-08-14)

The accepted I10 global Schur tangent can now be left-composed onto every
active resident local camera copy while preserving its exact scaled
camera-center offset and worker trust radius. The worker installs transported
centers and updates rollback/accepted landmark state; the existing
consensus-collapse/trust-rebase behavior remains the default. The request-
scoped transport path is tested and reports offset/trust invariants.

The four-scene three-arm gate compares no correction, the existing reset, and
transport. Transport partially repairs immediate continuation: I10--I14
accepts/rejects improve reset `3/7 -> 6/4` on Roman/Trafalgar and `6/4 -> 9/1`
on BAL1490/3068. Trafalgar accepts all five post-correction iterations and ends
`0.942954x` reset. However, Roman still accepts only one of five and ends
`1.103501x` reset; BAL3068 is `1.005310x` reset. Aggregate transport/reset is
`1.020074x` on 1DSfM and `1.002651x` on BAL. BAL1490 rejects the correction
before either commit policy and ties exactly.

Reject transport as a quality mechanism without breadth or trigger tuning.
Product-space collapse explains part of the stall but is not the common basin
cause. Retain the mode default-off as state-transition diagnostic
infrastructure. See `benchmark_results/mid_schur_transport_i10_gate/report.md`.

### Direct-Tangent Camera Parameterizations (2026-08-14)

The `microlie.pdf` convention audit confirms production `se3_left` is Sola's
left/global perturbation for `p_c = R p_w + t`, with tangent `[rho, theta]`,
`R+ = Exp(theta) R`, and `t+ = Exp(theta)t + J_l(theta)rho`. Its direct map
`dr/dtheta = J_l(r)^-1`, `dt/drho = I`, `dt/dtheta = -[t]x` is correct.
`se3_right` also matches the paper's right/local perturbation.

A translation-first `so3_left` mode now implements the product manifold
`SO(3) x R3` with independent translation while preserving the common tangent
order `[translation, rotation, intrinsics]`. One shared helper supplies the
mode-correct direct tangent map to residual assembly and proximal metrics.
State and point-action Jacobians pass central finite differences for left-SE3,
right-SE3, and product-SO3. The historical manifold factorial used the
non-direct path and is not the current comparison.

Frozen K24/I30 C1+C5 evaluation with global Schur-PCG tolerance `1e-2` and cap
1000 gives product-SO3/left-SE3 `0.996338x` geometric and `0.981871x` summed SSE
over all 15 1DSfM, but W/T/L `4/0/11` and a `1.246462x` Piazza tail. On all 29
BAL it gives `0.996646x` geometric, `0.995410x` summed, W/T/L `23/0/6`, worst
`1.001555x`, and `0.987873x` optimization time. The larger cap is necessary:
Trafalgar needs up to 690 PCG iterations; cap 400 violates the residual gate.

True I90 sentinels reject global promotion: Roman is `0.999458x` left-SE3 but
Trafalgar is `1.060555x`, so their ratio is `1.029553x` and the I30 Trafalgar
gain reverses. BAL1490 remains `0.998090x`; BAL3068 is stability-only because
left-SE3 exhausts at I54 while product-SO3 completes. Right-SE3 is rejected by
its `1.398362x` Roman/Trafalgar I30 ratio despite favorable BAL sentinels.

Keep left-SE3 as the common default. Retain product-SO3 default-off as a strong,
bounded BAL-oriented ablation and frozen component, not a scene-selected or
cross-family policy. See
`benchmark_results/camera_parameterization_direct_tangent_report.md`.

Product-SO3 tuning follow-up (2026-08-14): a frozen C1 x C5 factorial identifies
C1 as robust and C5 as interaction-sensitive. C1/plain is `0.772264x` on six
1DSfM and `0.955724x` on BAL5, 11/11 wins. C1+C5/C1 is `0.988354x` 1DSfM but
loses 4/6; BAL is neutral. Delaying C5 to start 15 improves development tails
but fails held-out transfer (`1.008025x` left-SE3 all-15), so close C5 timing.

Product camera metric scales `10/25/35/40/45/50/75` show scale 35 as the
development leader, but confirmation remains unsafe: `0.994506x` left-SE3
all-15 with Piazza `1.316080x`, Tower `1.193405x`, and `2.031953x` time;
BAL29 is `0.997011x` left-SE3 but `1.000366x` scale 25. A coherent
translation/rotation tangent metric ratio `{0.5,1,2}` also rejects both
non-unit values on both families; ratio 1 is bitwise neutral and best.

Retain product-SO3 scale 25/start5/ratio1 as the frozen diagnostic component.
Do not continue C5 timing, scalar metric, or subspace-ratio grids. The next
parameter direction is a bounded product-aware trust envelope; the next
structural direction is camera-center `SO(3) x R3`. Full details remain in
`benchmark_results/camera_parameterization_direct_tangent_report.md`.

The product-aware trust envelope is complete and rejected for common promotion.
Trust telemetry confirms the default `1e6` maximum is active. Maximum-radius
grid `{1e3,1e4,1e5,1e6}` selects `1e5` on development (`0.976718x` product
control on six 1DSfM, `0.997610x` on BAL5, 10/11 wins). Frozen confirmation
reverses on 1DSfM: all-15 is `1.010903x` product control and `1.007201x`
left-SE3, with Piazza `1.238627x`; BAL29 stays favorable at `0.999542x`
product control. Do not interpolate or tune the DABA initial cap from held-out
results. Scalar SO3 tuning is closed. Next implement the structurally distinct
camera-center product manifold, keeping physical BAL storage and evaluation
unchanged.

Camera-center product result (2026-08-14): `so3_center_left` keeps physical
`[R,t]` storage while defining `C=-R^T t`, with `R+=Exp(theta)R`,
`C+=C+delta_C`, and `t+=-R+C+`. Direct state and point-action Jacobians pass
finite differences. The four-scene sentinel is strong (`0.964637x` left-SE3 on
Roman/Trafalgar, `0.953493x` on BAL1490/3068), but frozen development rejects
transfer: six 1DSfM `1.011018x`, W/T/L `2/0/4`, worst Gendarmenmarkt
`1.103175x`; BAL5 `1.005759x`, `1/0/4`, worst BAL245 `1.025866x`. Do not tune
or expand. Retain the coherent camera-center mode default-off as an ablation;
left-SE3 remains the common default.

Final product-SO3 C1-only breadth (2026-08-14): disabling C5 does not repair
1DSfM generalization. Frozen SO3+C1/left-SE3-C1+C5 is `1.067747x` on the nine
held-out scenes and `1.039726x` over all 15, with summed `1.027391x`, W/T/L
`7/0/8`, and Piazza `1.496952x`. It is also `1.043547x` the SO3+C1+C5 all-15
endpoint, despite `0.851369x` optimization time. All 29 BAL remain strong and
bounded at `0.996776x` left-SE3, 23/29 wins, worst `1.001555x`; C5 is nearly
inactive there (`1.000131x` C1/C1+C5).

Close product-SO3 for common-policy promotion. Keep left-SE3 as the global
default and retain product-SO3 default-off only for reproducibility and a
BAL-oriented publication ablation. Do not continue product-SO3 C5, metric,
trust, or coordinate sweeps. See
`benchmark_results/so3_c1_confirmation_k24_i30/report.md`.

Trust-rebase artifacts:

- `benchmark_results/bootstrap_trust_rebase_bal49_off/`
- `benchmark_results/bootstrap_trust_rebase_bal49_on/`
- `benchmark_results/bootstrap_trust_rebase_roman_i5_off/`
- `benchmark_results/bootstrap_trust_rebase_roman_i5_on/`

Initial basin-race artifact:

- `benchmark_results/bootstrap_basin_race_roman_i5_r3/`

Startup Schur-model consensus clipping is a separate globalization mechanism.
It is not part of the default research preset.

Composing clipping with transient majorization gives:

- `0.824880x` raw DRS over all 15;
- W/T/L `13/1/1`;
- `2.202410x` Ceres;
- all 15 complete;
- Madrid regresses `8.18%` versus raw.

Keep clipping separate from the default preset. Archive the frozen composition
for re-test with later orthogonal mechanisms; do not promote it standalone from
its current aggregate gain.

Final and repeated Schur corrections are polishing. Keep their results and
code, but never count their gains as DRS consensus gains.

## 6. Completed Equal-Budget I10 Gate

The first apples-to-apples quality gate used fresh raw and transient-majorizer
I10 handoffs followed by identical BSR50 + Python10 correction tails, with
startup clipping disabled.

Roman candidate/control is `1.034679x`; Trafalgar is `0.990213x`; the two-scene
geometric mean is `1.012202x`. Both arms accept 60 corrections and every
accepted PCG solve converges. The candidate starts much better and remains
better through 30 accepted corrections on both scenes, but Roman crosses near
correction 40 and finishes worse. This does not qualify as a standalone I10+60
improvement because the aggregate regresses, so the six-scene I10 gate was not
launched. It does not invalidate the transient metric at other horizons or in
composition.

## 7. Completed Equal-Budget I5 Gate

The exact weekend-horizon gate used fresh raw and transient-majorizer I5
handoffs followed by identical BSR50 + Python10 tails, with startup clipping
disabled.

The six-scene candidate/control geometric mean is `0.964289x`, but W/L is only
`3/3`. Gendarmenmarkt regresses `9.52%`, Union `3.49%`, and Vienna `1.45%`.
Candidate/Ceres is `1.047056x` with 2/6 wins. Candidate/preserved-weekend is
`1.004871x`, so it does not beat the existing quality reference. All accepted
PCG solves converge. This does not qualify for standalone all-15 promotion, but
its `3.57%` aggregate gain with bounded regressions is retained as a portfolio
component rather than declared invalid.

The DRS handoff gain is nevertheless real: before polishing the six-scene
candidate/control ratio is `0.741372x`, and after one accepted correction it is
`0.614929x`. The advantage decays under the fixed tail to `0.886639x` after 10,
`0.947095x` after 20, and `0.967378x` after 50 corrections. Gendarmenmarkt,
Vienna, and Union cross from better to worse near accepted corrections 12, 11,
and 25.

## 8. Archived Historical Task (Superseded)

This section records the earlier transient-majorizer/polishing investigation.
It is not the active restart point. The controlling direction is
**Section 1A, Active K1-to-Core Decision**. Do not resume work from this section
or from its final "stop here" language.

Do not launch another selector, startup-duration, or correction-count sweep.

Use the completed equal-budget artifacts to diagnose why repeated global Schur
polishing erases the improved DRS handoff basin. The next analysis must compare
raw and majorized states at accepted corrections 0, 1, 10, 20, and the first
crossing on Gendarmenmarkt, Vienna, and Union. It should inspect:

1. actual/model gain ratios and damping at matched accepted-correction counts;
2. BSR-to-Python phase behavior;
3. gradient norms, PCG iterations, and relative residuals;
4. whether the two states approach the same basin or diverge into different
   stationary neighborhoods.

No new run is justified until this analysis names one falsifiable mechanism.
The transient majorizer remains useful DRS-only work and a possible component
of a later combined policy, but it does not replace the end-to-end quality
workflow by itself.

Completed artifact analysis rules out correction implementation failure. The
three losing scenes cross during the BSR screening phase before Python
confirmation, while both branches keep accepting converged, positive-gain
steps. Similarity-aligned final-state comparisons show worse camera geometry for
the Frobenius-majorized branch on all three crossing scenes. Camera-center P95
over scene scale is `1.331` versus raw `0.264` on Gendarmenmarkt, `4.415` versus
`3.821` on Vienna, and `0.149` versus `0.087` on Union. Gendarmenmarkt rotation
P95 is `22.5` versus `13.1` degrees; Union low-32 camera-mode energy is `0.292`
versus `0.089`.

The tested mechanism was over-isotropic coupling regularization. The current
majorizer adds `||H_ij||_F I` to both incident 9x9 camera blocks, discarding the
directional structure of each dropped Schur coupling. The candidate tight
directional block bound was

```text
camera i: (H_ij H_ij^T)^(1/2)
camera j: (H_ij^T H_ij)^(1/2)
```

computed from the 9x9 SVD. It was implemented and rejected at the DRS-only I5
gate, then removed from production code. It is `1.154393x` raw geometrically
over Gendarmenmarkt, Vienna, Union, Roman, and Trafalgar and loses all five
comparisons to the Frobenius majorizer (`1.748798x`). Roman regresses `21.47%`,
Union `62.97%`, and Trafalgar `3.56%` versus raw. No equal-budget polishing gate
was run. Directional local coupling shape alone does not repair the basin
interaction.

No next implementation mechanism is currently validated. Stop here rather than
resume scalar, selector, duration, or correction-tail tuning. The next session
must formulate a new mechanism from the accepted-step and geometric evidence
before editing solver behavior.

The proposed per-cluster observability-variance consensus weighting is also
rejected before implementation. Crossing and non-crossing scenes have
essentially identical mean observability variance (`0.00167767` versus
`0.00167654`), and Vienna has the lowest variance despite crossing earliest.
Weighting only the consensus RHS would additionally cease to be a metric
projection, so that formulation is mathematically inadmissible.

## 9. Completed Metric-Transition Gate

A distinct mechanism is now supported by existing trajectories: the failed I5
and I10 equal-budget candidates hand off to Schur polishing while the majorizer
is still active. The successful I90 policy first restores raw DRS. Existing
matched trajectories show candidate/raw ratios of `0.832060x` at active I10,
`0.891971x` on the first raw iteration I11, `0.818951x` after five raw
equilibration iterations at I15, and `0.718095x` at I20. I15 wins all six scenes.

The tested hypothesis was that repeated polishing erases the gain because it receives a state from
the active majorized resolvent before the raw DRS fixed-point geometry has
re-equilibrated.

The frozen sentinel gate was:

```text
Control:   fresh raw shared-only DRS I15 + BSR50 + Python10
Candidate: majorizer I2-I10, raw DRS I11-I15 + BSR50 + Python10
```

Roman candidate/control is `1.031458x`; Trafalgar is `0.938572x`; the two-scene
geometric mean is `0.983919x`. Every accepted PCG solve converges. Roman's
`3.15%` regression prevents standalone promotion under the old sentinel rule,
but the `1.61%` aggregate gain means the result is retained as mixed evidence,
not rejected as an invalid mechanism. Five raw DRS equilibration iterations do
not uniformly repair the polished-basin interaction; an orthogonal change may
still offset Roman while preserving Trafalgar's gain.

No further scalar tuning run is authorized. The retained portfolio components
are the DRS-only transient Frobenius majorizer and the mixed I15 transition
result. A new end-to-end candidate requires a genuinely new, orthogonal basin
mechanism, not more raw equilibration, duration, scale, selector, or
correction-tail tuning.

## 10. Initial Distributed Schur Bootstrap

A genuinely orthogonal mechanism is now implemented: one exact distributed
global Schur correction before DRS. It uses the existing cluster-local landmark
elimination and coordinator PCG solve, accepts only a converged solve with lower
independent pixel SSE, and then resets every DRS camera copy, center, consensus,
accepted snapshot, and residual consistently. Rejected proposals restore the
worker state. The feature is disabled by default.

Frozen I5 2x2 results:

- six development scenes: bootstrap `0.634916x` raw, 6/6 wins; combined with
   transient majorization `0.529600x`, 6/6 wins;
- nine held-out scenes: bootstrap `0.790317x`, 9/9 wins; combined `0.727051x`,
   8 wins and one bounded Piazza regression (`1.039475x`);
- all 15: bootstrap `0.724048x` raw at `1.840345x` time; combined `0.640499x`
   raw at `1.997332x` time;
- BAL 52/245: bootstrap `0.943348x` raw at `1.307400x` time, 2/2 wins;
- every startup correction is accepted with converged PCG and worker/evaluator
   SSE agreement.

The combined all-15 interaction is synergistic overall on the held-out cohort
(`0.948551x` relative to the product of individual effects), while development
interaction is partly redundant. This is a valid portfolio advance despite one
bounded scene loss.

Absolute unpolished I5 endpoints remain far from Ceres (`18.367841x` for
bootstrap and `16.248335x` for combined), so the subsequent frozen gate tested
equal-budget end-to-end quality, not another bootstrap parameter test.

## 11. Completed Bootstrap Quality Gates

The first gate used Roman and Trafalgar in a frozen 2x2 with identical BSR50 + Python10
tails:

```text
baseline | transient majorizer
bootstrap | bootstrap + transient majorizer
```

The baseline and majorizer arms already exist in
`benchmark_results/equal_budget_i5_bsr50_python10_majorizer/`. Run only the
bootstrap and combined arms with startup clipping disabled and unchanged d3,
scale `0.5`, threshold `0.55`, and correction settings. Report aggregate and
per-scene quality, runtime, accepted corrections, PCG convergence, and
interaction. Retain aggregate-positive bounded-loss outcomes as components;
advance to six scenes if the portfolio quality/time result is reasonable and no
hard safety blocker appears.

The completed six-scene equal-budget 2x2 selects bootstrap-only as the current
portfolio candidate:

- bootstrap/raw `0.943613x`, W/T/L `4/0/2`;
- bootstrap/Ceres `1.024605x`;
- bootstrap/preserved-weekend `0.983324x`, W/T/L `3/0/3`;
- runtime `1.047083x` fresh raw;
- all accepted correction PCG solves converge;
- worst raw regression is Gendarmenmarkt `+6.87%`; Vienna is `+4.50%`;
- bootstrap + majorizer is weaker after polishing (`0.965469x` raw), so the
   majorizer remains a separate archived component rather than part of this gate.

This is a reasonable quality/time portfolio advance despite two bounded scene
losses. The subsequent frozen gate used fresh raw versus bootstrap-only I5 + BSR50 +
Python10 on all nine held-out scenes. Do not enable transient majorization or
startup clipping. Compare both with the preserved weekend artifact and Ceres.

The held-out equal-budget result is `0.985914x` fresh raw, W/T/L `4/0/5`, at
`1.020972x` runtime. It is `1.016626x` the preserved weekend endpoints and
`1.096599x` Ceres. Every accepted correction PCG solve converges.

Combining development and held-out cohorts gives the authoritative all-15
bootstrap-only portfolio result:

- bootstrap/fresh raw `0.968770x`, W/T/L `8/0/7`;
- bootstrap/preserved-weekend `1.003172x`, W/T/L `6/0/9`;
- bootstrap/Ceres `1.067214x`, 3/15 Ceres wins;
- runtime `1.031338x` fresh raw;
- no accepted nonconverged PCG solve.

This is retained as an aggregate-positive, low-overhead component. It is not a
standalone replacement for the weekend quality workflow because it is
effectively tied but slightly worse at equal budget. The completed next evidence
gate tested bootstrap transfer on larger BAL 1490, 1778, and 3068 with fresh
matched I5 controls, unchanged damping, and no majorizer composition.

The completed all-15 accepted-correction traces define a Pareto curve without
additional optimization runs. Bootstrap + BSR40 reaches `1.054381x` the weekend
quality reference (`1.121692x` Ceres) in `1238.8s`, or `0.410x` weekend time.
Bootstrap + BSR50 reaches `1.021438x` weekend (`1.086646x` Ceres) in `1556.2s`,
or `0.515x` time, and is `0.971716x` fresh raw at the same accepted-correction
budget. Full bootstrap + 50+10 is `1.003172x` weekend but takes `3118.1s`, so it
is not Pareto-useful relative to the unchanged `3020.4s` weekend workflow.
Retain BSR50 as the preferred speed-quality component and BSR40 as a faster
diagnostic; neither replaces the weekend quality reference.

Large-BAL I5 with standard BSR is `0.937708x` raw geometrically on 1490, 1778,
and 3068, W/T/L `2/0/1`, but costs `2.428407x` time and raises memory sharply.
The mathematically equivalent `bsr_low_memory` operator with a strict 1000-step
PCG cap reproduces standard quality on 1490/1778 and changes 3068 by only
`0.999976x`; it is faster and approximately halves aggregate Schur memory.
On 1778 it requires 507 iterations, so the original 500 cap correctly rejects
it. Separate initial-operator/budget controls now allow low-memory/1000 for the
bootstrap while leaving a final 50+10 tail on BSR/500.

The I30 persistence gate rejects bootstrap before unchanged long BAL DRS:
1490 is `1.003492x` raw, 1778 `1.006037x`, and 3068 `1.161254x`; geomean is
`1.054431x` with 0/3 wins, `1.090418x` time, and `2.163590x` memory. PCG
converges in every case. The correction itself is excellent at I1, but current
DRS trajectories cross by I2; best-checkpoint tracking preserves the bootstrap
state at short horizons. Retain bootstrap as a quality handoff/basin proposal,
not as a prefix to unchanged long BAL DRS.

Worker trust-state rebasing was re-audited against the C++ request dispatch.
The initialization RPC does execute `UpdateStepSizeAndSolve` before the
external Schur correction, so its radius and rollback snapshots can describe
pre-bootstrap geometry. The opt-in
`--initial-shared-schur-rebase-trust-state` commit now resets the radius to the
configured initial DABA/DRS policy and atomically snapshots accepted corrected
cameras and landmarks. Defaults remain unchanged pending a matched test.

## 12. Completed Bootstrap Basin-Guard Gate

Existing rows show that short-run safeguard annealing can accept first-step SSE
increases of `1.94x` on Roman, `2.43x` on Trafalgar, and `7.00x` on Vienna after
an independently accepted bootstrap. A new opt-in basin guard keeps the
bootstrap SSE as a physical ceiling, rejects candidates at or above it through
the existing accepted-state recovery path, and releases permanently after the
first accepted strict improvement. It does not change DRE calculations,
long-run safeguards, or default behavior.

The frozen I5 gate is mixed. Guard-only versus unguarded bootstrap is `0.936138x`
on Roman, `1.000000x` on Trafalgar, `1.102274x` on Vienna, and `1.301368x` on
3068. It finds and releases on strict improvements in Roman, Vienna, and 3068;
Trafalgar remains safely at the bootstrap ceiling. Blocking initially worse
states can nevertheless block later better basins.

The guard/majorizer 2x2 gives `0.918872x` bootstrap over Roman, Trafalgar, and
Vienna with 3/3 wins, but is antagonistic relative to majorizer alone
(`0.741225x` bootstrap). Per scene, guard+majorizer is `0.915956`, `0.847664`,
and `0.999232`; the guard uniquely helps Trafalgar but offsets majorizer gains
on Roman and Vienna. BAL 3068 declines majorization and exactly reproduces the
guard-only path.

Status: retain the guard, disabled by default, as a mixed portfolio component
with its full effect vector. Do not promote it or compose it by default. Re-test
only with a later orthogonal mechanism that specifically needs physical basin
containment; best-checkpoint tracking remains the preferred general protection.

## 13. Mandatory Experiment Discipline

Before launching a run, write down:

1. hypothesis;
2. one changed mechanism;
3. exact control artifact or fresh control command;
4. budget class;
5. cheap falsifying check;
6. mandatory transfer gate.

After every run, report in this order:

1. budget class;
2. candidate/control ratio;
3. candidate/Ceres ratio when available;
4. completion and safeguard state;
5. runtime ratio;
6. per-scene ratio vector and worst regression;
7. decision: reject, retain as component, or promote standalone.

Never use `promote` for a DRS-only result when the claimed reference is a
polished endpoint. Never compare I90 DRS-only with I5+50+10 without displaying
both budgets in the same sentence.

## 14. Promotion Ladder

Results are portfolio decisions, not one-scene vetoes.

- **Reject** only when the mechanism is unsafe, incomplete, worsens aggregate
   quality, has a catastrophic tail, or its causal premise is falsified.
- **Retain as component** when aggregate quality improves and regressions are
   bounded, even if breadth is mixed. Preserve the full per-scene log-ratio
   vector so an orthogonal change can be tested for complementary effects.
- **Promote standalone** only after broad transfer, bounded tail risk, and an
   equal-budget improvement over the relevant quality reference.

Default risk bands for development gates are guidance, not hard statistical
claims: regressions below `5%` are small, `5%`--`10%` require explicit
justification, and above `10%` block standalone promotion unless a broader
frozen gate demonstrates compelling compensation. Any recovery exhaustion,
nonfinite state, accepted nonconverged PCG solve, or safeguard weakening remains
a hard blocker.

A change advances only through these gates:

1. focused tests and worker build;
2. Roman/Trafalgar short sentinel;
3. six development scenes at the requested horizon;
4. nine held-out 1DSfM scenes with frozen settings;
5. representative BAL transfer with matched controls;
6. all-15/all-29 only after prior gates pass;
7. equal-budget end-to-end comparison with the quality references.

One bounded sentinel regression does not invalidate an aggregate improvement.
It blocks standalone promotion only when its magnitude or recurrence creates
unacceptable tail risk. Do not tune on held-out scenes and then report them as a
held-out gate.

## 15. Completed Routes Not To Repeat Unchanged

Do not restart these without a new mechanism and explicit justification:

- fixed scalar inner stopping or per-cluster stopping selectors;
- per-worker observability thresholding;
- permanent Schur majorization;
- plain landmark-marginalized diagonal Schur metric;
- global scalar raw/Schur blends;
- fixed subspace blends;
- directional SVD Schur block majorization;
- per-cluster observability-variance consensus weighting;
- fixed raw-DRS equilibration after transient majorization (mixed component;
   do not tune its length without an orthogonal mechanism);
- one fixed-camera landmark refinement after every local solve;
- weakening PCG convergence or physical-SSE safeguards;
- resuming unchanged DRS after a mid-trajectory global correction;
- more repeated Schur corrections as the primary DRS research direction.

## 16. Restart Checklist

After a crash or context loss:

1. Read this file, especially **Section 1A, Active K1-to-Core Decision**.
2. Treat the lower archived sections and
   `benchmark_results/1dsfm_k1_algorithmic_bridge_report.md` as evidence, not
   as authority for the next experiment.
3. Check `git status --short`; never revert unknown user changes.
4. Confirm the weekend artifact exists.
5. Confirm no benchmark process is still running before reusing ports.
6. Build and test when code may have changed:

```bash
cd serverTest
cmake --build build_admm -j2
PYTHONPATH="$PWD/build_admm/generated/proto:$PWD" \
  .venv/bin/python -m pytest -q test_*.py
```

Expected current maintained source suite: 128 passed.

7. State the active budget class and exact next experiment before launching it.
   No solver experiment is currently authorized. The active task is P0 offline
   manuscript consolidation from existing authoritative artifacts. The frozen
   combined-stack K4/K16 transfer is closed development-negative; do not resume
   breadth expansion or retune it.
8. Update this file whenever a result changes the north star, quality reference,
   current research preset, frozen next experiment, or promotion decision.

## 17. Canonical Files

- Current research report:
  `benchmark_results/1dsfm_k1_algorithmic_bridge_report.md`
- Compact DRS/Ceres scoreboard:
   `benchmark_results/drs_ceres_current_scoreboard.md`
- General historical handoff: `CONTINUATION_HANDOFF.md`
- Coordinator: `serverTest/client_drs.py`
- Worker transport: `serverTest/client_admm.py`
- Worker solver: `serverTest/main.cpp`
- Protocol: `serverTest/proto/test.proto`
- General runner: `serverTest/run_drs_failure_top3_live.sh`
- Repository research memory: `/memories/repo/k1_solver.md`

When another document conflicts with this file on the current objective,
scoreboards, or next experiment, this file wins until explicitly updated.
