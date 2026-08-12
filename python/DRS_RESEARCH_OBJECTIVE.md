# DRS Research Objective and Restart Contract

Updated: 2026-08-10

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

Do not reopen broad K1 parameter, trust-radius, tolerance, or basin sweeps.
Do not replace `main.cpp`/`client_drs.py` with Ceres or BAE. The active task is
step 2: carry a justified finite inner algorithm into the normal K>1 DRS path
while preserving the existing consensus framework and C1--C5 switches.

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
C1+C5 was `1.129336x` base on all-15 1DSfM but `0.995443x` on all-29 BAL. A
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
4. **Collective trust trial/commit:** not implemented. Per-worker radius capture
   and one-shot restoration now support coherent race rollback, but globally
   compatible retry evaluation and commit still require a dedicated protocol.
5. **Early nonlinear distributed bootstrap:** not implemented. It should reuse
   the race/commit infrastructure and remain separately budgeted from final
   polishing; do not start it until the short-race failure is understood.

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
   Current active gate: C4 single-trajectory inner-only PCG/Nesterov carry-over,
   first K2 and then K24 if safe, with outer acceleration off.
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
