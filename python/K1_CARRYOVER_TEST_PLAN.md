# K1 Carryover Test Plan

Date: 2026-08-20
Status: active; Phase 0 complete
Authoritative source checkpoint: `d61e292`

This file is the restart contract for the next K1-to-K>1 experiments. Read it
together with `DRS_RESEARCH_OBJECTIVE.md` after any crash or context loss. Do
not infer the active phase from the newest log directory.

## Objective

Integrate the already successful one-step distributed shared-camera Schur move
into late product-space DRS without replacing DRS, introducing scene routing,
or turning the method into a repeated Schur-polishing tail.

The primary hypothesis is:

> A single safeguarded global camera proposal can be followed by useful DRS
> iterations if all local primal, product-space, rollback, metric, and trust
> state is rebuilt coherently around the accepted corrected cameras.

The secondary hypothesis is:

> A rollback-safe local trial with shared cameras fixed can improve unique
> cameras and landmarks without the cross-cluster basin damage caused by the
> existing full `local_steps=2` intervention.

## Established Evidence

Do not rerun these factors unless a validation test detects drift:

- Direct left-SE3 tangent assembly is required and is behavior-neutral relative
  to the mature tangent path on the matched K24 sentinel: `1.000000140x`.
- Exact `T^T M T`, shared-only product-space semantics, points-p95/Jacobi
  scaling, landmark preconditioning, and the current Nesterov kernel are
  already integrated.
- Fixed L2 is rejected: `1.063412x` L1 SSE at `1.584348x` optimization time.
- The current interior-forcing ratio is rejected as a fixed-L2 trigger:
  precision `0.429`, sign accuracy `0.375`, rank correlation `-0.071`.
- One frozen endpoint correction improves all-15 1DSfM to `0.793908x` its
  mature endpoint and transfers safely to BAL29 at `0.999518x` actual
  pre-correction SSE, W/T/L `26/3/0`.
- On Roman/Trafalgar, the distributed correction is `0.970927x` the one-step
  K1 oracle, so copying more K1 local iterations is not the next mechanism.
- Existing mid-correction reset and dual-offset-preserving transport do not
  solve continuation. Transport/reset is `1.020074x` on the 1DSfM pair and
  `1.002651x` on the BAL pair. Do not broaden those arms.

Canonical reports:

- `benchmark_results/k1_inner_on_mature_base_k24_i30/report.md`
- `benchmark_results/k1_inner_on_mature_base_k24_i30/forcing_reliability.md`
- `benchmark_results/k1_mature_i90_endpoint_oracle_i1/report.md`
- `benchmark_results/mid_schur_transport_i10_gate/report.md`

## Non-Negotiable Controls

Every experiment in this plan must:

1. keep K24 distributed product-space DRS and one global policy;
2. keep direct left-SE3, exact tangent metrics, shared-only semantics, and the
   mature Nesterov/DRS-trust/curvature-0.4/metric-75/C1 backbone;
3. allow at most one accepted global Schur correction per trajectory;
4. use start camera/landmark damping `0.005859375`, doubling after rejection;
5. use `bsr_low_memory`, Jacobi PCG, and relative tolerance `1e-6`;
6. accept a correction only with `linearTermination == 0`, finite lower
   physical SSE, positive damped model gain, and residual below `1e-6`;
7. preserve the input state exactly if every correction attempt is rejected;
8. retain best-state restoration after an accepted correction;
9. report physical pixel SSE, correction/model telemetry, optimization time,
   worker CPU, transport, peak RSS, and fallback damping;
10. never use BAL49 for selection or promotion. It is smoke-only.

No repeated correction tail, scene-specific trigger, weakened residual gate,
Ceres replacement, BAE replacement, product-SO3 promotion, or local-step sweep
is in scope.

## Cohorts

Development selection uses only:

- 1DSfM: Gendarmenmarkt, Piccadilly, Roman Forum, Trafalgar, Union Square,
  Vienna Cathedral;
- BAL: 52, 245, 1490, 1778, 3068.

Frozen confirmation uses:

- held-out 1DSfM: Alamo, Ellis Island, Madrid Metropolis, Montreal Notre Dame,
  Notre Dame, NYC Library, Piazza del Popolo, Tower of London, Yorkminster;
- final breadth: all 15 1DSfM and all 29 BAL.

Held-out or BAL29 results may reject a policy but may not select a replacement.
Any policy change after held-out evaluation restarts confirmation.

## Phase 0: Make the Protocol Durable

The successful endpoint campaign used uncommitted changes in
`/home/chvogel/bundlePalm_k1_long/python`. Before new solver work, port only
these default-preserving changes into the main branch:

1. permit zero-iteration final-Schur runs when solver switch, local rebase, and
   acceleration restart controls are disabled at zero;
2. reject positive switch/rebase/restart iterations when `iterations == 0`;
3. expose camera and landmark Schur damping through
   `run_drs_failure_top3_live.sh` and encode nondefault damping in the variant;
4. retain focused regression tests for the validation behavior.

Validation:

```text
pytest focused zero-iteration and bridge tests
python py_compile for new Python tooling
bash -n for experiment runners
full maintained serverTest suite
worker build
```

Commit these changes before launching Phase 1. Record the new commit below and
never launch a benchmark from a dirty source tree.

Phase 0 commit: `3e1cf1f`

Validation record:

- focused detached coherent-worktree tests: `4 passed`, `47 deselected`;
- main Python and shell syntax checks: passed;
- patch is byte-identical to the detached implementation used for the
  successful endpoint campaign;
- main focused-suite collection is blocked by a pre-existing edited-source
  mismatch: `client_drs.py` imports `drs_state_for_consensus`, absent from the
  current `drs_consensus.py`;
- broad detached collection is blocked by the pre-existing missing
  `analyze_sfm_solver_state_gap` module;
- main worker build is blocked by the pre-existing protobuf/generated-source
  mismatch in the dirty protocol worktree.

Do not repair or revert those unrelated files as part of this campaign. Use
the coherent detached worker/client for benchmark execution until the user's
protocol edits are reconciled separately.

## Phase 1: Freeze Matched Controls

Create a dedicated runner and analyzer under `serverTest/` and write outputs to:

```text
benchmark_results/k1_carryover_late_correction/
```

Required matched arms at K24/I90:

- `mature`: integrated mature backbone, no Schur correction;
- `final_one`: mature backbone plus the proven one correction at termination;
- `mid_reset_reference`: one I60 correction followed by the existing collapse
  and trust rebase, development sentinels only;
- `mid_rebuild`: one I60 correction followed by the coherent rebuild from
  Phase 2.

I60 is a single frozen late checkpoint, not a duration sweep. The
`mid_reset_reference` arm is a discriminating control for the new rebuild; do
not run existing transport or reset beyond development.

Before quality runs, prove on Roman, Trafalgar, BAL52, and BAL3068 that
`mature` and `final_one` reproduce their committed references within `1e-10`
relative SSE when deterministic and otherwise within the measured repeat
envelope. Any larger mismatch blocks the phase.

## Phase 2: Coherent Correction Rebuild

Implement one default-off rebuild mode after an accepted mid-run correction.
The rebuild must not preserve stale dual offsets. It must:

1. install the accepted corrected shared cameras and recovered landmarks;
2. snapshot coordinator and worker accepted/rollback state;
3. hold duplicated shared cameras fixed at the corrected consensus;
4. run one local refresh of unique cameras and owned landmarks;
5. reject and restore atomically if any local proximal objective increases or
   becomes nonfinite;
6. reconstruct local camera copies, centers, consensus, metric state,
   accepted/best states, worker landmark slots, and trust/rollback state from
   the refreshed accepted geometry;
7. verify the reconstructed consensus has zero shared-camera copy disagreement
   before the next ordinary DRS oracle;
8. continue ordinary DRS with no second Schur correction.

Required tests before a benchmark:

- default-off trajectory is bitwise unchanged;
- shared cameras are bitwise fixed during local refresh;
- accepted refresh does not increase any local proximal objective;
- rejected refresh restores coordinator and worker state byte-for-byte where
  serializable and numerically exactly elsewhere;
- correction rejection is a complete no-op;
- rebuild with a zero camera step preserves physical geometry;
- worker/global SSE agreement remains within the existing tolerance;
- first post-rebuild DRS iteration is finite and uses reconstructed, not stale,
  metric/trust state.

Development promotion gate for `mid_rebuild`:

- all rows complete;
- no accepted nonconverged correction;
- no regression relative to the immediate accepted correction state because
  best-state restoration remains active;
- `mid_rebuild/mid_reset_reference < 1` geometrically on both development
  families;
- post-correction DRS produces additional accepted progress on both families;
- worst regression versus `final_one` is reported and below `1.02x`;
- aggregate `mid_rebuild/final_one` is at most `1.00x` on both families before
  held-out expansion.

If this gate fails, stop correction scheduling work. Keep the rebuild as
negative evidence and proceed only to the isolated interior trial if its unit
contract remains independently useful.

## Phase 3: Test One Global Late Trigger

Only after Phase 2 passes, replace fixed I60 with one deterministic global
trigger:

```text
At k >= 60, trigger once when the best physical SSE has not improved during
five consecutive completed outer iterations. If this never occurs, apply the
same correction at the I90 endpoint.
```

This rule has no scene parameter and no tunable continuous threshold. Internal
damping retries belong to the one proposal event. After an accepted pre-I90
correction, use the Phase 2 rebuild and continue DRS; never apply a second
correction.

Development arms:

- `final_one`;
- `mid_rebuild_i60`;
- `stagnation5_rebuild`.

Promotion requires all Phase 2 safety gates plus:

- exactly zero or one correction event and at most one accepted correction;
- trigger iteration and reason recorded for every row;
- `stagnation5_rebuild/final_one <= 1.00x` geometric SSE on both development
  families;
- no material contradiction in summed SSE;
- correction and rebuild overhead reported separately.

Do not tune warmup 60, window 5, or add a second trigger after viewing results.
A failure closes adaptive scheduling for this mechanism.

## Phase 4: Isolated Shared-Fixed Interior Trial

This is lower priority and must remain independent of Phase 3 promotion.
Reuse the rollback-safe fixed-shared-camera primitive from Phase 2, but test it
without a global Schur move.

Matched development arms:

- `shared_l1`: one ordinary local step;
- `shared_l1_interior_trial`: one ordinary step plus one shared-fixed interior
  trial accepted per cluster only on local proximal-objective decrease;
- `final_one`: endpoint-correction quality reference, not an equal-work local
  arm.

The trial may update only unique cameras and owned landmarks. It must not alter
shared cameras, centers, dual offsets, penalty, or trust state when rejected.
Report accepted-cluster fraction and additional local work.

Promotion requires:

- state-neutral rejection tests;
- completion and finite states everywhere;
- geometric SSE below `shared_l1` on both development families;
- no scene worse than `1.02x` `shared_l1`;
- aggregate optimization overhead below `1.25x`;
- no use of the rejected q-ratio or full `local_steps=2` as a selector.

If it fails, close interior forcing and retain coordinated global correction as
the sole K1-derived optimization carryover.

## Phase 5: Frozen Confirmation

For each candidate that passes development:

1. freeze source commit, runner, flags, trigger, and cohorts;
2. run nine held-out 1DSfM scenes once;
3. run all 15 1DSfM and all 29 BAL unchanged;
4. run deterministic repeats after one warm-up on Roman, Trafalgar, BAL52, and
   BAL3068;
5. compare against `mature`, `final_one`, and Ceres left-SE3;
6. report geometric and summed SSE, W/T/L, worst case, accepted/rejected
   correction counts, damping distribution, residuals, time, CPU, transport,
   and peak RSS.

Final promotion requires one common policy, no accepted nonconverged solve, no
material aggregate regression on either family, no unsafe tail, and a clear
quality or quality-time advantage over `final_one`. Otherwise retain
`final_one` as the endpoint correction and report the candidate as negative.

## Crash and Resume Contract

Each phase must have its own subdirectory and status file:

```text
benchmark_results/k1_carryover_late_correction/
  phase0_protocol/
  phase1_controls/
  phase2_rebuild_development/
  phase3_trigger_development/
  phase4_interior_development/
  phase5_heldout/
  phase5_all15_all29/
  campaign_state.md
  report.md
```

Every runner must support `OVERWRITE=0` and skip only a row whose JSONL result
and status entry both say completed. Write one result row immediately after each
case. Never infer completion from an NPZ file.

At phase start, record:

- source commit and `git status --short`;
- worker binary hash;
- exact command and environment;
- dataset manifest and cohort role;
- output directory and ports;
- baseline artifact paths.

At phase completion:

1. run the phase analyzer with a completeness assertion;
2. update the status table below and `campaign_state.md`;
3. write a compact Markdown/JSON report;
4. commit source, tests, runner, report, compact JSONL, and TSV evidence;
5. leave logs, memory traces, and NPZ states untracked;
6. update `DRS_RESEARCH_OBJECTIVE.md` and `CONTINUATION_HANDOFF.md` before the
   next phase.

Large BAL saved states have already proved unreliable. Prefer in-process
correction/rebuild for BAL245/427/744/951/1490/1778. If restart from NPZ is
required, independently evaluate its physical SSE against the recorded state
before use; mismatch above `1e-6` relative quarantines the checkpoint.

Before reusing ports after a crash, verify no coordinator or worker remains
alive. Do not delete a partial output directory; resume it with `OVERWRITE=0`
or quarantine it under an explicit `_invalid_<reason>` name.

## Phase Status

| Phase | Status | Commit | Authoritative artifact | Decision |
|---|---|---|---|---|
| 0 protocol durability | complete | `3e1cf1f` | focused tests and syntax checks | proceed |
| 1 matched controls | complete | `f4d5e8b` | `benchmark_results/k1_carryover_late_correction/phase1_controls/report.md` | proceed |
| 2 canonical rebuild oracle | complete, cross-family negative | pending commit | `benchmark_results/k1_carryover_late_correction/phase2_restart_oracle_i90_stop60/report.md` | stop scheduling |
| 3 global late trigger | closed | -- | Phase 2 result | do not tune |
| 4 interior-only trial | active | -- | -- | -- |
| 5 frozen confirmation | blocked by development | -- | -- | -- |

## Immediate Next Action

Implement only the rollback-safe shared-fixed interior trial and its
state-neutrality tests. The matched Phase 2 process-restart oracle requested an
I90 horizon, matched all first 60 trajectory rows exactly, applied one frozen
correction, and restarted fresh DRS for I30. Restart/terminal-correction is
`0.933819x` on Roman/Trafalgar but `1.010695x` on BAL52/3068. This fails the
cross-family gate, so do not tune correction timing or implement the Phase 3
trigger. The common late-correction policy remains terminal-only.
