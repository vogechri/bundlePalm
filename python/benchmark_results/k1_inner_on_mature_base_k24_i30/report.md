# K1 Inner Algebra On The Mature DRS Backbone

Date: 2026-08-19

## Question

Can the useful K1 local-solver algebra be introduced into the mature K24 DRS
backbone without replacing or modifying the baseline implementation?

The isolated front end `serverTest/client_drs_k1_bridge.py` pins the fresh
post-crash mature policy and changes ordered factors:

1. `legacy`: historical post-Hessian tangent path and all-camera proximal terms;
2. `direct`: direct left-SE3 tangent normal equations only;
3. `direct_shared`: direct tangent plus shared-only product-space proximal terms;
4. `direct_shared_l2`: the same arm with two local nonlinear steps per oracle.

All arms use K24/I30, local Nesterov, persistent DRS trust, curvature `0.4`
recovery/decay, camera metric 75, proposal damping `0.5`, block/full consensus,
points-p95/Jacobi/preconditioning, and the legacy outer Nesterov safeguard.
There is no final Schur correction.

The run uses the clean `4db26e4` client and worker because unrelated protocol
changes are present in the main worktree. The fresh legacy control is separately
named: it is configuration-matched but reaches `1.064232x` the preserved Roman
I30 checkpoint, consistent with the documented post-crash source boundary.
Only fresh within-executable comparisons are causal.

## Result

| Arm | Roman / legacy | Trafalgar / legacy | Geomean / legacy | Completion |
|---|---:|---:|---:|---:|
| Legacy | 1.000000000 | 1.000000000 | 1.000000000 | 2/2 |
| Direct tangent | 1.000000281 | 0.999999998 | 1.000000140 | 2/2 |
| Direct + shared-only | 1.058888799 | 0.620650331 | 0.810678533 | 2/2 |
| Direct + shared-only L2 | 1.207315836 | 0.615572806 | 0.862085145 | 2/2 |

Direct+shared-only relative to direct tangent is `0.810678420x` geometrically.
Trafalgar rejections decrease from 6 to 1; Roman remains at 2. Optimization
work rises from 23.11 seconds for legacy to 38.36 seconds for shared-only.

Fixed L2 is `1.063411833x` shared-only L1 geometrically. It improves Trafalgar
only to `0.991819x` L1 but regresses Roman to `1.140172x`, raises rejections to
3/3, and costs `1.584348x` L1 optimization time.

The `direct_shared_diag` arm enables only the existing interior-defect
diagnostic. Its 30 SSE values, saved cameras, saved points, endpoint SSE, and
rejection counts are bitwise equal to `direct_shared` on both scenes.

The normalized defect does not predict the L2 outcome. Roman and Trafalgar
have similar all-iteration medians (`0.0563` and `0.0481`), each has six
clusters crossing the old window-three `0.3` activation threshold, and both
decay to about `0.0127` median by I30. Yet L2 substantially hurts Roman and
barely helps Trafalgar.

## Interpretation

Direct tangent assembly is a correctness requirement and is essential in the
K1 bridge, but it is behavior-neutral on this mature K24 sentinel. The legacy
post-Hessian path and direct per-observation path produce effectively the same
accepted endpoint here. Therefore copying or retuning the Nesterov linear
solver is not currently justified: both arms already use the same Nesterov
kernel and direct tangent does not recover K1 quality.

Shared-only semantics are the first material factor. They remove the proximal
penalty from unique interior cameras, producing a large Trafalgar gain and a
bounded Roman loss. This is consistent with earlier breadth: product-space
semantics alter local interior minimization and have scene-dependent basin
effects. Uniformly increasing local nonlinear depth is not a safe repair: L2
amplifies the Roman loss while barely changing Trafalgar. A future inner policy
requires a checkable relative forcing or stationarity diagnostic, not an
unconditional extra solve. It is not how to copy the K1 Nesterov loop.

The existing normalized interior-defect threshold is not that rule. It is
behavior-neutral but does not separate the opposite L2 outcomes.

The broader reliability cohort confirms rejection. Six development 1DSfM
scenes plus BAL1490/3068 compare behavior-neutral L1 diagnostics with fixed L2.
For `q = sqrt(interior defect / proximal displacement) > 1`, TP/FP/FN/TN is
`3/4/1/0`: precision `0.429`, recall `0.750`, and sign accuracy `0.375`.
Maximum-q rank correlation with L2 benefit is `-0.071`. See
`forcing_reliability.md` and `forcing_reliability.json`.

This rejects q as a standalone trigger for the existing full-L2 intervention.
It does not reject relative inexact-prox forcing in principle, because fixed L2
also re-updates shared cameras while q measures unique-camera/landmark
stationarity. A future test would need a true interior-only trial with shared
cameras fixed and local proximal-objective acceptance. Do not implement that
larger worker change without first preserving rollback and proving the trial is
state-neutral when rejected.

Do not promote from two scenes and do not extract a fixed-depth C++ interior
solver from this result. Existing all-15 matched structural factorials
already show a modest aggregate shared-only cost on the mature backbone, while
this pair identifies why: large heterogeneous scene effects. Any next mechanism
must target measurable interior solve accuracy or forcing under shared-only
semantics and must preserve the fresh legacy/direct controls.
