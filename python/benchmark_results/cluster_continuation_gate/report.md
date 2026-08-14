# K4 to K24 Cluster-Count Continuation Gate

Date: 2026-08-13

## Transfer Contract

`client_drs.py --state` stores canonical solver cameras and points. Historically,
`--initial-state` interpreted every NPZ as raw BAL geometry and canonicalized it
again, so direct cross-process chaining was unsafe. The coordinator and runner
now support an explicit `--initial-state-frame canonical`; raw remains the
backward-compatible default. Every staged gate reports zero relative SSE error
between the K4 endpoint and K24 initial state.

A fresh K24 process intentionally resets DRS centers, worker trust state,
scaling, acceleration, and partition ownership. Only accepted canonical geometry
is transferred. This is cluster-count continuation, not full optimizer-state
resume.

## Solver Gate

The maintained Nesterov local solver is not viable at K4 under this test. Roman
K4 overflows its tangent camera step before either DRS or DABA trust can reject
it (`step_norm=inf`, camera back-transform non-finite). The stable gate therefore
uses the project's Schur-PCG local solver with block-Jacobi preconditioning and
DABA trust in both staged and direct arms.

Frozen global schedule:

- acquisition: K4/I3;
- target: K24 for the remaining equal iteration budget;
- direct control: K24 for the complete budget;
- direct left-SE3 tangent equations, points-p95 normalization, initial Jacobi
  scaling, shared-only proximal terms, and full 9x9 consensus;
- no scene-specific settings or branch selection.

## K24/I6 Shallow Gate

| Scene | Staged/direct SSE | Staged/direct time |
|---|---:|---:|
| Roman Forum | 0.776041 | 1.888726 |
| Trafalgar | 1.286742 | 1.817298 |
| BAL1490 | 0.957964 | 1.961432 |
| **Geometric mean** | **0.985315** | **1.888235** |

The shallow signal is mixed and expensive. It was advanced unchanged because
Roman and BAL1490 improve and the aggregate is positive.

## K24/I30 Persistence

| Scene | Staged/direct SSE | Staged/direct time |
|---|---:|---:|
| Roman Forum | 0.744715 | 1.473755 |
| Trafalgar | 0.945498 | 1.209459 |
| BAL1490 | 0.997938 | 1.365497 |
| **Geometric mean** | **0.889034** | **1.345145** |

Continuation wins all three scenes at equal total outer iterations. Roman gains
`25.5%`, Trafalgar `5.45%`, and BAL1490 `0.21%`. This is a real fast-horizon
frontier, albeit with a substantial runtime premium from two process
initializations and two partitions.

## K24/I60 + 16 Schur Quality Gate

Both arms receive the same 16-correction final global Schur workflow.

| Scene | Pre-Schur handoff | Final SSE | Total time |
|---|---:|---:|---:|
| Roman Forum | 0.725244 | 0.953985 | 1.031584 |
| Trafalgar | 1.009605 | 1.038197 | 0.899932 |
| BAL1490 | 1.002656 | 1.002877 | 1.118126 |
| **Geometric mean** | **0.902116** | **0.997753** | **1.012516** |

The I30 advantage does not remain uniformly safe. Roman still wins `4.60%`, but
Trafalgar regresses `3.82%` and BAL1490 regresses `0.29%`. Final geometric mean
is effectively neutral.

## Decision

Retain K4/I3 to K24 continuation as a default-off fast-horizon diagnostic and a
promising I30 frontier. Do not add it to the named I60+16 quality preset because
cross-scene endpoint safety fails. Do not tune K or the split on these three
outcomes. A broader fixed I30 transfer gate would be justified only if the
runtime premium is acceptable; quality promotion would require a mechanism
that preserves the early basin advantage through polishing.
