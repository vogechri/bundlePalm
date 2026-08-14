# I10 Mid-Run Global Schur Rebase Gate

Date: 2026-08-14

## Mechanism

After ten accepted K24 outer iterations, materialize the worker's accepted
landmarks, build one global distributed Schur system, and apply one correction.
Commit only a converged, strictly physical-SSE-decreasing candidate. On
acceptance, reset product-space centers to the corrected consensus, save the
accepted/best landmark snapshots, reset recovery/acceleration state, and rebase
worker trust without changing geometry. On rejection, restore the original
accepted state exactly.

The feature is default-off through
`--mid-shared-schur-correction-iteration`; the frozen gate uses iteration 10,
camera/landmark damping 3, low-memory BSR, Jacobi-preconditioned CG at `1e-6`,
and three fixed-camera landmark refinement steps.

A Nesterov/Themelis pilot overflowed before reaching I10 in the local tangent
step. The behavior gate therefore uses the stable Schur-PCG/DABA K24 policy in
both candidate and control arms.

## Four-Scene Sentinel

| Scene | Immediate SSE ratio | Final I30 ratio | Time ratio |
|---|---:|---:|---:|
| Roman Forum | 0.772229 | 0.969760 | 1.107711 |
| Trafalgar | 0.751587 | 0.939618 | 1.015300 |
| Montreal Notre Dame | 0.674685 | 0.860366 | 1.136294 |
| BAL3068 | 0.873450 | 0.993657 | 1.440725 |
| **Geometric mean** | - | **0.939472** | **1.164859** |

All four corrections converge and are accepted. Several subsequent DRS
iterations reject, so most quality gain comes directly from the Schur
correction rather than improved post-rebase dynamics.

## Breadth Versus Direct I30

| Family | Accepted | Geometric SSE | Summed SSE | W/T/L | Time |
|---|---:|---:|---:|---:|---:|
| 1DSfM all 15 | 15/15 | 0.977835 | 0.948698 | 12/0/3 | 1.822800 |
| BAL all 29 | 27/29 | 1.017020 | 1.013890 | 2/2/25 | 2.040605 |

1DSfM losses are Gendarmenmarkt `1.088646x`, Madrid `1.135156x`, and NYC
Library `1.073025x`. BAL's worst loss is scene 52 at `1.092448x`; BAL1490 and
BAL1778 decline the correction through nonconverged linear solves and reproduce
the control endpoint.

## Time-Matched Controls

The rebased I30 workflow is not a Pareto improvement.

- Against direct K24/I40 on all 15 1DSfM scenes: geometric/summed SSE
  `1.030822x/1.001725x`, W/T/L `3/0/12`, while candidate/direct time is
  `1.419161x`.
- Against direct K24/I60 on 1DSfM: geometric/summed SSE
  `1.088164x/1.064560x`, W/T/L `2/0/13`; the candidate uses `0.688761x` the
  I60 time, so this is a farther quality control rather than the nearest time
  match.
- Against direct K24/I60 on BAL29: geometric SSE `1.025366x`, W/T/L `1/0/28`,
  while candidate/direct time is `1.185564x`.

## Decision

Reject fixed I10 mid-run Schur rebasing as a global or family-specific policy.
The correction is an effective one-step physical-SSE reducer, but the
product-space restart does not preserve that advantage efficiently. Subsequent
DRS often stalls or rejects, and direct K24 continuation reaches better quality
with less or comparable work.

Keep the implementation default-off as a diagnostic for measuring the gap
between the current distributed trajectory and an accepted global Schur step.
Do not tune the trigger iteration or damping from these outcomes.

Artifacts were produced by `serverTest/run_mid_schur_rebase_breadth.sh`.
