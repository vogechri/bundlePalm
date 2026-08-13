# Repeated Initial Distributed Schur Gate

## Mechanism

The existing accepted initial distributed Schur correction can now take a
bounded number of nonlinear corrections before DRS. Each additional correction
rebuilds the exact local Schur systems at the accepted state, requires converged
PCG and a strict independently evaluated physical-SSE decrease, halves camera
and landmark damping after acceptance, and stops on rejection or the existing
minimum-relative-decrease rule. The default maximum remains one.

The frozen candidate uses K24, at most three startup corrections,
`bsr_low_memory`, Jacobi PCG at relative tolerance `1e-6`, and damping sequence
`3, 1.5, 0.75`. No scene-specific setting is used.

## Large-BAL I5 Screen

| Scene | Accepted | Stop | Final SSE | Cap-3/raw |
|---|---:|---|---:|---:|
| BAL 1490 | 2 | rejected PCG | 4,171,867.712 | 0.970891 |
| BAL 1778 | 0 | rejected PCG | 5,060,949.722 | 1.000089 |
| BAL 3068 | 3 | correction cap | 6,081,051.113 | 0.825007 |

The geometric ratio to the preserved raw controls is `0.928729x`. All accepted
steps have converged PCG. BAL1490's third step lowers candidate SSE but is
correctly rejected at the 1000-iteration cap. On current source BAL1778's first
solve also reaches the cap; matched current cap-one and cap-three behavioral
trajectories and endpoints are bitwise identical. The older preserved
low-memory/1000 artifact converged on 1778, so it is not a current-source
control for this gate.

## Matched Current-Source I30 Gate

| Scene | Cap-3/cap-1 SSE | Cap-3/cap-1 time | Result |
|---|---:|---:|---|
| BAL 1490 | 1.000860 | 1.608262 | bounded loss |
| BAL 3068 | 0.917651 | 1.995019 | win |
| **Geometric mean** | **0.958353** | **1.791232** | mixed aggregate gain |

The startup advantage decays substantially during ordinary DRS. The result is
aggregate-positive versus one correction, but it is not a robust long-DRS
prefix and costs materially more.

## I60+16 Quality Composition

The unchanged quality workflow was tested on Roman Forum and Trafalgar with
the same cap-three startup:

| Scene | Candidate/control SSE | Candidate/control time |
|---|---:|---:|
| Roman Forum | 0.962247 | 1.020778 |
| Trafalgar | 1.081335 | 0.995565 |
| **Geometric mean** | **1.020055** | **1.008093** |

All three startup corrections and all sixteen final corrections are accepted
with converged linear solves. Roman's I60 handoff improves `15.12%`, while
Trafalgar's improves only `1.41%`; final polishing then leaves an `8.13%`
Trafalgar regression.

## Decision

Retain repeated startup correction as default-off short-handoff diagnostic
infrastructure. Reject it as an addition to the named quality workflow and do
not broaden to the fixed six/held-out 1DSfM ladder. More initial corrections do
not provide a globally robust basin policy; the next mechanism must change how
the distributed trajectory preserves or acquires the basin, not extend this
correction count.
