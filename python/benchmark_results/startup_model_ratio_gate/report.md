# Model-Ratio Startup Schur Gate

Date: 2026-08-13

## Mechanism

The repeated initial distributed Schur correction now has an isolated,
default-off model-ratio damping policy. Correction zero preserves the existing
converged physical-SSE acceptance rule. Later corrections additionally require
a damped gain ratio above the existing minimum. Accepted corrections use the
standard LM damping update

`max(1/3, 1 - (2 rho - 1)^3)`.

Rejected corrections restore accepted geometry and retry with increasing
camera/landmark damping. The existing correction cap, retry cap, linear solve
budget, physical-SSE check, and minimum-relative-decrease stop are unchanged.
Final Schur polishing retains its independently selected damping policy.

## K24/I1 Gate

Matched cap-three startup with damping 3, `bsr_low_memory`, CG tolerance `1e-6`,
a 1000-iteration startup linear cap, and at most six attempts:

| Scene | Model-ratio/geometric SSE | Accepted corrections | Time ratio |
|---|---:|---:|---:|
| Roman Forum | 0.830477 | 3/3 | 1.118221 |
| Trafalgar | 0.865090 | 3/3 | 1.003670 |
| BAL1490 | 0.467159 | 3/2 | 1.300394 |
| **Geometric mean** | **0.694946** | - | - |

Roman and Trafalgar use damping `3, 1, 1/3` after strong gain ratios. On
BAL1490, damping `1/3` lowers physical SSE but reaches the 1000-iteration CG
cap; retry damping `2/3` converges and is accepted. The accepted BAL handoff
improves from `25,167,073.279` to `11,757,030.589`. The geometric artifact's
`12,666,928.341` candidate is rejected and is not its accepted handoff.

## Persistence

| Scene/workflow | Startup ratio | Handoff ratio | Final ratio | Time ratio |
|---|---:|---:|---:|---:|
| Roman I60+16 | 0.830477 | 0.922973 | 1.043756 | 1.188061 |
| Trafalgar I60+16 | 0.865090 | 0.946382 | 0.935121 | 1.635521 |
| BAL1490 I30 | 0.467159 accepted handoff | - | 1.007092 | 1.366498 |

Roman/Trafalgar final geometric-mean ratio is `0.987946x` at `1.393951x`
geometric-mean runtime. Roman improves at I60 but reverses during final Schur
polishing. BAL1490 loses its large startup advantage during ordinary DRS.

## Decision

Reject model-ratio startup damping as an addition to the named quality
workflow. It is a materially better bounded startup mechanism, especially when
a low-damping correction fails to converge, but its basin advantage is not
reliably preserved to the final endpoint and its runtime cost is too high for
the small aggregate final gain.

Keep the isolated startup policy and telemetry default-off for diagnostics.
Production fast, balanced, and quality presets remain unchanged.

## Follow-up: Conservative Damping Floor

A startup-only model-ratio minimum factor was added with the standard `1/3`
default. The proposed `1/2` floor was tested on the same K24/I1 gate.

| Scene | Half/geometric SSE | Half/one-third SSE | Half damping path |
|---|---:|---:|---|
| Roman Forum | 1.000000 | 1.204127 | `3, 1.5, 0.75` |
| Trafalgar | 1.000000 | 1.155949 | `3, 1.5, 0.75` |
| BAL1490 | 0.784964 | 1.680291 | `3, 1.5, 0.75(reject), 1.5(reject), 6(accept)` |

As predicted from gain ratios above one, the half floor exactly reproduces
geometric damping on Roman and Trafalgar. On BAL1490 it eventually recovers a
third correction, but requires two failed 1000-iteration solves and remains
`68.0%` worse than the one-third policy. Reject the half floor without a long
persistence run.

## Follow-up: Delayed Dual-Branch Commit

Existing matched geometric and one-third trajectories were replayed to test
whether a fixed physical-SSE checkpoint can select the final winning branch.

- BAL1490 favors model-ratio through I8, crosses to geometric at I9, and ends
	with geometric better at I30.
- Roman favors model-ratio at every I1-I60 checkpoint and through its nine
	accepted final corrections, but geometric first beats the completed model
	branch at final correction 12.
- Trafalgar favors model-ratio at I1-I10, geometric around I20-I50, then
	model-ratio again at I60 and after all final corrections.

No fixed early horizon predicts all three final winners. A safe dual-branch
policy would need to carry and polish both branches nearly to completion,
roughly duplicating the workflow. Reject delayed commit as a practical global
policy; retain full-branch comparison only as expensive diagnostic evidence.
