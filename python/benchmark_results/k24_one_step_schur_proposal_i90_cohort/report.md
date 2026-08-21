# K24 I90 One-Step Schur Proposal Cohort

The frozen seven-scene cohort was run through I120 with one proposal at I90.
Every candidate matches its frozen control exactly through I89. The proposal
uses one shared-camera Schur matvec/Jacobi correction, eight fixed scales from
`1` through `1/128`, precise worker-SSE selection, and atomic DRS state rebuild.

| Scene | Selected scale | Worker-SSE ratio | I90 ratio | I120 ratio | Control/candidate rejections |
|---|---:|---:|---:|---:|---:|
| Montreal Notre Dame | 0.5 | 0.781501 | 0.781512 | 0.782042 | 13 / 13 |
| Piazza del Popolo | 0.5 | 0.847381 | 0.847566 | 0.855744 | 15 / 15 |
| Roman Forum | 0.5 | 0.886890 | 0.888157 | 0.797016 | 11 / 8 |
| Trafalgar | 0.25 | 0.983615 | 0.983759 | 0.983698 | 16 / 16 |
| Yorkminster | 0.25 | 0.961324 | 0.961766 | 0.983972 | 14 / 16 |
| BAL52 | 0.0625 | 1.000000 | 0.999999 | 1.000000 | 0 / 0 |
| BAL3068 | declined | 1.000000 | 1.000000 | 0.999416 | 3 / 3 |

All seven cases complete. The five-scene 1DSfM geometric I120 ratio is
`0.876147`, W/L `5/0`. The two BAL controls are effectively neutral at
`0.999708` geometrically; BAL3068 declines the proposal, so its small separate-
run endpoint difference is not an applied-proposal effect.

## Crash Recovery

The initial breadth command caused two abrupt WSL exits while BAL3068 entered
the I90 diagnostic. Its log ends after I89 and both `/usr/bin/time` files are
empty, consistent with termination outside the child processes. Kernel OOM
records did not survive the WSL restart.

The run still enabled the already-rejected coupled-consensus oracle inside the
proposal diagnostic. That oracle materializes a large sparse coupled system and
is irrelevant to the applied one-step proposal. Trafalgar had already reached
`8,768,184 KiB` coordinator RSS. The coupled oracle is now explicit opt-in.

BAL3068 was resumed alone with the same solver/proposal policy, the coupled
oracle disabled, `MALLOC_ARENA_MAX=2`, and a `14 GiB` per-process virtual-memory
ceiling. It completed in 185 seconds with coordinator/worker peaks
`8,358,544/4,603,516 KiB`. Current system headroom returned to 24 GiB available.

Do not rerun large proposal cohorts with `--schur-coupled-consensus-oracle`.
Run large BAL cases alone with a process memory ceiling and resume completed
rows with `OVERWRITE=0`.

## Decision

The proposal passes this frozen transfer gate and remains our distributed DRS
acceleration, not Ceres and not a full Schur correction. Before all-15/all-29
breadth, create a resumable runner/analyzer that defaults to low-memory proposal
telemetry and serializes large BAL cases under an explicit memory ceiling.