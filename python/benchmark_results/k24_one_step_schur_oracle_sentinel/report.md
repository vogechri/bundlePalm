# K24 One-Step Schur Residual Oracle

Roman Forum and BAL52 were run through I120 with behavior-neutral diagnostics
at I90 and I120. Their trajectory prefixes reproduce the frozen I200 diagnostic
exactly.

Starting from the nominal shared DRS tangent, the oracle applies one restricted
global Schur matvec and one block-Jacobi residual correction. It does not run
PCG, apply a camera update, or mutate DRS state.

| Scene | I | Nominal cosine | One-step cosine | Nominal/Schur norm | One-step/Schur norm | Nominal camera gain | One-step/Schur gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| Roman | 90 | 0.063400 | 0.762741 | 0.053248 | 0.694855 | 329.999 | 0.880592 |
| Roman | 120 | 0.134128 | 0.774795 | 0.035384 | 0.706848 | 986.591 | 0.887622 |
| BAL52 | 90 | 0.554641 | 0.928512 | 1.204499 | 0.887096 | 3.204 | 0.874758 |
| BAL52 | 120 | 0.412323 | 0.950051 | 1.886274 | 0.998320 | 2.094 | 0.969446 |

One distributed curvature action repairs most of the direction and model gap
on both families. This is substantially different from projection-only coupled
metrics and the transient diagonal majorizer, both of which failed.

The next gate applies at most one I90 proposal. It converts the one-step tangent
to a candidate consensus state, evaluates its precise worker SSE against the
ordinary candidate, and selects it only when lower. Selection must rebuild DRS
centers/residuals atomically through `drs_state_for_consensus`. The gate remains
default-off and must report Roman/BAL52 quality, selection, safeguards, and
exact no-selection behavior.