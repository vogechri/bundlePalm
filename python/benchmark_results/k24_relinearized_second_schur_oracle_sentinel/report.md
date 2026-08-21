# K24 Relinearized Second Schur Oracle Sentinel

A behavior-neutral I90 oracle rebuilds the distributed Schur systems at the selected depth-one candidate, takes one fresh shared-camera block-Jacobi residual action from zero, and evaluates eight incremental worker-SSE scales. The fresh action is telemetry-only; the retained depth-one proposal remains the only applied candidate.

| Scene | Depth-one SSE ratio | Eligible | Best fresh scale | Fresh / depth one | Relative decrease | Full-scale / depth one | Trajectory/state |
|---|---:|---:|---:|---:|---:|---:|---|
| gendarmenmarkt | 0.993459701 | yes | 0.125 | 0.999545168 | 0.000454832 | 1.026235158 | exact |
| roman_forum | 0.886889776 | yes | 0.0078125 | 1.000227288 | -0.000227288 | 1.194153924 | exact |
| bal52 | 1.000000000 | no | declined | 1.000000000 | 0.000000000 | 1.000000000 | exact |

Relinearization is rejected as a second applied action. Gendarmenmarkt has only a `0.0455%` incremental decrease, below the frozen `0.1%` progress floor, while Roman has no descent at any scale. Full fresh steps regress `2.62%` and `19.42%`. BAL52 skips the rebuild because depth one declines. This closes stale linearization as the reason fixed depth two failed; do not broaden or lower the progress floor.
