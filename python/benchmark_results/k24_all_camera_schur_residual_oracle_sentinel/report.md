# K24 All-Camera Schur Residual Oracle Sentinel

A behavior-neutral I90 oracle applies one block-Jacobi Schur residual correction to the full nominal camera tangent, including unique cameras. It uses the same eight worker-SSE scales and `1e-3` floor as the retained shared-only correction. The all-camera candidate is telemetry-only; shared-only depth one remains applied.

| Scene | Shared-only SSE ratio | All-camera scale | All-camera SSE ratio | All/shared candidate | Active cameras | Correction norm | Trajectory/state |
|---|---:|---:|---:|---:|---:|---:|---|
| gendarmenmarkt | 0.993459701 | 0.25 | 0.995379928 | 1.001932869 | 567 | 6.13686e13 | exact |
| roman_forum | 0.886889776 | declined | 1.000000000 | 1.127535831 | 961 | 2.16217e11 | exact |
| bal52 | 1.000000000 | declined | 1.000000000 | 1.000000000 | 52 | 0.708508 | exact |

The all-camera correction is rejected. It is worse than shared-only on Gendarmenmarkt, has no admissible Roman candidate, and exposes enormous unique-camera correction norms on 1DSfM. BAL52 still declines. This confirms that unique cameras should remain locally owned in the proposal; the globally coupled residual action is useful only as a repair of duplicated-camera consensus motion. Do not broaden or scale-tune the all-camera route.
