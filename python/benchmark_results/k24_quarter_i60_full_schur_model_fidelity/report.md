# K24 I60 Full-Schur Physical Oracle

A converged shared-camera Schur tangent and the promoted two-direction Krylov tangent are evaluated with identical eight-scale searches, three landmark-response steps, and rollback-safe worker state. All 120 ordinary-control trajectory rows are exact.

| Scene | Krylov scale | Krylov/ordinary | Full scale | Full/ordinary | Full/Krylov | Model/actual preference | Schur residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| gendarmenmarkt | 1 | 0.976786612 | 0.25 | 0.987349579 | 1.010813996 | krylov/krylov | 8.265e-07 |
| madrid_metropolis | 1 | 0.998736863 | declined | 1.000000000 | 1.001264735 | krylov/krylov | 9.981e-07 |
| tower_of_london | 1 | 0.888135051 | 1 | 0.853252002 | 0.960723261 | full/full | 9.049e-07 |
| yorkminster | 1 | 0.790326602 | 0.5 | 0.795729545 | 1.006836343 | full/krylov | 9.308e-07 |

Full/Krylov geometric ratio is `0.994705087x`, W/T/L `1/0/3`. Full Schur helps Tower, but loses on Gendarmenmarkt and Yorkminster and declines Madrid. More linear convergence is therefore not a transferable proposal mechanism. The hybrid linear model ranks the two directions correctly on three scenes but incorrectly favors full Schur on Yorkminster. Keep bounded Krylov2 and move to basin/model diagnosis.

Gate status: **passed**.
