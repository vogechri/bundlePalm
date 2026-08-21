# K24 Coupled-Consensus Oracle Sentinel

Roman Forum and BAL52 were rerun through I200 with a behavior-neutral coupled
consensus oracle. Both reproduce the reference trajectories, endpoint cameras,
and endpoint points exactly.

The oracle converts the same reflected copies to left-SE3 tangents and projects
them with each cluster's reduced Schur blocks plus the frozen Frobenius
off-diagonal diagonal majorizer. It is computed but never applied.

| Scene | I | Nominal cosine | Coupled cosine | Coupled/Schur norm | Coupled camera-model gain | Coupled/Schur model gain |
|---|---:|---:|---:|---:|---:|---:|
| Roman | 90 | 0.063400 | 0.000364 | 336.705 | -2.989e9 | -5737.37 |
| Roman | 120 | 0.134128 | 0.000156 | 452.719 | -5.003e9 | -10018.4 |
| Roman | 160 | 0.127310 | 0.000140 | 1249.47 | -3.675e10 | -74747.7 |
| Roman | 200 | 0.043521 | 0.000071 | 439.083 | -4.474e9 | -9161.13 |
| BAL52 | 90 | 0.554641 | 0.131096 | 274.594 | -2.263e5 | -36980.1 |
| BAL52 | 120 | 0.412323 | -0.003780 | 3870.71 | -9.921e6 | -2.671e6 |
| BAL52 | 160 | 0.480670 | -0.004747 | 26512.0 | -3.700e8 | -3.760e8 |
| BAL52 | 200 | 0.128226 | 0.008602 | 29388.7 | -3.522e8 | -3.838e8 |

The oracle fails decisively. Cross-camera curvature cannot be inserted only in
the coordinator projection after local copies were generated under the block
metric. That mixes two product-space metrics and destroys both families.

The next consistent experiment is the frozen transient Frobenius Schur
majorizer, which uses the same majorized metric in local proximal solves and
consensus projection. The recovered worker client currently raises
`NotImplementedError` for its observability and majorizer dispatch, so restore
that verified historical transport path before attempting the matched mature
C1/Nesterov Roman/BAL52 gate.