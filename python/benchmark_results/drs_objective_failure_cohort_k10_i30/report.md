# Pixel Versus DABA-Ray Objective on the Failure Cohort

Date: 2026-07-27

## Scope

This experiment isolates the objective model inside the mature DRS solver. It
does not run DABA's MM algorithm or DABA's ADMM baseline.

- scenes: Ladybug 646, 931, 1064, and 1266;
- K10, 30 outer iterations, one local nonlinear step;
- fixed scalable landmark partition, full `9x9` consensus metric, Jacobi
  scaling, acceleration, relative safeguard, and initialization;
- changed only the optimized residual: Snavely pixels or DABA's regularized
  weighted 3D ray;
- every saved state was evaluated independently in its native objective and in
  standard pixels.

These four scenes complement the earlier five-scene objective experiment. They
belong to the broad ADMM baseline's failure cohort, but the mature full-block
DRS configuration used here is already substantially more stable than that
baseline.

## Pixel Cross-Evaluation

| Scene | Pixel mean px | Ray mean px | Pixel RMSE | Ray RMSE | Pixel / ray max px | Ray noninvertible |
|---:|---:|---:|---:|---:|---:|---:|
| 646 | **0.7435** | 0.7532 | **1.0523** | 1.0814 | 28.26 / 33.70 | 0 |
| 931 | **0.7514** | 0.7604 | **1.0508** | 1.0677 | 24.06 / 24.90 | 0 |
| 1064 | **0.7567** | 0.7668 | **1.0542** | 1.0739 | 14.46 / 22.67 | 0 |
| 1266 | **0.7577** | 0.7698 | **1.0627** | 1.0844 | 21.04 / 22.93 | 0 |

| Aggregate | Pixel objective | DABA-ray objective | Ray gap |
|---|---:|---:|---:|
| Geomean mean px | **0.7523** | 0.7625 | +1.36% |
| Geomean pixel RMSE | **1.0550** | 1.0768 | +2.07% |
| Hard best-basin restorations | 11 | **9** | -2 |
| Total acceleration resets | 16 | **15** | -1 |

Both objectives have finite native trajectories and reduce their own objective
on every scene. Best states occur at iteration 28 or 29. Native ray and pixel
cost values are not comparable to each other.

## Interpretation

The DABA ray objective gives a small stability benefit on this cohort, reducing
hard restorations from 11 to 9. It does not eliminate the need for the mature
consensus metric or safeguard, and it is consistently worse under the standard
pixel objective. Unlike the earlier scene-245 result, these four scenes show no
catastrophic pixel tail or inverse-projection failure.

The defensible conclusion is that DABA's ray geometry improves conditioning,
but only modestly on these historical failure scenes once full-block consensus
and accepted-state restoration are active. It is not the main explanation for
their recovery, and it should remain a separate objective ablation rather than
replace the paper's pixel target.
