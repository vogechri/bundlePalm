# Pixel Versus DABA-Ray Objective Conditioning Ablation

Date: 2026-07-27

## Protocol

- scenes: 1723, 52, 245, 394, 871
- K10, 30 DRS outer iterations, one local nonlinear step
- fixed landmark-owned partition, full `9x9` block consensus, Jacobi scaling,
  acceleration, relative DRE/primal safeguard, and initialization
- changed only the optimized residual model:
  standard Snavely pixels or DABA's regularized weighted 3D ray
- every state independently evaluated in its native objective and in pixels

The ray implementation uses three residual components, fixed initial-focal
observation normalization and weights, and DABA's `delta = 1e-6` distance
regularization. Worker replies echo the objective model and the coordinator
rejects protocol mismatches.

## Pixel Cross-Evaluation

| Scene | Pixel-optimized mean px | Ray-optimized mean px | Mean gap | Pixel RMSE | Ray RMSE | Ray noninvertible |
|---|---:|---:|---:|---:|---:|---:|
| 1723 | **0.7643** | 0.7745 | +1.34% | **1.0660** | 1.0899 | 14 |
| 52 | 0.5810 | **0.5793** | -0.28% | **1.1810** | 1.1817 | 0 |
| 245 | **0.6981** | 0.7842 | +12.33% | **1.2943** | 5.7564 | 0 |
| 394 | 0.5784 | **0.5763** | -0.37% | 1.0700 | **1.0675** | 0 |
| 871 | 0.6915 | **0.6911** | -0.07% | **1.1205** | 1.1207 | 0 |

Across scenes, geometric-mean pixel quality is:

| Optimized objective | Geomean mean px | Geomean pixel RMSE | Hard rejections | Fallbacks | Total time s |
|---|---:|---:|---:|---:|---:|
| Pixel | **0.6587** | **1.1433** | 12 | 43 | **147.26** |
| DABA ray | 0.6750 | 1.5473 | **4** | **39** | 192.86 |

Ray optimization is 2.48% worse in geometric-mean mean pixel error and 35.34%
worse in geometric-mean pixel RMSE. It is 30.96% slower in this implementation,
partly because it has three residual components and currently uses the generic
Ceres Jacobian path while pixels use the optimized batched path.

## Tail Behavior

The mismatch is not a uniform degradation. Ray optimization often matches or
slightly improves median and p90/p95 pixel errors. The decisive failure is tail
control. On scene 245:

| Optimized objective | Median px | p90 | p95 | RMSE | Maximum px |
|---|---:|---:|---:|---:|---:|
| Pixel | 0.4343 | 1.4336 | 2.0778 | 1.2943 | 81.66 |
| DABA ray | **0.4256** | **1.3705** | **1.9728** | 5.7564 | 889.66 |

Thus the regularized ray objective gives a good fit for most observations but
permits rare states with extreme pixel projection error. On 1723, 14
observations have no real inverse radial projection under the fitted ray model.

## Native Objective And Stability

Both methods reduce their own native objectives on every scene and reach their
best recorded state at iteration 29, except pixel 871 at iteration 27. Native
ray and pixel scalar costs are not comparable.

Ray optimization requires substantially fewer hard outer rollbacks (`4` versus
`12`) and slightly fewer fallback trials (`39` versus `43`). This supports the
conditioning hypothesis: avoiding depth division and regularizing the ray
distance makes the distributed path easier to stabilize. It does not guarantee
the standard pixel metric users require.

Pixel-optimized states have extremely large DABA-ray cross-costs. This confirms
strong objective misalignment in both directions; it should not be interpreted
as evaluator failure because each native state exactly reproduces its recorded
optimized objective.

## Conclusion

The experiment explains part of DABA's apparent robustness. Its ray objective
is more stable under the same distributed solver, but it can hide severe pixel
outliers and inverse-projection failures. The paper should therefore retain the
standard Snavely pixel objective as the primary target and present DABA ray
results separately, with cross-evaluated pixel distributions rather than only
mean error.

All ten cases completed and all saved states independently reproduce their
recorded native objective exactly.