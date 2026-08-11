# C4 post-crash K2 inner-solver gate

This is the separately named executable post-crash baseline required by
`DRS_RESEARCH_OBJECTIVE.md`. It does not claim reproduction of the historical
pre-crash K2 trajectory.

## Frozen comparison

- scenes: Roman Forum and Trafalgar
- clusters / outer iterations / local steps: `K2 / I30 / L1`
- shared-only product-space DRS, block proximal metric, full consensus metric
- direct left-SE3 tangent normal equations
- points-p95 scene normalization and initial Jacobi camera scaling
- DABA trust policy with persistent radius and diagonal trust damping
- block curvature `0.00625`, curvature recovery, maximum curvature `64`
- outer acceleration disabled
- 12 threads per cluster
- the only algorithmic variant is `nesterov` versus `schur_pcg`

## Result

| Scene | Solver | Final SSE | Optimization s | Rejections | Median inner iterations | Maximum residual |
|---|---:|---:|---:|---:|---:|---:|
| Roman Forum | Nesterov | 4,800,609.51 | 21.12 | 5 | 22.25 | 0.2131 |
| Roman Forum | Schur-PCG | 4,399,807.62 | 17.72 | 6 | 13.75 | 0.00990 |
| Trafalgar | Nesterov | 20,574,048.01 | 41.23 | 11 | 9.75 | 0.8220 |
| Trafalgar | Schur-PCG | 16,442,461.57 | 34.32 | 8 | 8.75 | 0.00997 |

Schur-PCG reaches `0.9165x` the Nesterov SSE on Roman and `0.7992x` on
Trafalgar. Its optimization time is `0.8391x` and `0.8324x`, respectively.
This passes the two-scene K2 safety/usefulness gate without tuning. The next
test is the unchanged two-solver comparison at K24; C1/C5 remain disabled.