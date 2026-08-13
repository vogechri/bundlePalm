# C2 Worker-Metric Block Preconditioner Gate

## Correction To The Earlier C2 Test

The older successful `client_acc.py` block preconditioner did not form its
transform from the raw global initial pixel Jacobian. It first ran one local
proximal solve per worker, collected each worker's full `9x9` camera metric,
aggregated those copy metrics into `U_all`, and formed `U_all^{-1/2}` with a
relative eigenvalue floor of `1e-6`.

The current coordinator now reproduces that sequence in the default-off
`CAMERA_SCALING=worker_block_jacobi_initial` mode:

1. initialize workers without a camera transform;
2. run the existing first local proximal solve;
3. collect the packed full worker metric blocks;
4. sum blocks per global camera;
5. compute the determinant-normalized inverse square root with floor `1e-6`;
6. transform solved local copies, centers, and consensus;
7. install the map and transformed cameras in every worker.

This differs from `block_jacobi_initial`, which uses raw initial physical pixel
Jacobian blocks before a local solve.

## Smoke

BAL49 K2/I1 completes and accepts its first candidate. Worker-derived SSE is
`148,836.275`, versus `148,837.566` for the raw-Jacobian transform. The reported
coordinate-scale ratio falls from `7.16e4` to `1.74e4`.

## Current Left-SE3 I3 Gate

| Scene | Worker metric / Jacobi I3 SSE | Rejections | Worker/raw-transform SSE |
|---|---:|---:|---:|
| Roman Forum | 0.973334 | 0/3 | 0.251869 |
| Trafalgar | 1.395777 | 1/3 | 0.462369 |
| BAL1778 | 87.637697 | 2/3 | 0.947948 |

Roman accepts all steps and improves Jacobi by `2.67%`. Trafalgar's first
candidate reaches `5.09e12` SSE and is rejected; later accepted steps do not
catch Jacobi. BAL1778 rejects its first two steps and accepts I3 at `486.1M`,
still `87.64x` its Jacobi I3 state.

## Interpretation

The user's recollection is correct: worker-metric block preconditioning is a
real and materially better mechanism than raw-Jacobian whitening. The remaining
transfer boundary is the current direct left-SE3 formulation. The old path used
camera-coordinate worker metrics directly; current workers rebuild direct
tangent normal equations and congruence-transform the proximal metric. Applying
the old camera-coordinate transform changes that trust/proximal geometry and is
not uniformly safe.

## Decision

Retain the faithful worker-metric mode default-off and record Roman as positive
evidence. Do not promote or broaden under direct left-SE3 because Trafalgar and
BAL1778 fail the fixed gate. Idea 2 is closed for the current production
formulation, but the old result is not invalidated: a future additive-coordinate
reproduction or a tangent-space-only block preconditioner is a distinct
mechanism and must not be represented by the rejected raw-Jacobian experiment.
