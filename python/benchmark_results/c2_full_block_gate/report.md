# C2 Full Block-Coordinate Gate

## Mechanism

The current DRS coordinator now supports a default-off
`CAMERA_SCALING=block_jacobi_initial` mode. It accumulates each camera's full
initial `9x9` Gauss--Newton block, floors eigenvalues at `1e-10` of the block
maximum, and uses a globally determinant-normalized inverse square root as the
coordinate map. The map is transported through the worker's existing full
camera-transform payload.

Coordinator physical/scaled state conversion, left-SE3 worker tangent
conversion, metric diagnostics, and result scaling summaries are matrix-aware.
The default diagonal Jacobi path is unchanged.

## Smoke

BAL49 K2/I1 completes with finite independently evaluated SSE and finite
coordinate-map singular-value metadata. Unit tests verify full-map round trips,
broadcasting over product-space copies, and whitening of synthetic SPD camera
blocks.

## Frozen I3 Screen

Only camera scaling changes from the matched Jacobi controls.

| Scene | Full block / Jacobi I3 SSE | Rejected iterations | Scale ratio |
|---|---:|---:|---:|
| Roman Forum | 3.864440 | 3/3 | 1.597e14 |
| Trafalgar | 3.018749 | 3/3 | 1.281e14 |
| BAL1778 | 92.449861 | 3/3 | 3.126e13 |

Every full-block candidate is independently rejected. Roman and Trafalgar
candidate SSE reaches `3.60e22` and `1.64e25` on I1. BAL1778 candidates are
also above the accepted initial state on every iteration. Safeguards correctly
preserve the initial physical state; there are no nonfinite states or worker
failures.

Full blocks reduce BAL1778's first local linear maximum from 78 to 62, but the
resulting trust/proximal geometry is catastrophically incompatible with the
current DRS resolvent. This is not a generic linear-conditioning win.

## Decision

Reject full block-coordinate C2 and close idea 2. Do not tune eigenvalue floors,
normalization, trust radii, or proximal curvature from these failures. Retain
both Ruiz and full-block modes default-off as coordinate-invariance and
conditioning diagnostics. The next queued mechanism is idea 1, refreshed
cross-camera C3.
