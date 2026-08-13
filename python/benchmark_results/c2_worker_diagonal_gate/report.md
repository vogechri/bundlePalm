# C2 Worker-Derived Diagonal Jacobi Gate

## Requested Mechanism

The default-off `CAMERA_SCALING=worker_diagonal_jacobi_initial` mode follows the
requested sequence:

1. run the initial unscaled local proximal solve;
2. collect workers' packed full `9x9` metrics;
3. aggregate copy blocks per camera into `U_all`;
4. take `sqrt(max(diag(U_all), 1e-6 * max(diag(U_all))))`;
5. geometric-mean normalize the scales;
6. transform solved copies, centers, consensus, and worker camera state.

This uses ordinary diagonal transport and leaves full `9x9` metric consensus
unchanged.

## BAL49 Smoke

The first candidate is accepted at `140,819.468` SSE. This is approximately
`5.4%` below both dense-transform diagnostics (`148,836.275` worker-derived and
`148,837.566` raw-Jacobian). The scale ratio is `1.108e4`.

## Frozen I3 Gate

| Scene | Worker diagonal / current Jacobi I3 SSE | Rejections | Linear work |
|---|---:|---:|---|
| Roman Forum | 0.9999999999 | 0/3 | unchanged |
| Trafalgar | 1.0000000000 | 0/3 | unchanged |
| BAL1778 | 1.000083083 | 0/3 | median/max unchanged |

Roman and Trafalgar trajectory rows agree with current `jacobi_initial` to about
`1e-9`. BAL1778 is `0.0083%` worse and has identical local linear-iteration
counts at every step.

## Source Clarification

The historical `client_acc.py` source has two distinct mechanisms:

- diagonal Jacobi-style `GetPcgScalingDiag(...)` modes;
- `block_jacobi_gmean`, which explicitly returns diagonal scales of one and
  installs the dense `get_camera_block_transforms(U_all)` map.

Thus the requested worker-derived diagonal variant is now tested faithfully,
but the historical mode named `block_jacobi_gmean` was the dense block map, not
`diag(U_all)^{-1/2}`.

## Decision

Worker-derived diagonal Jacobi is redundant with the current initial global
Jacobi scaling under direct left-SE3 DRS. Retain it default-off as a provenance
and invariance diagnostic; do not run longer horizons or tune the floor. The
old improvement from diagonal preconditioning should be interpreted relative
to unscaled or weaker historical baselines, not as evidence that recomputing
the same diagonal after one worker initialization improves current Jacobi DRS.
