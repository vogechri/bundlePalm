# SfM_Init-Derived 1DSfM: Union Square Pilot

Date: 2026-07-27

## Status

The first external-validity pilot is complete end to end. It is a
**SfM_Init-derived 1DSfM** benchmark, not a DABA Table II reproduction.

## Provenance and Conversion

- public 1DSfM archive SHA-256:
  `2a608789cef64083177f68da9dfb725d791cf56ba51684a4a70304f14f0a7147`;
- SfM_Init revision:
  `fd012ef93462b8623e8d65fa0c6fa95b32270a3c`;
- Chatterjee-Govindu ICCV 2013 source archive SHA-256:
  `a2c7b925ee75b9089b08d24d6fc87fb42f505751cfb1f506cb8bf53b9c196993`;
- deterministic random seed: `20260727`;
- SIFT-to-Bundler conversion uses the corrected half-pixel convention from the
  March 2015 SfM_Init release;
- radial distortion initializes to zero because `coords.txt` supplies focal
  length and principal point but no distortion coefficients.

The source graph has 930 component cameras, 25,561 epipolar geometries,
279,963 tracks, and 1,005,282 feature observations. Every feature reference is
valid. Source hashes are stored in `source_inventory.json`.

## Initialization

Global rotations use an edge-order spanning-tree initialization, sparse
L1-IRLS, and the published Chatterjee-Govindu robust mean with a 5-degree scale.
All matrices are projected onto SO(3) before downstream geometry.

| Rotation diagnostic | Value |
|---|---:|
| Initial median relative residual | 12.456 degrees |
| Final median relative residual | 3.027 degrees |
| Final p95 relative residual | 56.988 degrees |
| Cameras / EG edges | 930 / 25,561 |

The SfM_Init greedy `k=6` track cover selects 730 tracks and creates 17,580
camera-point constraints. Deterministic 1DSfM voting uses 48 projections and
threshold 0.10:

| Cleanup diagnostic | Value |
|---|---:|
| Total constraints | 43,141 |
| Retained / rejected | 34,176 / 8,965 |
| Retained camera-camera | 17,021 |
| Retained camera-point | 17,155 |
| Connected components after cleanup | 1 |
| Component cameras retained | 930 / 930 |

The original weighted chordal translation objective uses camera-edge weight 1
and camera-point weight 0.726991. Ceres reaches its 1000-iteration ceiling with
median angular residual 4.537 degrees and p95 33.672 degrees. This is an
initialization, not a converged geometric reconstruction.

## Declared Triangulation Filters

Tracks are triangulated by multi-ray least squares and retained only if they
satisfy all of:

- positive depth in every retained view;
- maximum ray parallax at least 1 degree;
- triangulation condition number at most `1e6`;
- median initial reprojection error at most 20 px;
- maximum initial reprojection error at most 100 px;
- point degree at least 2;
- camera degree at least 10 after iterative bipartite cleanup.

No threshold was chosen to match DABA's counts. The resulting BAL file has:

| Cameras | Points | Observations | SHA-256 |
|---:|---:|---:|---|
| 768 | 82,614 | 265,464 | `3d5e366960498c546f56dcd09fe7998e9d48d022f777a95756a053bb3f1f35d5` |

Independent initial pixel evaluation gives mean 11.606 px, median 8.812 px,
p95 32.805 px, RMSE 16.442 px, and maximum 99.952 px. All values and state
entries are finite.

## First Solver Gate

Both solvers optimize the standard Snavely L2 pixel objective from the same BAL
file. Ceres uses iterative Schur/Jacobi for 40 iterations and 16 threads. DRS
uses the mature K10/30 protocol with one local nonlinear step, Jacobi scaling,
full `9x9` consensus, acceleration, and accepted-state restoration.

| Solver | Best/final SSE | Mean px | Median px | RMSE | Optimization/overall s |
|---|---:|---:|---:|---:|---:|
| Ceres | **4,745,369.79** | **2.2660** | **1.1998** | **4.2280** | **3.59** |
| DRS K10/30 | 5,022,896.67 | 2.3493 | 1.2000 | 4.3499 | 10.23 |

DRS reaches its best state at iteration 29. Its native/common evaluator relative
disagreement is `1.12e-8`. Ceres's saved-state disagreement is `5.57e-11`.

This is a successful correctness and generalization pilot, but not evidence of
a runtime advantage. DRS reaches near-Ceres quality while taking about 2.85x
the optimization time in this single-host implementation.

## Next Gate

Before expanding to Gendarmenmarkt, run Union Square K20/K30 and the partition
diagnostics. The conversion thresholds remain frozen. Any later change requires
a new benchmark version and content hash.
