# 1DSfM Provenance and Readiness Audit

## Archive

The official numerical archive was downloaded from:

`http://landmark.cs.cornell.edu/projects/1dsfm/datasets.tar.gz`

- size: 682,117,483 bytes;
- SHA-256: `2a608789cef64083177f68da9dfb725d791cf56ba51684a4a70304f14f0a7147`;
- gzip integrity check: passed;
- archive date lineage: the project page describes corrections through March
  2015, including corrected principal points and coordinate conventions.

Only the six DABA Table II target directories were extracted.

## Identity Gate

The archive supplies `gt_bundle.out`, not the exact preprocessed BAL files used
by DABA. Its raw Bundler reconstructions do not match DABA Table II:

| Dataset | Raw Bundler cameras / points | DABA cameras / points / observations |
|---|---:|---:|
| Gendarmenmarkt | 1,463 / 158,513 | 706 / 93,672 / 364,029 |
| Piccadilly | 7,351 / 341,219 | 2,289 / 209,504 / 999,878 |
| Roman Forum | 2,364 / 400,382 | 1,063 / 265,047 / 1,292,756 |
| Trafalgar | 15,685 / 597,770 | 5,032 / 388,956 / 1,826,071 |
| Union Square | 5,961 / 57,311 | 796 / 46,066 / 230,811 |
| Vienna Cathedral | 6,288 / 526,139 | 836 / 265,553 / 1,333,280 |

Filtering the Bundler reconstruction by `cc.txt`, nonzero focal length, and
minimum track degree does not recover the published triples. The reason is
structural: `gt_bundle.out` is an independent comparison reconstruction and
does not contain the same cameras or tracks as `tracks.txt` and `coords.txt`.

The dense source graph also does not map to Table II through a single minimum
track-degree rule. For example, Roman Forum contains 595,896 source tracks;
requiring at least three component views leaves 199,214, while Table II reports
265,047 points.

## Missing Initialization

The intended dense inputs provide:

- `cc.txt`: component camera identifiers;
- `EGs.txt`: relative rotations and translation directions;
- `coords.txt`: focal length, principal point, and image features;
- `tracks.txt`: image-feature tracks.

They do not provide initialized global poses and triangulated 3D points for the
DABA Table II graph. Producing a valid BA problem requires running the historical
1DSfM pipeline for rotation averaging, translation estimation/outlier removal,
and triangulation. DABA's public repository contains neither that preparation
pipeline nor its resulting Table II BAL files; its README documents only BAL
downloads. Public exact-count searches did not locate the missing artifacts.

DABA's included `BundlerDatasetToBALDataset` function is not a solution to this
identity gap. It converts `gt_bundle.out` and re-fits per-camera focal/distortion
parameters by linear least squares, so it both uses the wrong graph and changes
the supplied intrinsics.

## Decision

No optimization run from this archive should be labeled a DABA Table II
reproduction yet. The camera/point/observation identity gate and published
initial trivial-loss gate both fail before optimization.

Two defensible paths remain:

1. Obtain the authors' preprocessed BAL files or exact filtering and
   initialization procedure. This is the shortest route to a Table II
   comparison.
2. Port the SfM_Init Python 2 pipeline and triangulation to current tooling,
   then publish the resulting data as a distinct raw-1DSfM benchmark unless its
   counts and initial objectives exactly match DABA.

The chosen near-term paper path is option 2, named **SfM_Init-derived 1DSfM**.
Option 1 remains an opportunistic exact-reproduction gate and does not block the
external-validity benchmark.

The archive is retained locally so either path can proceed without another
download.

## Reproducible Cleanup Route

The original [SfM_Init](https://github.com/wilsonkl/SfM_Init) implementation is
public and provides a better starting point than inventing cleanup rules. Its
source is pinned locally under `third_party/SfM_Init` at commit
`fd012ef93462b8623e8d65fa0c6fa95b32270a3c`. Its
`scripts/eccv_demo.py` performs:

1. L1-IRLS global rotation averaging from `EGs.txt`, restricted by `cc.txt`;
2. construction of camera-camera translation directions;
3. addition of camera-to-point constraints from `tracks.txt` and `coords.txt`;
4. 1DSfM outlier voting with 48 projected one-dimensional problems and default
  rejection threshold `0.10`;
5. robust chordal translation estimation with Ceres.

This pipeline produces global camera rotations and positions plus a cleaned
translation graph. It does not triangulate points, write a complete
reconstruction from those estimates, or run bundle adjustment. It is also
Python 2 code and wraps a MATLAB rotation solver, so a direct execution is not
a maintainable route.

A correctness-first modern port should preserve the original graph and cleanup
semantics, while replacing only obsolete infrastructure:

1. port the file parsers and 1DSfM voting code to Python 3;
2. reproduce the original rotation result first, then replace the MATLAB
  wrapper with a documented robust rotation-averaging implementation;
3. preserve the corrected SIFT-to-centered-Bundler coordinate conversion from
  the March 2015 release;
4. reproduce camera positions and report retained edge counts and residuals;
5. triangulate tracks with deterministic multi-view DLT, cheirality checks,
  minimum parallax, and reprojection filtering;
6. remove cameras and points iteratively only when they fall below declared
  degree thresholds;
7. write BAL, then require an independent initial pixel evaluation and a
  BAL-to-state round trip before optimization.

The resulting benchmark should be named `SfM_Init-derived 1DSfM`. Its cleanup
thresholds, retained counts, coordinate conventions, and initial pixel-error
distribution must be published. It becomes comparable across our own methods,
but not one-to-one with DABA unless the Table II counts and initial ray metrics
also match.

The official [DeepLM](https://github.com/hjwdzh/DeepLM) repository is available
and matches DABA reference [19] (Huang, Huang, and Sun, CVPR 2021). Its bundle
adjustment example downloads and loads standard BAL files; it does not document
1DSfM conversion or provide the missing DABA Table II preprocessing artifacts.
MegBA likewise documents BAL inputs only.
