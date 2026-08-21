# K24 DRS/Schur Direction-Alignment Diagnostic

Date: 2026-08-21
Status: direction diagnosis complete; coupled-consensus oracle next
Source checkpoint: `6e9d436`

This is the restart contract for the behavior-neutral diagnostic following the
integrated K24/I200 quality ceiling. It does not alter DRS proposals or apply
mid-run corrections.

## Question

At late iterations, is ordinary DRS merely taking a smaller version of the
coordinated global Schur direction, or is it moving in a materially different
camera direction/model geometry?

The diagnostic compares, before ordinary iterations I90, I120, I160, and I200:

- reference: the frozen low-damping global Schur camera tangent;
- candidate: the nominal accepted DRS consensus camera tangent for that
  iteration.

## Frozen Configuration

Use the integrated mature K24/I200 configuration and final one-correction policy
from `K24_I200_QUALITY_PLAN.md`. Alignment uses the actual correction settings:

- camera/landmark damping `0.005859375`;
- `bsr_low_memory`, Jacobi PCG, tolerance `1e-6`, maximum 500 iterations;
- no Schur-model clipping or any other behavior-changing option.

The first breadth pass showed that BAL3068 reaches tolerance at I90 but hits
the 500-iteration diagnostic ceiling at I120/I160/I200. Recover only that scene
with a diagnostic-only 5000-iteration ceiling. The terminal correction remains
capped at 500, so this recovery cannot alter solver behavior.

The diagnostic must reproduce the reference I200 trajectory, correction
acceptance, endpoint cameras, and endpoint points exactly.

## Cohort

- Roman Forum and Trafalgar: original K1/mature sentinels;
- Montreal Notre Dame and Yorkminster: corrected-I200 losses;
- Piazza del Popolo: recovery exhaustion at I188;
- BAL52 and BAL3068: stable cross-family controls.

No scene is used to select a scalar setting.

## Telemetry

At every available checkpoint report:

- all-camera and shared-camera cosine;
- diagonal-weighted all/shared cosine;
- DRS/Schur tangent norm ratio;
- translation, rotation, and intrinsics cosine/norm ratios;
- gradient-action ratio;
- camera-only damped model reduction split into shared and unique camera
  components, with a shared-DRS/shared-Schur ratio;
- similarity-gauge fraction and quotient-space alignment/model action;
- metric-invariant coherence of the reflected copy votes entering each shared
  camera consensus projection;
- Schur PCG termination, iterations, residual, and solve time.

Piazza is expected to have no I200 checkpoint because its ordinary trajectory
ends at I188. This is reported, not repaired.

## Gate

The run is valid only when:

- all diagnostic rows reproduce their reference trajectories and final states;
- every available diagnostic Schur solve terminates successfully below `1e-6`;
- the diagnostic field is present at every checkpoint reached by the reference;
- no clipping or correction is applied at the diagnostic checkpoints.

This experiment is descriptive. It must name a falsifiable mechanism before any
new solver behavior is implemented. Do not tune damping, checkpoint times, or
subspace scales from these seven scenes.

## Crash Recovery

Artifacts live under:

```text
benchmark_results/k24_schur_alignment_i200/
  diagnostic/
  summary.json
  report.md
```

Use `OVERWRITE=0`; keep logs, memory traces, NPZ states, and absolute manifests
untracked. Commit the runner/analyzer/plan before launch.

## Immediate Next Action

The seven-scene diagnostic passes exact trajectory/state neutrality and all
available Schur solves converge below `1e-6`. Its late shared-camera weighted
cosine is only `0.022--0.050` in median, and median shared DRS model gain is
only `0.000007--0.000645x` the shared Schur gain. Unique-camera motion has a
nonpositive camera model in every row. BAL3068 additionally has
`1e17--1e18` shared DRS/Schur norm ratios and effectively zero cosine.

Similarity-gauge drift is falsified. Median DRS gauge fraction is
`1.58e-9` at I90 and falls to `1.93e-13` at I200. Quotient projection leaves
the median shared cosine (`0.0503 -> 0.0505` at I90,
`0.0486 -> 0.0488` at I200) and norm ratio unchanged. BAL3068 also has gauge
fractions below `1.1e-15`, so its `1e17--1e18` norm anomaly is not similarity
gauge motion.

Reflected-copy cancellation is real but not the discriminating mechanism.
Median global coherence is only `0.0199--0.0310`, but across all 27 rows its
Spearman correlation is only `0.190` with shared Schur cosine and `0.223` with
shared model gain; excluding anomalous BAL3068, both correlations are slightly
negative. BAL52 is the direct counterexample: it has lower coherence than most
1DSfM rows while retaining much stronger Schur alignment and model gain.

The Roman/BAL52 sentinel closes the copy-level mechanism. Raw-copy global
cosines are near zero in both. Exact metric-contribution cosines are also near
zero, with signed action balance `0.0068--0.1091` on Roman and
`0.0191--0.0602` on BAL52. Median best-copy cosines overlap. Projection improves
both directions, so neither erased coherent copy signal nor incorrect copy
weighting explains why projected BAL52 aligns much better than Roman. See
`benchmark_results/k24_schur_contribution_alignment_sentinel/report.md`.

The projected-camera sentinel shows broad misalignment rather than a small
outlier set. Roman has median camera cosine `0.21--0.28`, only `71--75%`
positive cameras, and signed action balance `0.42--0.80`. BAL52 has median
cosine `0.39--0.59`, `81--100%` positive cameras, and action balance
`0.96--1.00`. See
`benchmark_results/k24_schur_camera_distribution_sentinel/report.md`.

The next experiment is a behavior-neutral cross-camera coupled-consensus
oracle. Using the same reflected copies and distributed Schur factors, compute
but do not apply the factorized coupled projection alongside the current
per-camera block projection on Roman and BAL52. The mechanism passes only if it
materially improves Roman's camera-wise/global alignment while preserving
BAL52. Only then implement the same coupled metric consistently in both local
proximal solves and consensus projection. Do not replace DRS with a Ceres or
global Schur step, and do not blindly switch the mature Nesterov backbone to
the historical Schur-PCG C3 preset.
