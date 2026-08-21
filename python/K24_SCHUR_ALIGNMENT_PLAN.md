# K24 DRS/Schur Direction-Alignment Diagnostic

Date: 2026-08-21
Status: ready to run
Source checkpoint: `23718d1`

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

Run the committed seven-scene diagnostic and compare it exactly with the
existing I200 breadth artifacts.
