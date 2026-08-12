# I60 Performance and Tail Diagnostics

Date: 2026-08-12

## Current Worker Profile

Current-source K24/I10 timing with local-solve metrics enabled was collected on
Roman and Trafalgar under the promoted DRS controls.

| Scene | Local critical s | Jacobian s | Assembly s | Nesterov s | Cost s |
|---|---:|---:|---:|---:|---:|
| Roman | 1.192 | 0.296 | 0.609 | 0.253 | 0.011 |
| Trafalgar | 2.908 | 0.428 | 1.341 | 1.006 | 0.025 |

Assembly is now larger than Nesterov on both profiles. Within Nesterov,
repeated cross-Hessian actions dominate: Roman spends `0.096s` in $W^T$ and
`0.067s` in $W$ iteration actions; Trafalgar spends `0.319s` and `0.270s`.

Direct-tangent mode currently evaluates each projection once for the raw DRS
metric and again for tangent normal equations. A default-off dual-coordinate
prototype assembled both systems from one physical Jacobian evaluation. It was
numerically close through I10 (maximum candidate relative delta below
`7e-9`) and reduced local critical time by `7.0%` Roman / `4.6%` Trafalgar.
However, all-15 I60+16 changed every Schur decision sequence: total work fell
`6.3%`, but geometric SSE regressed `0.584%`, summed SSE `0.239%`, and the worst
scene regressed `4.40%`. The experimental code was removed. A one-pass dual
assembly is not exact enough for the accepted nonlinear trajectory.

A second default-off experiment traversed edges by conflict-free camera and
landmark adjacency while preserving each output block's edge order. It is
bitwise trajectory-exact through I10, but worsens optimization time
`3.3%/2.4%` on Roman/Trafalgar. In particular, iterative $W^T$ time rises
`14%/42%`; camera-sorted edge storage gives better camera-vector locality than
landmark adjacency. The code was removed.

## I60 Correction Frontier

The unchanged all-15 I60 handoff was extended from 16 to 20 corrections.

| Allocation | Total optimization s | SSE / base-I200 | SSE / Ceres | Summed SSE / Ceres |
|---|---:|---:|---:|---:|
| I60 + 16 | 380.685 | 0.834064 | 1.215246 | 1.109252 |
| I60 + 20 | 430.320 | 0.813073 | 1.184661 | 1.092148 |

Corrections 17--20 improve I60+16 by `0.974832x` geometrically and
`0.984581x` summed SSE at `1.130382x` time. They improve 12 scenes and tie the
three that had already reached the progress stop. I60+20 is a valid
higher-budget quality-ceiling point, not a replacement for the I60+16 named
quality preset.

## Madrid and Tower

Similarity-aligned I60+16 geometry was compared with base-I200 and Ceres.

- Madrid: low graph modes do not explain the gap. The first 8 camera-center
  modes contain only `0.045%` of the displacement energy versus base and
  `0.088%` versus Ceres; the first 32 contain `24.4%`/`7.5%`. Rotation P95 is
  `7.70` degrees versus base and `17.35` degrees versus Ceres. Corrections
  17--20 improve Madrid only `0.93%`.
- Tower: the gap has a larger low-mode component. The first 8 modes contain
  `12.8%` versus base and `5.3%` versus Ceres. Rotation P95 is `6.33`/`9.40`
  degrees. Corrections 17--20 improve Tower `11.67%`, so Tower remains strongly
  under-polished at cap 16.

Both reconstructions have heavy-tailed point coordinates after similarity
alignment: point RMS and camera/point hybrid swaps are dominated by extreme
outliers and are not useful selectors, while P95 geometry remains finite.
Madrid is primarily a different non-low-mode camera/point basin; Tower is a mix
of low-mode basin error and remaining Schur descent.

Tower-only continuation subsequently stops naturally at correction 53 with
`0.701186x` base-I200 and `1.316613x` Ceres. Thus its cap-16 diagnosis was
correct, but the final residual gap persists after the frozen Schur progress
criterion and is a basin issue.

## Decision

1. Keep I60+16 as the named quality preset and retain I60+20 as a higher-budget
   quality-ceiling point.
2. Do not promote the dual-coordinate local assembly; exactness must be by
   construction, not merely close at I10.
3. For worker speed, next target reusable assembly/data structures or exact
   $W^T/W$ kernel acceleration, not another algebraically reconstructed metric.
4. For quality, continue Tower with deeper polishing if a ceiling is needed;
   treat Madrid as a separate non-low-mode basin problem rather than tuning the
   graph prior globally.
