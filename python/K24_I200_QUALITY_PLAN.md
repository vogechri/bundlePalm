# Integrated K24/I200 Quality Gate

Date: 2026-08-21
Status: complete; breadth aggregate-positive but not promoted
Source checkpoint: `0b5b401`

This is the restart contract for the remaining apples-to-apples longer-horizon
quality gate. It evaluates the integrated mature K24 method, not the historical
pre-integration I200 artifact and not the failed plain-resolvent I1000 preset.

## Fixed Method

Both arms use one global configuration:

- K24, one thread per cluster;
- direct left-SE3 tangent assembly and exact tangent metric consistency;
- shared-only block/full product-space DRS;
- points-p95 normalization, Jacobi camera scaling, and landmark preconditioning;
- local Nesterov, maximum 300, tolerance `1e-2`, enhanced through I30;
- persistent DRS trust, initial radius 10, maximum radius `1e4`;
- block regularization `1e-4`, curvature `0.4`, curvature recovery ceiling 64;
- curvature decay after I5 at ratio `0.5`;
- C1 Themelis acceleration with line-search grid `0,1` and restart 3;
- no C5, proposal damping, bootstrap, mid correction, or repeated correction;
- one terminal distributed Schur correction with damping `0.005859375`,
  geometric fallback, `bsr_low_memory`, Jacobi PCG, and tolerance `1e-6`.

## Sentinel Cohort

Selection uses only Roman Forum, Trafalgar, BAL52, and BAL3068. BAL49 remains
smoke-only and is not used.

## Matched Arms

Both processes request an I200 horizon so all horizon-dependent safeguards are
identical:

1. `i90_horizon200`: stop after 90 completed ordinary DRS iterations, then
   apply the one terminal correction;
2. `i200`: complete 200 ordinary DRS iterations, then apply the same correction.

The first 90 trajectory rows must match exactly in SSE, refined candidate SSE,
rejection decisions/counts, and proximal-oracle counts. This is the causal
check that failed when an I60 process was compared with an I90 process.

## Promotion Gate

Advance unchanged to all 15 1DSfM and all 29 BAL only when:

- every sentinel completes its requested ordinary DRS iterations;
- the two arms have exact matching I90 prefixes;
- no accepted Schur solve is nonconverged or has residual at least `1e-6`;
- rejected correction attempts preserve their prestates;
- corrected-I200/corrected-I90 geometric SSE is below one on 1DSfM and no
  greater than one on BAL;
- summed SSE does not materially contradict geometric SSE;
- no scene-specific setting or duration change is introduced.

Report raw-I200/raw-I90, corrected-I200/corrected-I90,
corrected-I200/raw-I200, Ceres ratios, completion, damping, time, and peak RSS.

## Crash Recovery

Artifacts live under:

```text
benchmark_results/k24_i200_terminal_correction_sentinel/
  i90_horizon200/
  i200/
  summary.json
  report.md
```

The runner uses `OVERWRITE=0`; one completed status and JSONL row is required
before a case is skipped. Commit runner/analyzer/plan before launching. Keep
logs, memory traces, NPZ states, and absolute-path manifests untracked.

## Phase Status

| Phase | Status | Artifact |
|---|---|---|
| Four-scene sentinel | passed | `k24_i200_terminal_correction_sentinel/report.md` |
| All-15/all-29 breadth | complete, failure-inclusive | `k24_i200_terminal_correction_breadth/report.md` |
| Publication decision | diagnostic ceiling only | Piazza incomplete at I188 |

## Immediate Next Action

Do not run another duration, curvature-ceiling, or correction-damping sweep.
The breadth result reaches corrected-I200/corrected-I90 `0.959795x` on 1DSfM
and `0.996820x` on BAL, but Piazza recovery-exhausts at I188. Retain this as a
failure-inclusive quality ceiling. The next justified gate is behavior-neutral
DRS-versus-Schur direction alignment telemetry at I90/I120/I160/I200.
