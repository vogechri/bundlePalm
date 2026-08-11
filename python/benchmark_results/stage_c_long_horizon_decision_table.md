# Stage-C Long-Horizon Decision Table

Date: 2026-08-11

## Maintained reference

The retained plain/C1/C5/C1+C5 publication matrix uses K24/I30 shared-only
block/full DRS, points-p95/Jacobi coordinates, one local nonlinear step,
Schur-PCG at `1e-2`, direct left-SE3 tangent assembly, persistent DABA trust,
and regularization recovery. The post-crash C4 carry-over gate is a separate
matched family with curvature recovery and Nesterov as its named baseline; its
ratios must not be mixed with the publication-matrix ratios below.

## Breadth results

| Method | 1DSfM all-15 SSE | W/T/L | Worst | Optimization | True worker CPU | Large-BAL SSE | BAL W/L |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain reference | 1.000000x | - | 1.000000x | 1.000000x | 1.000000x | 1.000000x | - |
| C1 safeguarded outer DRS | 0.850595x | 13/0/2 | 1.045075x | 1.541091x | 1.726258x | 0.991375x | 2/1 |
| C5 adaptive local depth | 0.952889x | 13/0/2 | 1.016898x | 1.176137x | 1.468372x | 0.999815x | 1/0/2 ties |
| C1+C5 | **0.786739x** | **15/0/0** | **0.947483x** | 1.832873x | 2.544050x | 0.996026x | 2/1 |

C1+C5 was the strongest normalized all-15 row in this retained artifact set.
Its old family-specific routing is superseded by the fresh global factorial.

The table above is retained pre-recovery artifact evidence. A fresh executable
post-recovery confirmation was run over all 15 1DSfM and all 29 BAL scenes with
the same K24/I30 Stage-C protocol. It is authoritative for the current source:

| Method | 1DSfM/plain | W/T/L | Optimization | BAL/plain | W/T/L | Optimization |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 0.865147x | 13/0/2 | 1.557830x | 0.983780x | 28/0/1 | 1.820914x |
| C5 | 1.015807x | 5/0/10 | 1.207142x | 1.000057x | 4/15/10 | 1.158716x |
| C1+C5 | 0.870061x | 13/0/2 | 1.801873x | 0.983638x | 28/0/1 | 2.030313x |

The fresh matrix now contains all 176 DRS rows for the global
plain/C1/C5/C1+C5 factorial. C1 is the accepted first rung (`0.865147x` plain
SSE on 1DSfM, `0.983780x` on BAL). Current C5 alone is `1.015807x` and
`1.000057x`; C1+C5 relative to C1 is `1.005680x` and `0.999855x`. Retain both
innovations and the combined architecture. C5's common thresholds/work policy
are the next global tuning target; no scene-specific settings are allowed. The
complete report is
`stage_c_final_all15_all29_k24_i30/comparison_to_ceres_and_plain_drs.md`.

Matched 2x2 diagnostics classify all three losses. On Roman, C1 and C5 lose
individually (`1.098217x`, `1.055628x`) but interact beneficially, leaving
C1+C5 at `1.047329x`. On Yorkminster, both lose (`1.030953x`, `1.063671x`) and
interact adversely, amplifying C1+C5 to `1.184323x`. On BAL135, C5 is inactive
and trajectory-neutral; the `1.002734x` loss is entirely C1. A prefix-identical
Yorkminster continuation with safeguard annealing fixed at I30 reduces C1+C5
to `1.029215x` by I60, while C1 alone becomes a `0.908028x` win. See
`stage_c_loss_factorials_k24_i30_report.md`. These tails guide global tuning;
they are not scene vetoes and do not reject C1, C5, or their composition.

The completed global tuning ladder selects delayed C5 activation at I5 with
high/low `0.35/0.20`, window/dwell `3/3`, and depth 2. Tuned C1+C5/C1 reaches
`0.980053x` all-15 1DSfM and `0.999486x` all-29 BAL. Tuned C1+C5/plain reaches
`0.847890x` and `0.983275x`, W/T/L `14/0/1` and `28/0/1`. This common policy is
the promoted Stage-C stack. See `stage_c_tuned_c5_final_report.md`.

## Component decisions

### C1: Retain

C1 survives the I30 all-15 gate and is safe on the tested large-BAL cohort.
Retain it as the cumulative quality base and isolated ablation. Describe it as a
quality mechanism, not a practical speedup.

### C2: Isolated only

The C1 x C2 sentinel factorial is scene-dependent. Roman has positive synergy,
but normalized C1 destroys the strong raw-coordinate Trafalgar C1 gain. Do not
insert C2 automatically after C1; report coordinate equilibration as an isolated
study.

### C3: Artifact-backed, execution path unavailable

Factorized C3 has strong historical 1DSfM I5 evidence and negative large-BAL
transfer. After the 2026-08-11 editor rollback, standalone coupled/factorized
math and tests remain, but the full coordinator execution path lacks a complete
recoverable source snapshot. Do not launch new C3 experiments without an
explicit reconstruction task.

### C4: Retain as finite-work factorial

The maintained post-crash all-15 K24/I30 gate holds coordinates, damping,
preconditioning, trust, local depth, and outer acceleration fixed. PCG reaches
`0.985199x` summed SSE, `1.005578x` geometric-mean per-scene SSE, and
`0.931664x` geometric-mean optimization time relative to Nesterov. It wins SSE
on `8/15` scenes and runtime on `12/15`, but regresses NYC Library to
`1.195314x` and Piccadilly to `1.091320x` SSE. Retain Nesterov as the named
unchanged default trajectory and PCG as an independently switchable
speed-oriented C4 factorial level. Do not tune tolerances or select by scene.

### C5: Retain and tune globally

Current C5 activates selectively and adds moderate work. In the fresh global
factorial, C5/plain is `1.015807x` on 1DSfM and `1.000057x` on BAL; C1+C5/C1 is
`1.005680x` and `0.999855x`. Retain C5 and C1+C5 in the final architecture.
Tune one common threshold/work policy against both families to improve the
incremental C5-over-C1 row; do not disable or configure it by scene.

## Manuscript matrix

The current defensible long-horizon table is:

1. Plain normalized block/PCG reference.
2. Plain + C1.
3. Plain + C5.
4. Plain + C1 + C5.
5. Separate C2 coordinate factorial on sentinels.
6. Separate all-15 C4 solver/work factorial with Nesterov as its reference.
7. Historical C3 quality/cost table, explicitly marked as artifact-backed until
   its execution path is reconstructed.

## Next missing evidence

No further C1/C2/C4/C5 parameter sweep is justified. The next safe work is
manuscript/report consolidation and a reproducibility manifest containing exact
artifact paths, configurations, evaluator, worker CPU/RSS accounting, and the
2026-08-11 source-recovery boundary. A new experiment should be launched only
if that consolidation identifies a missing matched row in the maintained
C1/C5 matrix.

## Source reports

- `c1_themelis_all15_large_bal_i30_report.md`
- `c1_c2_sentinel_factorial_i30_report.md`
- `c1_c2_c4_sentinel_factorial_i30_report.md`
- `c1_c5_all15_large_bal_i30_report.md`
- `c4_k24_postcrash_all15_i30/DECISION.md`

## Reproducibility manifest

Machine-readable per-scene artifacts, configurations, endpoint metrics,
optimization time, proximal-oracle counts, worker CPU, worker RSS, repository
revision, and the source-recovery boundary are recorded in:

- `stage_c_reproducibility_manifest.json`

Regenerate and validate all artifact/config/timing coverage with:

```bash
serverTest/.venv/bin/python \
   serverTest/build_stage_c_reproducibility_manifest.py
```

The manifest intentionally records `working_tree_clean: false`; the research
workspace contains uncommitted recovery and report changes. The commit hash is
provenance for the checked-out base, not a claim that the manifest artifacts
were generated by an unmodified commit.
