# COLMAP Graph-Mode Schur Transfer Pilot

## Question

Does the fixed Stage-C global Schur correction remove the coherent weak
camera-graph deformation that additive K24/I30 DRS retained on Graham Hall's
low-mode 20-pixel perturbation?

## Fixed Quality Preset

The unchanged K24/I60+16 quality preset reaches `2,175,494.340` SSE versus
`2,170,906.996` for the historical additive K24/I30 baseline, a ratio of
`1.002113x`. Similarity alignment and projection onto the injected source-frame
mode give retained amplitudes `0.831443` and `0.459892`, respectively. The
quality preset therefore neither improves the pixel endpoint nor removes the
weak deformation on this pilot.

At the quality-preset handoff, Schur changes SSE from `2,175,505.507` to
`2,175,494.340` (`0.999994867x`). It accepts one correction and stops on the
`1e-3` progress rule.

## Isolated Old-State Restart

To separate the outer trajectory from Schur, the historical additive I30 NPZ
was loaded as an initial state. One required no-op outer iteration leaves its
handoff SSE unchanged to displayed precision. The same Schur phase then reaches
`2,170,561.207` SSE (`0.999840717x` handoff) and stops after one accepted
correction. Retained mode amplitude changes from `0.459892` to `0.459898`, a
ratio of `1.000013x`.

## Decision

Do not expand the fixed presets across the COLMAP graph-mode cohort. Exact
pixel-objective Schur polishing does not identify this nearly unobservable
coherent deformation. More corrections cannot supply the missing information;
removing the mode requires independently justified pose-graph, translation,
or other geometric evidence. This does not change the global Stage-C presets
or their 1DSfM conclusions.

Compact artifacts:

- `benchmark_results/colmap_graph_mode_schur_pilot_quality/*.jsonl`;
- `benchmark_results/colmap_graph_mode_old_state_schur/run/*.jsonl`;
- `serverTest/colmap_graph_mode_pilot_dataset.txt`.