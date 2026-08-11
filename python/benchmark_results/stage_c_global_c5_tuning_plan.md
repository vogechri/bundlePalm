# Global C5 Tuning Plan After C1

Date: 2026-08-11

## Fixed architecture

The final architecture contains both C1 safeguarded acceleration and C5 adaptive
local work. Every setting is global across scenes and benchmark families. There
is no scene classifier, scene-specific preset, or endpoint portfolio.

The cumulative evaluation order is:

1. plain DRS;
2. plain + C1;
3. plain + C1 + C5;
4. later C2/C4 factorial checks remain separate.

C5-only remains in every evaluation matrix to identify its main effect, but C5
is tuned as the incremental component after the accepted C1 rung.

## Current common setting

- maximum local depth: 2;
- high defect threshold: 0.30;
- low defect threshold: 0.15;
- rolling window: 3;
- dwell: 3;
- nominal depth: 1.

Fresh K24/I30 C1+C5/C1 geometric SSE is `1.005680x` on all-15 1DSfM and
`0.999855x` on all-29 BAL. The immediate target is a robust incremental gain on
both families without weakening safeguards.

## One-factor tuning order

1. **High threshold:** test `0.35`, `0.40`, `0.50` against `0.30`. This directly
   controls entry into depth two and is the first global selectivity knob.
2. **Low threshold / hysteresis:** after freezing the high threshold, test low
   thresholds `0.10`, `0.15`, `0.20` while maintaining `low < high`.
3. **Rolling window:** test `3` versus `5` after freezing thresholds.
4. **Dwell:** test `3` versus `5` after freezing the window.
5. **Maximum depth:** only then test depth `3`; do not mix it into threshold
   selection because it changes both activation and work magnitude.

Do not sweep C1 safeguard, C5 threshold, and inner-solver parameters together.
Each accepted row changes one mechanism from the current cumulative control.

## Evaluation ladder

For each one-factor candidate:

1. six development 1DSfM scenes plus a fixed BAL sentinel cohort;
2. freeze the setting;
3. nine held-out 1DSfM scenes;
4. all-15 1DSfM and all-29 BAL confirmation;
5. compare plain, C1, C5, and C1+C5 at the final setting.

Held-out results cannot be used to choose a second setting. If a candidate is
changed after held-out evaluation, restart the held-out gate.

## Acceptance criteria

A candidate advances only if:

- C1+C5/C1 geometric SSE is below one on both families at the final gate;
- summed SSE does not materially contradict the geometric result;
- no recovery exhaustion or nonfinite state occurs;
- delivered PCG residual remains within the configured `1e-2` gate;
- optimization, oracle, transport, and adaptive-depth overhead are reported;
- the worst tail is reported and bounded, but no scene-specific setting is
  introduced.

The objective is a single cross-family Pareto improvement. A scene tail guides
which global mechanism to inspect; it does not select the deployed parameters.

## Completed result

The scalar threshold/window/dwell/depth sweeps did not pass both development
families. Delayed activation did. The frozen policy is:

- start iteration: 5;
- high/low thresholds: `0.35/0.20`;
- rolling window / dwell: `3/3`;
- maximum depth: 2.

It was selected on the six-scene 1DSfM plus five-scene BAL development gate,
then run unchanged on nine held-out 1DSfM scenes and all 29 BAL scenes.
Final C1+C5/C1 geometric SSE is `0.980053x` on all-15 1DSfM and `0.999486x`
on all-29 BAL. C1+C5/plain is `0.847890x` and `0.983275x`, with W/T/L
`14/0/1` and `28/0/1`. Promote the common tuned C1+C5 stack; keep C1 and C5
independently switchable for ablations.
