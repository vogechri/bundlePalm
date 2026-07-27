# DRS Camera Consensus Projection Metric Ablation

Date: 2026-07-27

## Protocol

- dataset: Ladybug-1723
- 10 landmark-owned clusters
- 90 outer iterations
- one local nonlinear step
- standard Snavely pixel objective and independent evaluator
- `landmark_scalable` partition with the supplied weak-camera settings
- Jacobi geometric-mean camera scaling
- identical local nonlinear prox, DRE/primal relative safeguard, acceleration,
  rollback, and initial `Be = 5e-5`

Only the metric used by the reflected-camera consensus projection and matching
DRE terms changes. The local proximal solve continues to return and use its
full camera metric. Initial scaling is also computed from the same full metric.
This isolates curvature-aware reconciliation rather than the entire
variable-metric local method.

## Results

| Projection metric | Best iteration | Best pixel SSE | Mean px | SSE at 30 | SSE at 60 | Hard rejections | Fallback trials | Final hard-rejection `Be` | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Arithmetic identity | -1 | 124,050,155 | 3.8881 | 124,050,154 | 124,050,154 | 19 | 48 | 0.5 | 63.63 |
| Scalar per camera copy | 55 | 866,306 | 0.7940 | 1,055,152 | 866,306 | 14 | 47 | 0.5 | 56.93 |
| Diagonal 9-parameter | 89 | 805,805 | 0.7778 | 833,991 | 821,541 | 8 | 37 | 0.0128 | 57.79 |
| Full regularized 9x9 block | 89 | **764,872** | **0.7606** | **772,097** | **766,657** | **7** | **32** | **0.0064** | **55.33** |

Relative to full blocks:

- diagonal final SSE is 5.35% higher;
- scalar final SSE is 13.26% higher;
- arithmetic never improves the independently saved best state.

All four saved physical states independently reproduce their recorded standard
pixel SSE exactly.

## Interpretation

The ordering is monotone with directional metric information:

$$
\text{full block} < \text{diagonal} < \text{scalar} \ll \text{arithmetic}
$$

in final pixel SSE, while safeguard burden follows the reverse order. Full
blocks reach lower cost, require fewer fallback and hard-rejection trials, and
need only half the final hard-rejection `Be` of diagonal consensus. Scalar
consensus saturates `Be` at its global ceiling and stops improving after
iteration 55.

Arithmetic consensus is numerically unstable, not merely slow. It generates
nonfinite trial states, 19 hard rollbacks, 48 fallback trials, and saturates
`Be = 0.5`. The result file reports the independently saved initial best state;
it must not be interpreted as a stable endpoint. The final current trial state
is catastrophic even though best-state output remains finite.

This supports the BA-specific full block-metric consensus contribution. A
scalar penalty loses direction-dependent observability, diagonal consensus
loses camera-coordinate coupling, and arithmetic consensus treats strong and
nearly singular local camera copies equally.

The ablation is a projection-metric test, not yet a complete comparison of four
internally consistent proximal algorithms. A paper claim requires repeating it
on the fixed five-scene cohort and reporting the extra metric payload bytes.

## Artifacts

- `full.jsonl`, `full.npz`, `full.log`
- `diagonal.jsonl`, `diagonal.npz`, `diagonal.log`
- `scalar.jsonl`, `scalar.npz`, `scalar.log`
- `arithmetic.jsonl`, `arithmetic.npz`, `arithmetic.log`