# Final-only fixed-camera landmark polishing

## Question

Does one fixed-camera landmark refinement at the end of DRS provide more value
for short runs, with gain decaying approximately as the inverse outer-iteration
budget?

## Protocol

- Corrected ten-scene K30 baseline.
- Budgets: 1, 2, 3, 5, 10, 20, 30, 60, and 90 outer iterations.
- One landmark refinement step applied only once, after DRS terminates, to the
  best unrefined consensus cameras and their associated landmark state.
- Every budget has an independently run no-polishing control with the same
  iteration horizon. This matters because the relative safeguard schedule
  depends on the requested horizon.
- Every polishing trajectory exactly matches its horizon-matched control on all
  governing fields.
- All 90 polished saved states independently reproduce their recorded pixel
  SSE.

## Results

| Iterations | Geomean SSE gain | Median scene gain | Wins | Median added time | Median time ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 57.43% | 56.26% | 10/10 | 0.370 s | 1.071 |
| 2 | 52.17% | 49.62% | 10/10 | 0.140 s | 1.027 |
| 3 | 39.28% | 35.47% | 10/10 | 0.215 s | 1.042 |
| 5 | 26.28% | 19.05% | 10/10 | 0.153 s | 1.026 |
| 10 | 5.24% | 3.80% | 10/10 | 0.250 s | 1.045 |
| 20 | 1.56% | 0.43% | 10/10 | 0.284 s | 1.029 |
| 30 | 1.08% | 0.22% | 10/10 | 0.237 s | 1.021 |
| 60 | 0.62% | 0.07% | 10/10 | 0.320 s | 1.013 |
| 90 | 0.43% | 0.03% | 10/10 | 0.189 s | 1.007 |

A least-squares fit of geomean percentage gain to

\[
  g(N)=a+\frac{b}{N}
\]

gives

\[
  a=3.95,\qquad b=66.18,\qquad R^2=0.829.
\]

The positive fitted intercept is driven largely by scene 142, which retains a
3.8% refinement gain at 90 iterations. The median gain approaches zero much
more quickly than the geometric-mean gain.

## Conclusion

The inverse-budget intuition is substantially supported, though not as an exact
law. Final-only landmark polishing is especially valuable for intentionally
short DRS runs and costs one approximately constant distributed refinement
phase instead of one phase per outer iteration.

Use final-only polishing as the default output policy when the desired output
is a reduced physical BA state. Keep it separate from the DRS convergence
trajectory and report both the unpolished DRS state and polished output SSE.
For long runs its quality gain is small on most scenes, but its measured
relative overhead is also small.

Configuration:

```bash
LANDMARK_REFINEMENT_STEPS=0
CONSENSUS_LANDMARK_REFINEMENT_STEPS=1
CONSENSUS_LANDMARK_REFINEMENT_POLICY=final
```
