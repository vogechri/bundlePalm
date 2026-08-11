# C4 post-crash six-scene K24 breadth

This applies the frozen `K24/I30/L1` Nesterov-versus-Schur-PCG comparison to
the established six-scene 1DSfM set. No solver tolerance or outer policy was
changed after the Roman/Trafalgar gates.

| Scene | PCG / Nesterov SSE | PCG / Nesterov optimization time | Quality result |
|---|---:|---:|---|
| Gendarmenmarkt | 1.0124 | 0.8677 | loss |
| Piccadilly | 1.0913 | 0.9371 | loss |
| Roman Forum | 0.9789 | 0.8589 | win |
| Trafalgar | 0.9042 | 0.8560 | win |
| Union Square | 0.9757 | 0.9884 | win |
| Vienna Cathedral | 1.0160 | 0.8870 | loss |

PCG wins quality on `3/6` scenes and runtime on `6/6`. Summed SSE is
`0.9745x`, geometric-mean per-scene SSE is `0.9948x`, and geometric-mean
optimization time is `0.8979x` Nesterov. The Piccadilly `1.0913x` regression
prevents uniform promotion from this cohort. The next action is one unchanged
all-15 breadth decision; do not tune tolerances or introduce scene-dependent
solver selection.