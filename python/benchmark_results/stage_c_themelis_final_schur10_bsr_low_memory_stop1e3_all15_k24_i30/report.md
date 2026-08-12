# Ten-Correction Global-Schur 1DSfM Gate

Date: 2026-08-12

## Configuration

All 15 1DSfM scenes use the frozen K24/I30 base-backbone hybrid: local
Nesterov, persistent DRS trust, curvature 0.4 recovery/decay, camera metric 75,
shared-only proximal terms, proposal damping 0.5 on duplicated cameras, and
safeguarded Themelis outer acceleration. The final distributed Schur loop uses
`bsr_low_memory`, damping 3/3 with geometric decrease/increase, a maximum of ten
accepted corrections, independent physical-SSE acceptance, converged CG, and
the frozen `1e-3` relative-progress stop.

## Cumulative Quality

| Accepted corrections | Endpoint / I30 handoff | Endpoint / established I200 base |
|---:|---:|---:|
| 0 | 1.000000000 | 1.292154616 |
| 1 | 0.952526112 | 1.230811012 |
| 2 | 0.906519263 | 1.171363049 |
| 3 | 0.864158414 | 1.116626283 |
| 4 | 0.831941904 | 1.074997572 |
| 5 | 0.809954990 | 1.046587079 |
| 6 | 0.794156404 | 1.026172863 |
| 7 | 0.782746379 | 1.011429346 |
| 8 | 0.771093974 | 0.996372638 |
| 9 | 0.761595906 | 0.984099665 |
| 10 | 0.749790792 | 0.968845633 |

Correction eight crosses the established I200 base. After ten corrections the
candidate is `0.968846x` base-I200 and `1.411625x` Ceres geometrically; summed
SSE is `1.273158x` Ceres, with W/T/L `1/0/14`.

All 15 scenes accept all ten corrections and terminate at the correction cap.
I30 DRS takes `90.718s` and the correction tail takes `97.773s`, for `188.491s`
combined optimization work: `0.329301x` the established I200 base optimization
time (`572.399s`). Peak coordinator RSS is `2,660,804 KiB` on Trafalgar. Every
accepted linear solve converges. The geometric per-correction SSE decreases
remain positive through correction ten. Median damped model gain ratios decline
from `2.062` at correction one to `1.223` at correction seven, then
`0.877/0.677/0.456` at corrections eight through ten. This is declining
local-model calibration, not correction failure: independent physical SSE
continues to decrease on every scene.

## Decision

The previous three-correction result was under-polished. The global Schur
approximation does not break at correction three; the accepted LM sequence
closes and then surpasses the historical base-I200 endpoint with one global
policy at about one third of its optimization time. Retain ten corrections as
the current 1DSfM quality setting and keep the physical-SSE and converged-CG
acceptance gates. The ten-correction cap is still active on every scene, so
this run establishes a Pareto quality point, not Schur convergence.
