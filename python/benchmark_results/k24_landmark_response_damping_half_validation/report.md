# K24 I60 Landmark-Response Half-Damping Validation

Development used Yorkminster, Tower of London, and declined Madrid. The frozen global candidate halves camera and landmark damping from `0.005859375` to `0.0029296875`, retaining I60 timing, three landmark steps, eight scales, `1e-3` floor, atomic commit, canonical restart, and trust rebase. It was then run unchanged on the other 12 1DSfM scenes.

| Metric | Half damping | Base damping |
|---|---:|---:|
| Geometric delivered/control | 0.777853720 | 0.786532802 |
| Summed delivered/control | 0.748189557 | 0.754474107 |
| Half/base geometric | 0.988965392 | 1.000000000 |
| W/T/L half versus base | 11/1/3 | - |
| Selected/declined | 14/1 | 14/1 |

Half damping improves both absolute tails: Tower `0.885540525x` control versus base `0.892563541x`, and Yorkminster `0.802154874x` versus `0.814876156x`. It also materially improves Roman and Piazza. The three regressions versus base are bounded: Gendarmenmarkt `1.007920x`, Alamo `1.003642x`, and Trafalgar `1.002113x`. Madrid's immediate refined decrease grows from `0.0672%` to `0.0964%` but remains below the fixed `0.1%` floor and correctly declines.

This is a globally validated aggregate-positive component candidate, not a scene-specific setting. Do not continue a damping sweep. Run the frozen half-damping policy on BAL29 for safety before promotion.
