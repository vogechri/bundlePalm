# K24 I60 Landmark-Response Quarter-Damping Validation

The extended three-scene sensitivity decomposed camera versus landmark damping and tested one lower boundary point. Camera-half alone is slightly worse than base; landmark-half improves base, showing landmark regularization drives the gain. Quarter damping (`0.00146484375` for both camera and landmark Schur terms) improves both Yorkminster and Tower and was frozen before held-out validation on the other 12 scenes.

| Metric | Quarter | Half | Base |
|---|---:|---:|---:|
| Geometric delivered/control | 0.774687405 | 0.777853720 | 0.786532802 |
| Summed delivered/control | 0.746301671 | 0.748189557 | 0.754474107 |
| Quarter/half geometric | 0.995929421 | 1.000000000 | - |
| W/T/L quarter versus half | 12/1/2 | - | - |

Quarter damping improves the absolute tails further: Tower `0.873031962x` control and Yorkminster `0.792651817x`. It improves 12 scenes versus half damping, ties declined Madrid, and has two bounded regressions: Piazza `1.011494x` and Trafalgar `1.008284x` quarter/half. Madrid remains declined and moves away from the fixed floor (`0.0748%` immediate decrease), confirming damping sensitivity addresses accepted-tail quality rather than forcing marginal proposals.

Freeze quarter damping here; do not continue a halving sweep. Run unchanged BAL29 safety before promotion.
