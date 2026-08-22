# K24 Joint Camera/Landmark Proposals at I60 and I90

The promoted I60 joint camera/landmark proposal is followed by one additional, identically safeguarded proposal at I90 after 30 ordinary DRS iterations. Both checkpoints use one shared-camera Schur residual action, eight fixed scales, three rollback-safe landmark steps, the `1e-3` floor, atomic camera/landmark commit, canonical restart, and next-iteration trust rebase. Two proposals require an explicit default-off gate.

| Scene | I60+I90 delivered/control | I60-only | I60 decision | I90 decision | Rejections two/one |
|---|---:|---:|---|---|---:|
| Roman Forum | 0.628975086 | 0.652918451 | scale 1 | scale 1 | 9/9 |
| Trafalgar | 0.890794840 | 0.888640846 | scale 1 | scale 1 | 20/19 |
| BAL52 | 1.000000000 | 1.000000000 | declined | declined | 0/0 |
| BAL3068 | 1.000000000 | 1.000000000 | declined | declined | 3/3 |

The Roman/Trafalgar geomean improves from `0.761715173x` control with I60 only to `0.748523721x` with I60+I90. Roman gains materially while Trafalgar has a bounded `0.24%` regression. BAL declines both proposals exactly. Transfer the frozen two-proposal policy unchanged to all-15 once; do not add another checkpoint or proposal-count sweep.
