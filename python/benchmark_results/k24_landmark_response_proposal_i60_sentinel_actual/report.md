# K24 Joint Camera/Landmark Proposal at I60

The promoted one-action Schur proposal with three-step landmark response is moved from I90 to the previously frozen alternate checkpoint I60. The eight scales, `1e-3` floor, atomic camera/landmark commit, canonical restart, and selected-only trust rebase are unchanged. This is the sole bounded timing comparison; no checkpoint sweep is permitted.

| Scene | I60 delivered/control | I90 delivered/control | Applied | Scale | Rejections I60/I90 |
|---|---:|---:|---|---:|---:|
| Roman Forum | 0.652918451 | 0.649632297 | yes | 1 | 9/7 |
| Trafalgar | 0.888640846 | 0.895394379 | yes | 1 | 19/17 |
| BAL52 | 1.000000000 | 1.000000000 | no | declined | 0/0 |
| BAL3068 | 1.000000000 | 1.000000000 | no | declined | 3/3 |

The Roman/Trafalgar geomean is `0.761715173x` control at I60 versus `0.762677590x` at I90. BAL remains exact because both proposals decline. I60 is a safe, slightly aggregate-positive component candidate with mixed 1DSfM effects. Transfer unchanged to all-15 once; do not test another checkpoint.
