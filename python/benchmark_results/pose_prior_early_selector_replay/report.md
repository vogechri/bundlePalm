# Pose-Prior Early Selector Replay

## Question

Can the removable relative-pose-prior branch be selected early enough to retain
its quality gains without paying the known correction-nine oracle cost?

The replay uses the complete raw and prior K24/I60+10 trajectories from:

- `benchmark_results/equal_time_all15/i60_s10/`;
- `benchmark_results/pose_prior_balanced_all15_i60_s10/`.

At a decision point, both branches are charged through that point and only the
selected branch is charged for its recorded remainder. This is an artifact
replay estimate, not a measured concurrent implementation.

## Schur-Prefix Replay

Selecting the branch with lower physical SSE after each Schur prefix gives:

| Corrections | Geometric SSE/control | Summed SSE/control | W/T/L | Estimated time/control |
|---:|---:|---:|---:|---:|
| 0 | 0.981565 | 0.987642 | 7/7/1 | 1.640087 |
| 5 | 0.981565 | 0.987642 | 7/7/1 | 1.759168 |
| 8 | 0.981565 | 0.987642 | 7/7/1 | 1.841357 |
| 9 | 0.981570 | 0.987563 | 6/9/0 | 1.900388 |
| 10 | 0.980812 | 0.986745 | 9/6/0 | 1.959600 |

Notre Dame is the loss at every prefix through correction eight. Correction
nine remains the first loss-free physical-SSE selector and is too expensive as
a practical policy.

## Outer-Iteration Gate

A natural fixed grid was evaluated on the historical six-scene development
split. The earliest Pareto candidate selects the prior branch at I5 only when
its physical SSE is at least 5% below raw. On development it gives `0.993792x`
geometric SSE, `0.995742x` summed SSE, W/T/L `3/3/0`, and estimated runtime
`1.083392x`.

Frozen unchanged on the held-out nine, it gives `0.996138x` geometric SSE,
`0.998680x` summed SSE, W/T/L `4/3/2`, and estimated runtime `1.053427x`.
Notre Dame and Yorkminster regress `0.5167%` and `0.6987%`, despite I5
prior/raw signals of `0.900369` and `0.883230`. The candidate therefore fails
the loss-free held-out gate and must not be retuned on those scenes.

A retrospective all-15 rule at I11 with a 2% margin is loss-free and reaches
`0.983708x` geometric SSE at estimated `1.149724x` time, but it was discovered
using all 15 outcomes and is not independent validation.

## Decision

Reject early physical-SSE branch selection. Neither a Schur-prefix race nor a
development-selected outer-iteration dominance margin is composition-safe.
Do not implement the branch race or tune another threshold on the held-out
reversals. A practical basin mechanism now requires either a new proposal that
is robust without selection or an independently justified quality signal that
contains information beyond the pixel objective.