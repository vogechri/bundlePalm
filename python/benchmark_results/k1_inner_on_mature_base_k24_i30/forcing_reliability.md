# Relative Interior-Forcing Reliability

The candidate trigger is `q = sqrt(interior defect / proximal displacement) > 1`. The L1 diagnostic is bitwise trajectory-neutral. Fixed L2 is used only as available endpoint ground truth.

| Scene | Max q | q>1 iterations | Early area | L2/L1 | L2 complete |
|---|---:|---:|---:|---:|---:|
| bal1490 | 0.413430 | 0 | 0.000000 | 0.999018 | yes |
| bal3068 | 1.266555 | 1 | 0.266555 | 0.987166 | yes |
| gendarmenmarkt | 2.099168 | 5 | 1.281766 | 1.031202 | yes |
| piccadilly | 1.058552 | 1 | 0.058552 | 1.064862 | yes |
| roman_forum | 1.276678 | 2 | 0.000000 | 1.140172 | yes |
| trafalgar | 2.160977 | 5 | 1.847904 | 0.991819 | yes |
| union_square | 1.497022 | 4 | 0.132120 | 1.008528 | yes |
| vienna_cathedral | 1.163931 | 2 | 0.286478 | 0.958092 | yes |

## Reliability

At threshold 1: TP/FP/FN/TN = 3/4/1/0, precision `0.429`, recall `0.750`, and sign accuracy `0.375`. Maximum-q Spearman correlation with L2 benefit is `-0.071`.

Reject q>1 as a standalone forcing selector. It triggers on four 1DSfM losses and misses the small BAL1490 gain. Early-area ranking is only suggestive on eight scenes and is not a promotion statistic.

## Limitation

Fixed L2 repeats the full local nonlinear solve and updates shared cameras as well as interior variables. The diagnostic q measures only unique-camera/landmark stationarity. A true interior-only trial with shared cameras fixed is required before rejecting relative forcing itself; this experiment rejects only q>1 as a direct trigger for the existing full-L2 intervention.
