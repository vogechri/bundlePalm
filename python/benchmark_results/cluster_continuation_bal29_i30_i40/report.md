# K4 to K24 Continuation on BAL29

Date: 2026-08-14

## Protocol

The frozen K4/I3 to K24 schedule was evaluated on all 29 top-level BAL scenes
with the same Schur-PCG, block-Jacobi, persistent-DABA policy in every arm.
Canonical state handoff is exact. BAL49 is included in the complete corpus and
aggregates are also checked without it.

Comparisons:

- equal iteration: K4/I3 + K24/I27 versus direct K24/I30;
- approximate time match: staged I30 versus direct K24/I40;
- long tail: K4/I3 + K24/I87 versus direct K24/I90;
- polished tails: identical 16-correction global Schur polishing after I90 and
  I200.

## Aggregate Frontier

| Comparison | Geometric SSE | Summed SSE | W/T/L | Time |
|---|---:|---:|---:|---:|
| Staged I30 / direct I30 | 0.999377 | 0.999684 | 17/0/12 | 1.334163 |
| Staged I30 / direct I40 | 1.003477 | 1.004249 | 5/0/24 | 1.070811 |
| Staged I90 / direct I90 | 0.998902 | 0.997782 | 20/0/9 | 1.146393 |
| Staged I90+16 / direct I90+16 | 0.998969 | 0.997824 | 20/0/9 | 1.146703 |
| Staged I200+16 / direct I200+16 | 0.999876 | 0.999333 | 19/0/10 | 1.070250 |

Excluding BAL49 changes no decision. At I200+16 the 28-scene geometric/summed
ratios are `0.999664x/0.999328x` at `1.063816x` time.

## Interpretation

At the practical I30 horizon, continuation spends `33.4%` more time for only
`0.06%` geometric and `0.03%` summed improvement over direct I30. Giving direct
K24 ten additional iterations uses approximately the staged wall-time budget
and reverses the result: continuation is `0.35%` worse geometrically and
`0.42%` worse summed, losing 24 of 29 scenes.

The equal-iteration I90 tail retains a small path advantage, but it shrinks
toward one by I200+16. Both arms still usually record their best state at the
last outer iteration, and final Schur generally accepts only one tiny
correction. The experiment therefore does not establish distinct converged
minima. It shows finite-horizon path dependence whose aggregate value mostly
disappears when direct K24 receives comparable time.

Some scene-specific path effects remain: at I200+16 scene 871 is `0.993810x`,
245 is `0.995552x`, and 308 is `0.997403x`, while 394 is `1.005258x`, 1064 is
`1.002962x`, and 3068 is `1.001449x`. These cannot justify routing or tuning
from the current outcomes.

## Verdict

Cluster-count continuation is not a broadly useful BAL policy under the frozen
schedule. It mainly trades initialization and repartitioning time for earlier
progress and does not demonstrate a materially better minimum. Retain it as a
path/basin diagnostic and for exceptional 1DSfM cases such as Roman Forum; do
not promote it to a BAL or global preset.

Artifacts were produced by `serverTest/run_cluster_continuation_bal29.sh`.