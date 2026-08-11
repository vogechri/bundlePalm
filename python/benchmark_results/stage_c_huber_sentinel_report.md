# Stage-C Huber 0.5 Sentinel Gate

Date: 2026-08-11

## Contract

The same observation-level Huber loss with pixel scale `0.5` is used by:

- the distributed worker objective and IRLS normal equations;
- worker-owned consensus cost replies;
- coordinator safeguards, line searches, and best-state selection;
- the Python state evaluator;
- the centralized left-SE3 Ceres reference.

Raw pixel SSE remains a separately named reporting metric. It is not used as
the robust acceptance objective. Ceres native robust cost and the shared state
evaluator agree to better than `1e-10` relative error on the sentinel cohort;
worker and coordinator robust objectives agree exactly in the runtime smoke.

## Local-model repair

Using Ceres's generic robust Jacobian correction directly in the custom Schur
solver produced an indefinite or inconsistent local model: Roman I6 had only
10 positive actual decreases across 337 attempts and median trust ratio `-8.97`.
The repaired worker uses explicit observation-level Huber IRLS weights for a
correct-gradient positive-semidefinite Gauss-Newton system and exact Huber cost
for the trust numerator. Roman I6 then has 140 positive decreases across 217
attempts, median radius `7.57`, positive median trust ratio, and reduces robust
objective from 5,350,811 to 2,651,209.

## Raw-start Huber I30

| Scene | DRS robust objective | Ceres robust objective | DRS/Ceres | DRS raw SSE |
|---|---:|---:|---:|---:|
| Roman Forum | 2,419,580 | 261,534 | 9.2515x | 33,881,074 |
| Trafalgar | 6,253,449 | 950,878 | 6.5765x | 296,797,982 |
| BAL52 | 139,353 | 117,557 | 1.1854x | 601,765 |
| BAL3068 | 1,166,346 | 860,702 | 1.3551x | 166,093,489 |

Geometric DRS/Ceres robust-objective ratio is `7.8002x` on the two 1DSfM
sentinels and `1.2674x` on the two BAL sentinels. The robust path is numerically
healthy but not competitive from the raw initialization at I30.

## L2-to-Huber continuation

A common continuation starts from the frozen tuned L2 C1+C5 state and adds 30
Huber iterations.

| Scene | Final/start robust objective | Final/Ceres |
|---|---:|---:|
| Roman Forum | 0.995842x | 3.0899x |
| Trafalgar | 1.000000x | 2.1373x |
| BAL52 | 1.000000x | 1.1170x |
| BAL3068 | 0.980283x | 1.2210x |

The continuation gains only `0.997919x` geometrically on 1DSfM and `0.990092x`
on BAL while doubling the 30-iteration outer budget. It remains `2.5698x` and
`1.1679x` Ceres geometrically.

## Decision

Retain the shared Huber objective implementation and report it as a validated
robust-objective capability. Do not broaden the current raw-start or
L2-to-Huber policies: neither lies on the present quality/work frontier. The
promoted final configuration remains the tuned L2 C1+C5 stack. A future Huber
study requires a new continuation or globalization mechanism, not more scalar
threshold tuning.
