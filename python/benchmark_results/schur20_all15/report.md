# Frozen Schur-Correction Budget Frontier

Date: 2026-08-12

## Policy

All runs use the frozen K24/I30 base-backbone Themelis handoff followed by
`bsr_low_memory` global Schur corrections. Camera/landmark damping starts at
3/3, accepted corrections reduce damping geometrically, rejected attempts
increase it, CG must converge, independent physical SSE must decrease, and the
unchanged relative-progress stop is `1e-3`. Only the maximum correction budget
changes.

A five-scene development frontier (Madrid, Montreal, Roman, Tower, Trafalgar)
showed continued useful descent through correction 20. Corrections 11--20
improved the five-scene endpoint another `0.879704x`; correction 20 itself
improved it `1.3702%`. Madrid and Montreal reached the progress stop at
correction 20; the other three remained cap-limited. The correction cap was
therefore frozen at 20 before all-15 confirmation.

## All-15 Frontier

The cap-20 run reproduces the cap-10 I30 handoffs and first ten accepted
corrections exactly.

| Correction cap | SSE / established I200 base | SSE / Ceres | Schur seconds | Combined optimization seconds | Time / I200 base |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.292154616 | 1.882691700 | 0.000 | 90.998 | 0.158976958 |
| 3 | 1.116626283 | 1.626943872 | 23.915 | 114.913 | 0.200757475 |
| 8 | 0.996372638 | 1.451732224 | 67.303 | 158.301 | 0.276556853 |
| 10 | 0.968845633 | 1.411624901 | 100.163 | 191.161 | 0.333964108 |
| 13 | 0.934248984 | 1.361216983 | 164.031 | 255.029 | 0.445544621 |
| 15 | 0.907957797 | 1.322910267 | 210.018 | 301.016 | 0.525885524 |
| 20 | 0.869832394 | 1.267360894 | 294.236 | 385.234 | 0.673016881 |

All 15 scenes complete. The cap-20 endpoint is `0.673164x` its I30 handoff,
`0.869832x` the established I200 base, and `1.267361x` Ceres geometrically.
Summed SSE is `1.160986x` Ceres and W/T/L is `2/0/13`. Peak coordinator RSS is
`2,720,252 KiB` on Trafalgar.

Seven scenes reach the `1e-3` progress stop after 13--20 accepted corrections:
Ellis 13, Alamo/Yorkminster 16, Notre Dame 17, Piccadilly 18, and
Madrid/Montreal 20. Eight scenes remain cap-limited at 20. Every accepted CG
solve converges and every accepted correction decreases independently evaluated
physical SSE.

## Decision

The frozen correction-budget frontier contains at least two useful Pareto
points:

- cap 10: `0.968846x` base-I200 quality at `0.333964x` its optimization time;
- cap 20: `0.869832x` base-I200 quality at `0.673017x` its optimization time.

Neither dominates the other. Retain cap 10 as the faster quality preset and cap
20 as the quality-focused preset. Do not tune damping, tolerance, or progress
threshold from this cohort. Eight cap-limited scenes show that cap 20 is not a
convergence claim; further budget extension is a separate quality-ceiling study,
not required to establish the current Pareto frontier.
