# Equal-Time DRS/Schur Allocation

Date: 2026-08-12

## Question

Given approximately the same total optimization investment, should the method
spend more work on accelerated distributed DRS before handing off to global
Schur polishing, or hand off earlier and buy more Schur corrections?

All arms use K24, the same base-backbone hybrid, proposal damping 0.5,
safeguarded Themelis acceleration, and the same `bsr_low_memory` Schur LM policy
with a `1e-3` progress stop. Safeguard annealing is fixed at 30 iterations so
shared prefixes are identical. The I30+15 endpoint is extracted exactly from
the frozen cap-20 traces; the I60+10 and I5+17 arms are independent all-15
confirmations. Their shared I30/I5 prefixes are bitwise identical to the frozen
I30 trajectory.

## Development Gate

The five-scene development set is Madrid, Montreal, Roman, Tower, and Trafalgar.
Measured allocations near 150 seconds were:

| Allocation | Total optimization s | SSE / equal-time base | W/L vs base |
|---|---:|---:|---:|
| I60 + 10 Schur | 147.282 | 0.869741 | 3/2 |
| I30 + 15 Schur | 153.864 | 0.909167 | 3/2 |
| I5 + 17 Schur | 149.744 | 1.064508 | 1/4 |

This selected all three unchanged schedules for all-15 confirmation; it did not
select or tune damping, CG tolerance, or the progress threshold.

## All-15 Confirmation

Each arm is compared with the nearest established base-DRS checkpoint by summed
optimization time.

| Allocation | DRS s | Schur s | Total s | Equal-time base | Base s | Time ratio | SSE / base | Summed SSE / base | W/T/L | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I60 + 10 Schur | 193.623 | 105.610 | 299.233 | I104 | 298.788 | 1.001490 | 0.819300 | 0.757394 | 13/0/2 | 1.221176 |
| I30 + 15 Schur | 90.998 | 210.018 | 301.016 | I105 | 301.736 | 0.997616 | 0.852585 | 0.794247 | 10/0/5 | 1.288819 |
| I5 + 17 Schur | 13.646 | 276.191 | 289.837 | I101 | 289.730 | 1.000366 | 0.951988 | 0.901570 | 9/0/6 | 1.543085 |

All 45 runs complete without recovery exhaustion. I60+10 is `0.961553x`
I30+15 and `0.858391x` I5+17 geometrically. It wins 10/15 scenes against each
alternative. I30+15 is `0.892714x` I5+17 and also wins 10/15.

I60+10 has 337 seconds summed case wall time and peaks at 2.581 GiB coordinator
RSS / 2.622 GiB worker RSS. I5+17 has 326 seconds summed wall time and peaks at
2.537/2.580 GiB. Resource differences are small relative to endpoint-quality
differences.

The subsequently promoted symbolic-cache and vectorized numeric BSR path is
bitwise endpoint-identical. A fresh all-15 I60+10 run takes `195.292s` DRS plus
`94.206s` Schur, `289.498s` total. Its nearest base reference is I101 at
`289.730s`; the optimized balanced preset reaches `0.817178x` geometric and
`0.755262x` summed SSE, W/T/L `13/0/2`, with a `0.999199x` time ratio.

## Decision

For a balanced approximately 300-second all-15 budget, use I60+10. More DRS
work before handoff is more valuable than replacing it with additional Schur
corrections at this budget. Very early I5 handoff leaves a substantially worse
basin and also makes each Schur correction more expensive.

This does not invalidate I30+10 as the fast preset or I30+20 as the
quality-focused preset; those occupy different total-work budgets. The
same-time result adds a balanced preset and establishes an allocation rule:
do not hand off before I30, and near the middle budget prefer extending DRS to
I60 before increasing the Schur tail beyond ten corrections.
