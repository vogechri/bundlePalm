# K24 I90 One-Step Schur Proposal Breadth

One global I90 proposal, eight fixed geometric scales, precise worker-SSE selection, atomic DRS state rebuild, and I120 delivery. The rejected coupled-consensus oracle is disabled.

| Family | Completed | Selected/declined | Immediate I90 | Trajectory I120 | Delivered/control | Summed | W/T/L | Candidate/Ceres | Max RSS GiB C/W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm | 15/15 | 14/1 | 0.920695087 | 0.887752412 | 0.887673383 | 0.861947562 | 12/1/2 | 1.321671226 | 2.407/1.775 |

| Scene | Scale | Worker SSE | I90 | I120 trajectory | Delivered | Rejections C/P |
|---|---:|---:|---:|---:|---:|---:|
| alamo | 0.5 | 0.948638101 | 0.948866260 | 0.925691984 | 0.925691984 | 12/7 |
| ellis_island | 0.5 | 0.990576419 | 0.990576419 | 0.977816094 | 0.977816094 | 17/18 |
| gendarmenmarkt | 0.25 | 0.993459701 | 0.995591705 | 1.037741851 | 1.037741851 | 2/9 |
| madrid_metropolis | declined | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 4/4 |
| montreal_notre_dame | 0.5 | 0.781500845 | 0.781511689 | 0.718578849 | 0.718093115 | 13/12 |
| notre_dame | 0.5 | 0.802952514 | 0.805389203 | 0.755233009 | 0.755233009 | 7/4 |
| nyc_library | 0.25 | 0.988830434 | 0.991312781 | 0.992751719 | 0.992751719 | 3/10 |
| piazza_del_popolo | 0.5 | 0.847380614 | 0.847565538 | 0.734691883 | 0.734691883 | 15/14 |
| piccadilly | 0.5 | 0.975584607 | 0.978433879 | 0.961586976 | 0.961586976 | 8/9 |
| roman_forum | 0.5 | 0.886889776 | 0.888156681 | 0.788633492 | 0.788633492 | 11/8 |
| tower_of_london | 0.125 | 0.990429933 | 0.991428437 | 1.011274778 | 1.011274778 | 3/10 |
| trafalgar | 0.25 | 0.983614717 | 0.983759434 | 0.973069414 | 0.973069414 | 16/15 |
| union_square | 0.5 | 0.886124033 | 0.886738071 | 0.862390545 | 0.862559060 | 3/3 |
| vienna_cathedral | 0.5 | 0.809779402 | 0.809471870 | 0.747039468 | 0.746401332 | 2/2 |
| yorkminster | 0.25 | 0.961323905 | 0.961765917 | 0.937862772 | 0.937862772 | 14/11 |

Gate status: **failed**.

## Decision

Canonical product-state restart removes the catastrophic post-I90 live-state
instability. The aggregate effect is strong (`0.887673383x` geometric,
`0.861947562x` summed), with 12 wins, one exact declined tie, and two bounded
losses: Gendarmenmarkt `1.058521266x` and Tower of London `1.011274778x`.

The strict no-loss promotion gate therefore fails. Do not promote or tune the
scale grid, damping, proposal time, or progress floor from these tails. The
existing component policy permits an unchanged BAL safety transfer because the
losses are bounded and the aggregate gain is material. BAL must run serially
under the memory ceiling; its result decides whether this remains a research
component or closes entirely.
