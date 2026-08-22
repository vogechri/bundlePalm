# K24 Proposal Evaluation on Full 1DSfM

All complete K24/I120 proposal variants are compared on the same 15 scenes. Control is ordinary integrated DRS through I120; Ceres is the canonical external reference.

| Arm | Delivered/control geometric | Summed/control | W/T/L vs control | Delivered/Ceres geometric | Summed/Ceres | Worst control ratio | Rejections |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 1.000000000 | 1.000000000 | 0/15/0 | 1.488916139 | 1.427410018 | 1.000000000 | 275 |
| camera_i90 | 0.887673383 | 0.861947562 | 12/1/2 | 1.321671226 | 1.230352585 | 1.037741851 | 136 |
| camera_i90_rebase | 0.884651719 | 0.860613356 | 13/1/1 | 1.317172223 | 1.228448127 | 1.010662208 | 121 |
| joint_i90 | 0.790889111 | 0.760039171 | 14/1/0 | 1.177567561 | 1.084887527 | 1.000000000 | 118 |
| joint_i60 | 0.786532802 | 0.754474107 | 14/1/0 | 1.171081383 | 1.076943899 | 1.000000000 | 127 |
| joint_i60_i90 | 0.766084290 | 0.732090008 | 13/1/1 | 1.140635264 | 1.044992612 | 1.019556957 | 128 |

## Per-Scene Ceres Ratios

| Scene | Control | Camera I90 | Camera I90+rebase | Joint I90 | Joint I60 | Joint I60+I90 | Best proposal |
|---|---:|---:|---:|---:|---:|---:|---|
| yorkminster | 2.799773 | 2.625803 | 2.618202 | 2.297447 | 2.281468 | 2.101563 | joint_i60_i90 (2.101563) |
| tower_of_london | 2.050636 | 2.073756 | 2.048869 | 1.828847 | 1.830323 | 1.755354 | joint_i60_i90 (1.755354) |
| gendarmenmarkt | 1.305392 | 1.354660 | 1.319311 | 1.274094 | 1.250070 | 1.330922 | joint_i60 (1.250070) |
| roman_forum | 1.816899 | 1.432867 | 1.449441 | 1.180316 | 1.186287 | 1.142784 | joint_i60_i90 (1.142784) |
| union_square | 1.498030 | 1.292139 | 1.294377 | 1.172070 | 1.164794 | 1.129027 | joint_i60_i90 (1.129027) |
| montreal_notre_dame | 1.987128 | 1.426943 | 1.423623 | 1.136260 | 1.139342 | 1.053310 | joint_i60_i90 (1.053310) |
| nyc_library | 1.296185 | 1.286790 | 1.278252 | 1.133357 | 1.139328 | 1.125884 | joint_i60_i90 (1.125884) |
| madrid_metropolis | 1.121254 | 1.121254 | 1.121254 | 1.121254 | 1.121254 | 1.121254 | camera_i90 (1.121254) |
| vienna_cathedral | 1.818470 | 1.357308 | 1.356641 | 1.119196 | 1.117719 | 1.061417 | joint_i60_i90 (1.061417) |
| notre_dame | 1.777535 | 1.342453 | 1.343354 | 1.124107 | 1.107871 | 1.080931 | joint_i60_i90 (1.080931) |
| alamo | 1.201538 | 1.112254 | 1.114088 | 1.081184 | 1.079292 | 1.088680 | joint_i60 (1.079292) |
| ellis_island | 1.099519 | 1.075128 | 1.079290 | 1.060122 | 1.058775 | 1.050407 | joint_i60_i90 (1.050407) |
| piazza_del_popolo | 1.826305 | 1.341771 | 1.322496 | 1.059998 | 1.052706 | 0.954781 | joint_i60_i90 (0.954781) |
| trafalgar | 1.059668 | 1.031130 | 1.029942 | 0.948820 | 0.941664 | 0.943946 | joint_i60 (0.941664) |
| piccadilly | 0.804721 | 0.773810 | 0.770686 | 0.748522 | 0.725631 | 0.722648 | joint_i60_i90 (0.722648) |

## Decision

Joint camera/landmark response is the dominant proposal mechanism. Joint I60 is the no-loss common default. I60+I90 has the best aggregate but loses Gendarmenmarkt and remains a bounded-loss research variant. The absolute I60 quality tails are Yorkminster and Tower of London; Madrid is the sole declined 1DSfM proposal.
