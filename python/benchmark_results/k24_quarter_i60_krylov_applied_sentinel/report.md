# K24 I60 Quarter-Damped Two-Direction Krylov Proposal

Two preconditioned conjugate directions replace the promoted single block-Jacobi camera action. Eight scales, three fixed-camera landmark steps, the `1e-3` floor, atomic commit, canonical restart, and selected-only trust rebase remain unchanged.

The proposal-only path skips the converged Schur reference and is trajectory- and state-exact to the diagnostic path on all 15 1DSfM scenes and all 29 BAL scenes.

Lightweight/diagnostic elapsed geometric ratios are `0.991012x` for 1DSfM and `0.899325x` for BAL.

| Family | Completed | Candidate/quarter | Candidate/base | Candidate/control | Candidate/Ceres | W/T/L quarter | Selected | Max RSS GiB C/W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm | 15/15 | 0.985801044 | 0.970954614 | 0.763687653 | 1.137066872 | 13/0/2 | 15/15 | 2.016/1.759 |
| bal | 29/29 | 1.000000000 | 1.000000000 | 1.000000000 | 0.999038939 | 0/29/0 | 0/29 | 7.469/10.755 |

| Family | Median one-action cosine/norm/gain | Median Krylov cosine/norm/gain |
|---|---:|---:|
| 1dsfm | 0.591503/0.496443/0.916577 | 0.728366/0.668087/0.992271 |
| bal | n/a | n/a |

| Scene | Candidate/quarter | Candidate/control | Candidate/Ceres | Selected scale |
|---|---:|---:|---:|---:|
| alamo | 1.000488191 | 0.896977242 | 1.077752081 | 1 |
| bal1064 | 1.000000000 | 1.000000000 | 1.001798665 | declined |
| bal126 | 1.000000000 | 1.000000000 | 1.016852250 | declined |
| bal1266 | 1.000000000 | 1.000000000 | 1.002286985 | declined |
| bal135 | 1.000000000 | 1.000000000 | 1.001461213 | declined |
| bal142 | 1.000000000 | 1.000000000 | 1.003381351 | declined |
| bal1490 | 1.000000000 | 1.000000000 | 0.994770842 | declined |
| bal1723 | 1.000000000 | 1.000000000 | 0.996496923 | declined |
| bal173 | 1.000000000 | 1.000000000 | 0.998715334 | declined |
| bal1778 | 1.000000000 | 1.000000000 | 1.012137802 | declined |
| bal245 | 1.000000000 | 1.000000000 | 0.964296960 | declined |
| bal253 | 1.000000000 | 1.000000000 | 0.996460685 | declined |
| bal257 | 1.000000000 | 1.000000000 | 1.025701761 | declined |
| bal287 | 1.000000000 | 1.000000000 | 0.998682986 | declined |
| bal3068 | 1.000000000 | 1.000000000 | 0.990923943 | declined |
| bal308 | 1.000000000 | 1.000000000 | 1.002929535 | declined |
| bal356 | 1.000000000 | 1.000000000 | 1.020981401 | declined |
| bal394 | 1.000000000 | 1.000000000 | 1.007367510 | declined |
| bal427 | 1.000000000 | 1.000000000 | 0.996946389 | declined |
| bal49 | 1.000000000 | 1.000000000 | 1.002088395 | declined |
| bal52 | 1.000000000 | 1.000000000 | 0.970685033 | declined |
| bal646 | 1.000000000 | 1.000000000 | 1.002052314 | declined |
| bal744 | 1.000000000 | 1.000000000 | 0.997314467 | declined |
| bal783 | 1.000000000 | 1.000000000 | 1.000692664 | declined |
| bal871 | 1.000000000 | 1.000000000 | 1.000614884 | declined |
| bal88 | 1.000000000 | 1.000000000 | 1.000093968 | declined |
| bal89 | 1.000000000 | 1.000000000 | 1.006441419 | declined |
| bal931 | 1.000000000 | 1.000000000 | 1.000308565 | declined |
| bal951 | 1.000000000 | 1.000000000 | 0.962279887 | declined |
| bal961 | 1.000000000 | 1.000000000 | 1.000073324 | declined |
| ellis_island | 0.998664072 | 0.957563749 | 1.052859762 | 1 |
| gendarmenmarkt | 1.003041074 | 0.967044581 | 1.262372617 | 1 |
| madrid_metropolis | 0.991391382 | 0.991391382 | 1.111601312 | 1 |
| montreal_notre_dame | 0.952782488 | 0.538892954 | 1.070849038 | 1 |
| notre_dame | 0.988496894 | 0.603802416 | 1.073280133 | 1 |
| nyc_library | 0.996019934 | 0.857660770 | 1.111687330 | 1 |
| piazza_del_popolo | 0.984311246 | 0.540550902 | 0.987210646 | 1 |
| piccadilly | 0.993767400 | 0.890983536 | 0.716993435 | 1 |
| roman_forum | 0.976079823 | 0.607785960 | 1.104285768 | 1 |
| tower_of_london | 0.964625792 | 0.842149148 | 1.726941255 | 1 |
| trafalgar | 0.992405409 | 0.891076690 | 0.944245114 | 1 |
| union_square | 0.992964008 | 0.762409192 | 1.142111697 | 1 |
| vienna_cathedral | 0.981784394 | 0.591350091 | 1.075352450 | 1 |
| yorkminster | 0.971642711 | 0.770174360 | 2.156313408 | 1 |

Gate status: **passed**.
