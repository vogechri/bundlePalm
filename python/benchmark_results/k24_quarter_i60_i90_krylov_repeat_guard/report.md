# K24 Krylov Repeat-Intervention Guard

The promoted I60 Krylov2 proposal keeps its `1e-3` acceptance floor. A second I90 intervention must deliver at least `1%` immediate landmark-refined SSE decrease, accounting for restart and continuation opportunity cost. All other settings remain frozen.

| Family | Completed | Candidate/I60 | Candidate/control | Candidate/Ceres | W/T/L I60 | W/T/L control | Second selected | Max RSS GiB C/W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm | 15/15 | 0.978062825 | 0.746934503 | 1.112122837 | 9/6/0 | 15/0/0 | 9/15 | 1.818/1.771 |
| bal | 29/29 | 1.000000000 | 1.000000000 | 0.999038939 | 0/29/0 | 0/29/0 | 0/29 | 6.896/12.023 |

| Scene | Immediate ratio | Candidate/I60 | Candidate/control | Candidate/Ceres | Second decision |
|---|---:|---:|---:|---:|---:|
| alamo | 1.000000000 | 1.000000000 | 0.896977242 | 1.077752081 | declined |
| bal1064 | 1.000000000 | 1.000000000 | 1.000000000 | 1.001798665 | declined |
| bal126 | 1.000000000 | 1.000000000 | 1.000000000 | 1.016852250 | declined |
| bal1266 | 1.000000000 | 1.000000000 | 1.000000000 | 1.002286985 | declined |
| bal135 | 1.000000000 | 1.000000000 | 1.000000000 | 1.001461213 | declined |
| bal142 | 1.000000000 | 1.000000000 | 1.000000000 | 1.003381351 | declined |
| bal1490 | 1.000000000 | 1.000000000 | 1.000000000 | 0.994770842 | declined |
| bal1723 | 1.000000000 | 1.000000000 | 1.000000000 | 0.996496923 | declined |
| bal173 | 1.000000000 | 1.000000000 | 1.000000000 | 0.998715334 | declined |
| bal1778 | 1.000000000 | 1.000000000 | 1.000000000 | 1.012137802 | declined |
| bal245 | 1.000000000 | 1.000000000 | 1.000000000 | 0.964296960 | declined |
| bal253 | 1.000000000 | 1.000000000 | 1.000000000 | 0.996460685 | declined |
| bal257 | 1.000000000 | 1.000000000 | 1.000000000 | 1.025701761 | declined |
| bal287 | 1.000000000 | 1.000000000 | 1.000000000 | 0.998682986 | declined |
| bal3068 | 1.000000000 | 1.000000000 | 1.000000000 | 0.990923943 | declined |
| bal308 | 1.000000000 | 1.000000000 | 1.000000000 | 1.002929535 | declined |
| bal356 | 1.000000000 | 1.000000000 | 1.000000000 | 1.020981401 | declined |
| bal394 | 1.000000000 | 1.000000000 | 1.000000000 | 1.007367510 | declined |
| bal427 | 1.000000000 | 1.000000000 | 1.000000000 | 0.996946389 | declined |
| bal49 | 1.000000000 | 1.000000000 | 1.000000000 | 1.002088395 | declined |
| bal52 | 1.000000000 | 1.000000000 | 1.000000000 | 0.970685033 | declined |
| bal646 | 1.000000000 | 1.000000000 | 1.000000000 | 1.002052314 | declined |
| bal744 | 1.000000000 | 1.000000000 | 1.000000000 | 0.997314467 | declined |
| bal783 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000692664 | declined |
| bal871 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000614884 | declined |
| bal88 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000093968 | declined |
| bal89 | 1.000000000 | 1.000000000 | 1.000000000 | 1.006441419 | declined |
| bal931 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000308565 | declined |
| bal951 | 1.000000000 | 1.000000000 | 1.000000000 | 0.962279887 | declined |
| bal961 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000073324 | declined |
| ellis_island | 1.000000000 | 1.000000003 | 0.957563752 | 1.052859766 | declined |
| gendarmenmarkt | 1.000000000 | 1.000000000 | 0.967044581 | 1.262372617 | declined |
| madrid_metropolis | 1.000000000 | 1.000000000 | 0.991391382 | 1.111601312 | declined |
| montreal_notre_dame | 0.979341884 | 0.973556693 | 0.524642843 | 1.042532249 | scale 1 |
| notre_dame | 0.980717131 | 0.988817984 | 0.597050688 | 1.061278698 | scale 0.5 |
| nyc_library | 0.981649736 | 0.987712195 | 0.847122001 | 1.098027133 | scale 1 |
| piazza_del_popolo | 0.952458123 | 0.938710892 | 0.507421019 | 0.926705386 | scale 1 |
| piccadilly | 1.000000000 | 1.000000000 | 0.890983536 | 0.716993435 | declined |
| roman_forum | 0.962490105 | 0.971075854 | 0.590206270 | 1.072345245 | scale 1 |
| tower_of_london | 0.954829574 | 0.939288799 | 0.791021262 | 1.622096579 | scale 1 |
| trafalgar | 1.000000000 | 1.000000000 | 0.891076690 | 0.944245114 | declined |
| union_square | 0.975149099 | 0.974037947 | 0.742615484 | 1.112460133 | scale 1 |
| vienna_cathedral | 0.975261884 | 0.964661550 | 0.570452696 | 1.037351161 | scale 1 |
| yorkminster | 0.967685146 | 0.937256069 | 0.721850594 | 2.021017829 | scale 1 |

The repeat margin preserves every I60 endpoint within `1e-8` relative tolerance, removes all ordinary-control losses, and keeps BAL29 exact with both interventions declined. Promote the guarded I60+I90 policy as the common preset; retain unguarded repetition as a bounded-loss ablation. Do not tune the margin or checkpoints.

Gate status: **passed**.
