# K24 Quarter-Damped I60+I90 Krylov Interaction

The promoted lightweight Krylov2 proposal is applied unchanged at I60 and I90. Damping, scales, three landmark steps, `1e-3` floor, atomic commit, restart, and trust rebase are frozen.

| Family | Completed | Candidate/I60 | Candidate/control | Candidate/Ceres | W/T/L I60 | Second selected | Max RSS GiB C/W |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm | 15/15 | 0.979938558 | 0.748366977 | 1.114255670 | 11/0/4 | 15/15 | 1.858/1.777 |
| bal | 29/29 | 1.000000000 | 1.000000000 | 0.999038939 | 0/29/0 | 0/29 | 6.905/9.895 |

| Scene | Candidate/I60 | Candidate/control | Candidate/Ceres | Second scale | Rejection delta |
|---|---:|---:|---:|---:|---:|
| alamo | 1.000202646 | 0.897159010 | 1.077970483 | 1 | 2 |
| bal1064 | 1.000000000 | 1.000000000 | 1.001798665 | declined | 0 |
| bal126 | 1.000000000 | 1.000000000 | 1.016852250 | declined | 0 |
| bal1266 | 1.000000000 | 1.000000000 | 1.002286985 | declined | 0 |
| bal135 | 1.000000000 | 1.000000000 | 1.001461213 | declined | 0 |
| bal142 | 1.000000000 | 1.000000000 | 1.003381351 | declined | 0 |
| bal1490 | 1.000000000 | 1.000000000 | 0.994770842 | declined | 0 |
| bal1723 | 1.000000000 | 1.000000000 | 0.996496923 | declined | 0 |
| bal173 | 1.000000000 | 1.000000000 | 0.998715334 | declined | 0 |
| bal1778 | 1.000000000 | 1.000000000 | 1.012137802 | declined | 0 |
| bal245 | 1.000000000 | 1.000000000 | 0.964296960 | declined | 0 |
| bal253 | 1.000000000 | 1.000000000 | 0.996460685 | declined | 0 |
| bal257 | 1.000000000 | 1.000000000 | 1.025701761 | declined | 0 |
| bal287 | 1.000000000 | 1.000000000 | 0.998682986 | declined | 0 |
| bal3068 | 1.000000000 | 1.000000000 | 0.990923943 | declined | 0 |
| bal308 | 1.000000000 | 1.000000000 | 1.002929535 | declined | 0 |
| bal356 | 1.000000000 | 1.000000000 | 1.020981401 | declined | 0 |
| bal394 | 1.000000000 | 1.000000000 | 1.007367510 | declined | 0 |
| bal427 | 1.000000000 | 1.000000000 | 0.996946389 | declined | 0 |
| bal49 | 1.000000000 | 1.000000000 | 1.002088395 | declined | 0 |
| bal52 | 1.000000000 | 1.000000000 | 0.970685033 | declined | 0 |
| bal646 | 1.000000000 | 1.000000000 | 1.002052314 | declined | 0 |
| bal744 | 1.000000000 | 1.000000000 | 0.997314467 | declined | 0 |
| bal783 | 1.000000000 | 1.000000000 | 1.000692664 | declined | 0 |
| bal871 | 1.000000000 | 1.000000000 | 1.000614884 | declined | 0 |
| bal88 | 1.000000000 | 1.000000000 | 1.000093968 | declined | 0 |
| bal89 | 1.000000000 | 1.000000000 | 1.006441419 | declined | 0 |
| bal931 | 1.000000000 | 1.000000000 | 1.000308565 | declined | 0 |
| bal951 | 1.000000000 | 1.000000000 | 0.962279887 | declined | 0 |
| bal961 | 1.000000000 | 1.000000000 | 1.000073324 | declined | 0 |
| ellis_island | 0.994391281 | 0.952193043 | 1.046954567 | 1 | 2 |
| gendarmenmarkt | 1.015435987 | 0.981971868 | 1.281858584 | 1 | 1 |
| madrid_metropolis | 1.023649214 | 1.014837008 | 1.137889809 | 1 | 1 |
| montreal_notre_dame | 0.973556693 | 0.524642843 | 1.042532249 | 1 | 5 |
| notre_dame | 0.988817984 | 0.597050688 | 1.061278698 | 0.5 | 0 |
| nyc_library | 0.987712195 | 0.847122001 | 1.098027133 | 1 | 1 |
| piazza_del_popolo | 0.938710892 | 0.507421019 | 0.926705386 | 1 | 1 |
| piccadilly | 0.993136370 | 0.884868154 | 0.712072257 | 1 | 1 |
| roman_forum | 0.971075854 | 0.590206270 | 1.072345245 | 1 | 0 |
| tower_of_london | 0.939288799 | 0.791021262 | 1.622096579 | 1 | 0 |
| trafalgar | 1.002359498 | 0.893179183 | 0.946473058 | 0.5 | 8 |
| union_square | 0.974037947 | 0.742615484 | 1.112460133 | 1 | 1 |
| vienna_cathedral | 0.964661550 | 0.570452696 | 1.037351161 | 1 | 0 |
| yorkminster | 0.937256069 | 0.721850594 | 2.021017829 | 1 | -8 |

The second Krylov proposal is aggregate-positive on 1DSfM but introduces four I60 regressions and leaves Madrid above ordinary control. BAL29 remains bitwise exact with both proposals declined. Retain I60+I90 as a bounded-loss component, not the common preset; do not tune checkpoint, floor, scale, damping, or Krylov depth.

Gate status: **passed**.
