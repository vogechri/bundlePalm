# K24 Landmark-Response DRS Plus One Terminal Correction

Promoted K24/I120 landmark-response DRS followed by exactly one safeguarded distributed Schur correction. The correction uses damping `0.005859375`, `bsr_low_memory`, Jacobi PCG, relative tolerance `1e-6`, and progress floor `1e-3`.

| Family | Completed | Correction/handoff | Corrected/Ceres | Summed/Ceres | W/T/L correction | Accepted/no-op | Nonconverged no-op | Max RSS GiB C/W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm | 15/15 | 0.964506795 | 1.135771915 | 1.044180104 | 15/0/0 | 15/0 | 0 | 2.442/1.913 |
| bal | 29/29 | 0.999641721 | 0.998681005 | 0.994776429 | 26/3/0 | 26/3 | 3 | 8.306/12.051 |

| Scene | Correction/handoff | Corrected/Ceres | Status | Attempts | Final damping |
|---|---:|---:|---|---:|---:|
| alamo | 0.994949589 | 1.075723334 | accepted | 1 | 0.0029296875 |
| bal1064 | 0.999941371 | 1.001739931 | accepted | 1 | 0.0029296875 |
| bal126 | 0.999923039 | 1.016773992 | accepted | 1 | 0.0029296875 |
| bal1266 | 0.999769619 | 1.002056078 | accepted | 1 | 0.0029296875 |
| bal135 | 0.996343834 | 0.997799705 | accepted | 1 | 0.0029296875 |
| bal142 | 0.998679890 | 1.002056777 | accepted | 1 | 0.0029296875 |
| bal1490 | 1.000000000 | 0.994770842 | no-op | 6 | 0.375 |
| bal1723 | 0.999929943 | 0.996427112 | accepted | 1 | 0.0029296875 |
| bal173 | 0.999490956 | 0.998206944 | accepted | 1 | 0.0029296875 |
| bal1778 | 1.000000000 | 1.012137802 | no-op | 6 | 0.375 |
| bal245 | 0.999623498 | 0.963933900 | accepted | 1 | 0.0029296875 |
| bal253 | 0.999976215 | 0.996436984 | accepted | 1 | 0.0029296875 |
| bal257 | 0.999953517 | 1.025654083 | accepted | 2 | 0.005859375 |
| bal287 | 0.999964903 | 0.998647936 | accepted | 1 | 0.0029296875 |
| bal3068 | 0.998706718 | 0.989642398 | accepted | 2 | 0.005859375 |
| bal308 | 0.999928464 | 1.002857789 | accepted | 1 | 0.0029296875 |
| bal356 | 0.999883288 | 1.020862240 | accepted | 1 | 0.0029296875 |
| bal394 | 0.999635690 | 1.007000515 | accepted | 1 | 0.0029296875 |
| bal427 | 0.999346272 | 0.996294658 | accepted | 3 | 0.01171875 |
| bal49 | 0.999701432 | 1.001789203 | accepted | 1 | 0.0029296875 |
| bal52 | 0.999412357 | 0.970114618 | accepted | 1 | 0.0029296875 |
| bal646 | 0.999714024 | 1.001765750 | accepted | 1 | 0.0029296875 |
| bal744 | 0.999989205 | 0.997303701 | accepted | 4 | 0.0234375 |
| bal783 | 0.999895283 | 1.000587874 | accepted | 1 | 0.0029296875 |
| bal871 | 0.999929792 | 1.000544633 | accepted | 1 | 0.0029296875 |
| bal88 | 0.999914261 | 1.000008221 | accepted | 1 | 0.0029296875 |
| bal89 | 0.999992143 | 1.006433512 | accepted | 1 | 0.0029296875 |
| bal931 | 0.999984366 | 1.000292927 | accepted | 1 | 0.0029296875 |
| bal951 | 1.000000000 | 0.962279887 | no-op | 6 | 0.375 |
| bal961 | 0.999987165 | 1.000060488 | accepted | 1 | 0.0029296875 |
| ellis_island | 0.989433013 | 1.048919333 | accepted | 1 | 0.0029296875 |
| gendarmenmarkt | 0.992237266 | 1.264203841 | accepted | 1 | 0.0029296875 |
| madrid_metropolis | 0.994989402 | 1.115635606 | accepted | 1 | 0.0029296875 |
| montreal_notre_dame | 0.938488874 | 1.066367019 | accepted | 1 | 0.0029296875 |
| notre_dame | 0.934879838 | 1.050905156 | accepted | 1 | 0.0029296875 |
| nyc_library | 0.976930910 | 1.107211218 | accepted | 1 | 0.0029296875 |
| piazza_del_popolo | 0.911235415 | 0.965907443 | accepted | 1 | 0.0029296875 |
| piccadilly | 0.982989848 | 0.735789854 | accepted | 1 | 0.0029296875 |
| roman_forum | 0.951065924 | 1.122558651 | accepted | 1 | 0.0029296875 |
| tower_of_london | 0.954595155 | 1.745808149 | accepted | 1 | 0.0029296875 |
| trafalgar | 0.976092668 | 0.926136664 | accepted | 1 | 0.0029296875 |
| union_square | 0.967728933 | 1.134246244 | accepted | 1 | 0.0029296875 |
| vienna_cathedral | 0.956060725 | 1.070019640 | accepted | 1 | 0.0029296875 |
| yorkminster | 0.950537849 | 2.183810668 | accepted | 1 | 0.0029296875 |

Gate status: **passed**.
