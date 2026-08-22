# K24 Landmark-Response Parameter Sensitivity

One global parameter setting per arm is evaluated on Yorkminster, Tower of London, and declined Madrid. Proposal timing I60, scale grid, `1e-3` floor, restart, and trust policy are frozen.

| Arm | Camera damping | Landmark damping | Landmark steps | Geometric/control | Geometric/Ceres | Selected | Rejections |
|---|---:|---:|---:|---:|---:|---:|---:|
| damping_half | 0.0029296875 | 0.0029296875 | 3 | 0.892254792 | 1.659818697 | 2/3 | 21 |
| base | 0.005859375 | 0.005859375 | 3 | 0.899311715 | 1.672946351 | 2/3 | 25 |
| damping_double | 0.01171875 | 0.01171875 | 3 | 0.903406069 | 1.680562881 | 2/3 | 17 |
| landmarks_1 | 0.005859375 | 0.005859375 | 1 | 0.908384991 | 1.689824930 | 2/3 | 21 |
| landmarks_5 | 0.005859375 | 0.005859375 | 5 | 0.911856626 | 1.696283046 | 2/3 | 27 |
| camera_half | 0.0029296875 | 0.005859375 | 3 | 0.899982894 | 1.674194914 | 2/3 | 25 |
| landmark_half | 0.005859375 | 0.0029296875 | 3 | 0.895890448 | 1.666581933 | 2/3 | 25 |
| damping_quarter | 0.00146484375 | 0.00146484375 | 3 | 0.884512961 | 1.645416940 | 2/3 | 21 |

## Per-Scene Results

| Arm | Scene | Selected scale | Best scale | Immediate ratio | Delivered/control | Delivered/Ceres | Rejections |
|---|---|---:|---:|---:|---:|---:|---:|
| damping_half | madrid_metropolis | declined | 0.5 | 0.999036076 | 1.000000000 | 1.121253758 | 4 |
| damping_half | tower_of_london | 1 | 1 | 0.917187515 | 0.885540525 | 1.815921170 | 3 |
| damping_half | yorkminster | 1 | 1 | 0.819223442 | 0.802154874 | 2.245851590 | 14 |
| base | madrid_metropolis | declined | 0.5 | 0.999327842 | 1.000000000 | 1.121253758 | 4 |
| base | tower_of_london | 1 | 1 | 0.923871291 | 0.892563541 | 1.830322818 | 3 |
| base | yorkminster | 1 | 1 | 0.831759564 | 0.814876156 | 2.281468290 | 18 |
| damping_double | madrid_metropolis | declined | 1 | 0.999334457 | 1.000000000 | 1.121253758 | 4 |
| damping_double | tower_of_london | 1 | 1 | 0.932023059 | 0.904126733 | 1.854034716 | 3 |
| damping_double | yorkminster | 1 | 1 | 0.838312264 | 0.815491991 | 2.283192490 | 10 |
| landmarks_1 | madrid_metropolis | declined | 0.125 | 0.999912737 | 1.000000000 | 1.121253758 | 4 |
| landmarks_1 | tower_of_london | 1 | 1 | 0.933954263 | 0.914351593 | 1.875002180 | 3 |
| landmarks_1 | yorkminster | 1 | 1 | 0.866317141 | 0.819778687 | 2.295194265 | 14 |
| landmarks_5 | madrid_metropolis | declined | 0.5 | 0.999337352 | 1.000000000 | 1.121253758 | 4 |
| landmarks_5 | tower_of_london | 1 | 1 | 0.923799431 | 0.892081815 | 1.829334973 | 3 |
| landmarks_5 | yorkminster | 1 | 1 | 0.831207076 | 0.849914011 | 2.379566333 | 20 |
| camera_half | madrid_metropolis | declined | 0.5 | 0.999219728 | 1.000000000 | 1.121253758 | 4 |
| camera_half | tower_of_london | 1 | 1 | 0.922567473 | 0.895258763 | 1.835849738 | 3 |
| camera_half | yorkminster | 1 | 1 | 0.829628928 | 0.814243282 | 2.279696389 | 18 |
| landmark_half | madrid_metropolis | declined | 0.5 | 0.999183028 | 1.000000000 | 1.121253758 | 4 |
| landmark_half | tower_of_london | 1 | 1 | 0.918728500 | 0.887825150 | 1.820606104 | 3 |
| landmark_half | yorkminster | 1 | 1 | 0.822767511 | 0.809910959 | 2.267566867 | 18 |
| damping_quarter | madrid_metropolis | declined | 0.25 | 0.999251737 | 1.000000000 | 1.121253758 | 4 |
| damping_quarter | tower_of_london | 1 | 1 | 0.914278391 | 0.873031962 | 1.790270661 | 3 |
| damping_quarter | yorkminster | 1 | 1 | 0.816113873 | 0.792651817 | 2.219245186 | 14 |
