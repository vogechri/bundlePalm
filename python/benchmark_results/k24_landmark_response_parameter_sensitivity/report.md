# K24 Landmark-Response Parameter Sensitivity

One global parameter setting per arm is evaluated on Yorkminster, Tower of London, and declined Madrid. Proposal timing I60, scale grid, `1e-3` floor, restart, and trust policy are frozen.

| Arm | Damping | Landmark steps | Geometric/control | Geometric/Ceres | Selected | Rejections |
|---|---:|---:|---:|---:|---:|---:|
| damping_half | 0.0029296875 | 3 | 0.892254792 | 1.659818697 | 2/3 | 21 |
| base | 0.005859375 | 3 | 0.899311715 | 1.672946351 | 2/3 | 25 |
| damping_double | 0.01171875 | 3 | 0.903406069 | 1.680562881 | 2/3 | 17 |
| landmarks_1 | 0.005859375 | 1 | 0.908384991 | 1.689824930 | 2/3 | 21 |
| landmarks_5 | 0.005859375 | 5 | 0.911856626 | 1.696283046 | 2/3 | 27 |

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
