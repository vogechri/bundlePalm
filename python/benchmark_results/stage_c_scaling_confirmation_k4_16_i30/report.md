# Frozen Stage-C C1+C5 Cluster-Count Scaling Confirmation

One global L2 C1+C5 policy is held fixed for K=[4, 16], with one thread per cluster. Kmin is the smallest requested distributed configuration; Ceres quality uses the authoritative left-SE3 references. Times and resource totals cover the complete matched cohort.

## 1DSfM

| K | SSE/Kmin | SSE/Ceres | W/T/L | Opt. s | Opt./Kmin | Overall s | Worker CPU s | Worker RSS MiB | Traffic MiB | Oracles |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.000000 | 2.052536 | 0/15/0 | 160.607 | 1.000 | 175.445 | 399.820 | 1461.2 | 1161.0 | 755 |
| 16 | 0.953599 | 1.957296 | 9/0/6 | 93.793 | 0.584 | 121.393 | 581.980 | 1556.3 | 2237.8 | 749 |

## BAL

| K | SSE/Kmin | SSE/Ceres | W/T/L | Opt. s | Opt./Kmin | Overall s | Worker CPU s | Worker RSS MiB | Traffic MiB | Oracles |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.000000 | 1.007309 | 0/29/0 | 939.466 | 1.000 | 1124.443 | 2958.560 | 5145.5 | 3227.5 | 1652 |
| 16 | 1.002768 | 1.010098 | 9/0/20 | 479.087 | 0.510 | 703.537 | 4639.020 | 5224.0 | 6406.8 | 1604 |

## Decision

Retain two global, nondominated operating points with identical solver settings: K4 is the resource endpoint and K16 is the latency endpoint. K16/K4 geometric SSE is `0.953599x` on 1DSfM and `1.002768x` on BAL; optimization time is `0.584x` and `0.510x`. Worker CPU rises to `1.456x`/`1.568x`, and traffic rises to `1.927x`/`1.985x`. Do not route K by scene and do not alter C1 or C5.
