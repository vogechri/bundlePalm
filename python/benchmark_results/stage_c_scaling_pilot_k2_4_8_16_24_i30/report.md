# Frozen Stage-C C1+C5 Cluster-Count Scaling Pilot

One global L2 C1+C5 policy is held fixed for K=[2, 4, 8, 16, 24], with one thread per cluster. Kmin is the smallest requested distributed configuration; Ceres quality uses the authoritative left-SE3 references. Times and resource totals cover the matched sentinel cohort.

## 1DSfM

| K | SSE/Kmin | SSE/Ceres | W/T/L | Opt. s | Opt./Kmin | Overall s | Worker CPU s | Worker RSS MiB | Traffic MiB | Oracles |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1.000000 | 2.743000 | 0/2/0 | 99.688 | 1.000 | 105.781 | 148.830 | 1438.8 | 348.8 | 99 |
| 4 | 0.752038 | 2.062839 | 2/0/0 | 53.264 | 0.534 | 58.628 | 138.810 | 1454.7 | 411.3 | 101 |
| 8 | 0.783354 | 2.148738 | 2/0/0 | 34.213 | 0.343 | 41.699 | 150.400 | 1517.0 | 524.9 | 100 |
| 16 | 0.702196 | 1.926124 | 2/0/0 | 27.219 | 0.273 | 36.982 | 207.600 | 1559.6 | 688.2 | 99 |
| 24 | 0.832157 | 2.282605 | 1/0/1 | 28.482 | 0.286 | 32.622 | 291.440 | 1599.3 | 836.6 | 98 |

## BAL

| K | SSE/Kmin | SSE/Ceres | W/T/L | Opt. s | Opt./Kmin | Overall s | Worker CPU s | Worker RSS MiB | Traffic MiB | Oracles |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1.000000 | 1.080879 | 0/2/0 | 93.195 | 1.000 | 102.280 | 151.200 | 1756.0 | 227.7 | 112 |
| 4 | 0.968152 | 1.046455 | 2/0/0 | 54.798 | 0.588 | 67.091 | 155.130 | 1827.0 | 353.5 | 110 |
| 8 | 0.972024 | 1.050640 | 2/0/0 | 39.401 | 0.423 | 54.372 | 190.370 | 1828.7 | 475.5 | 113 |
| 16 | 0.990847 | 1.070985 | 2/0/0 | 22.291 | 0.239 | 41.763 | 194.010 | 1976.1 | 597.2 | 101 |
| 24 | 0.974831 | 1.053674 | 1/0/1 | 23.788 | 0.255 | 28.806 | 256.830 | 2035.1 | 776.1 | 101 |

## Gate

Use this pilot only to select globally defensible K values for the complete 15-scene and 29-scene scaling confirmation. Do not alter C1, C5, or scene-specific settings from this result.
