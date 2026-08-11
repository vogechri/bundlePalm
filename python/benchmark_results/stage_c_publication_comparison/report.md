# Final K1/K4/K16, Ceres, And BAE Publication Comparison

All rows use independently evaluated standard pixel SSE. Iteration and thread budgets are fixed but intentionally differ by solver role. CPU DRS, CPU Ceres, and RTX 5090 BAE times are reported in separate timing classes; no cross-hardware speedup is claimed.

## Complete 1DSfM cohort (15/15)

| Method | Fixed specification | Scenes | SSE/Ceres | Summed SSE/Ceres | W/T/L vs Ceres | Optimization s | Timing class |
|---|---|---:|---:|---:|---:|---:|---|
| Ceres | left-SE3, I90, T16 | 15 | 1.000000 | 1.000000 | 0/15/0 | 219.521 | CPU native solve |
| DRS K1 | specialized local diagnostic, I80, T24 | 15 | 1.022987 | 1.087127 | 11/0/4 | 1048.950 | CPU DRS optimization |
| DRS K4 | frozen C1+C5 resource endpoint, I30, T1/cluster | 15 | 2.052536 | 1.924183 | 0/0/15 | 160.607 | CPU DRS optimization |
| DRS K16 | frozen C1+C5 latency endpoint, I30, T1/cluster | 15 | 1.957296 | 1.848750 | 0/0/15 | 93.793 | CPU DRS optimization |

## Complete BAL cohort (29/29)

| Method | Fixed specification | Scenes | SSE/Ceres | Summed SSE/Ceres | W/T/L vs Ceres | Optimization s | Timing class |
|---|---|---:|---:|---:|---:|---:|---|
| Ceres | left-SE3, I90, T16 | 29 | 1.000000 | 1.000000 | 0/29/0 | 1803.780 | CPU native solve |
| DRS K4 | frozen C1+C5 resource endpoint, I30, T1/cluster | 29 | 1.007309 | 1.010726 | 8/0/21 | 939.466 | CPU DRS optimization |
| DRS K16 | frozen C1+C5 latency endpoint, I30, T1/cluster | 29 | 1.010098 | 1.016017 | 8/0/21 | 479.087 | CPU DRS optimization |

K1 has no authoritative matched all-29 BAL artifact and is therefore omitted from the BAL panel.

## Verified BAE inset (six 1DSfM scenes)

| Method | Fixed specification | Scenes | SSE/Ceres | Summed SSE/Ceres | W/T/L vs Ceres | Optimization s | Timing class | SSE/BAE-CG |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Ceres | left-SE3, I90, T16 | 6 | 1.000000 | 1.000000 | 0/6/0 | 103.504 | CPU native solve | 1.058573 |
| DRS K1 | specialized local diagnostic, I80, T24 | 6 | 1.041629 | 1.135638 | 4/0/2 | 543.595 | CPU DRS optimization | 1.102641 |
| DRS K4 | frozen C1+C5 resource endpoint, I30, T1/cluster | 6 | 1.922802 | 1.720695 | 0/0/6 | 93.262 | CPU DRS optimization | 2.035427 |
| DRS K16 | frozen C1+C5 latency endpoint, I30, T1/cluster | 6 | 1.745880 | 1.627073 | 0/0/6 | 52.823 | CPU DRS optimization | 1.848141 |
| BAE Schur-PCG CG | verified exported state, I90 | 6 | 0.944668 | 0.894484 | 4/0/2 | 56.170 | RTX 5090 GPU optimization | 1.000000 |
| BAE Schur-PCG Nesterov | verified exported state, I90 | 6 | 1.091773 | 1.147230 | 4/0/2 | 90.859 | RTX 5090 GPU optimization | 1.155722 |

## Interpretation

K1 remains a local-solver diagnostic: it reaches near-Ceres aggregate quality on all 15 1DSfM scenes but is neither the distributed method nor a matched work budget. K4 and K16 are the frozen global distributed resource and latency endpoints. They remain close to Ceres on all 29 BAL scenes, while difficult 1DSfM initialization retains a substantial quality gap. Verified BAE coverage is only six 1DSfM scenes, uses GPU hardware, and is basin-sensitive on Trafalgar; it is contextual evidence, not an all-scene or deterministic reference. No per-scene settings or solver routing are used.
