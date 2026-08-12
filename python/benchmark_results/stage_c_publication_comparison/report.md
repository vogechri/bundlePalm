# Final DRS, Schur, Ceres, And BAE Publication Comparison

All rows use independently evaluated standard pixel SSE. Iteration and thread budgets are fixed but intentionally differ by solver role. CPU DRS, CPU Ceres, and RTX 5090 BAE times are reported in separate timing classes; no cross-hardware speedup is claimed.

## Complete 1DSfM cohort (15/15)

| Method | Fixed specification | Scenes | SSE/Ceres | Summed SSE/Ceres | W/T/L vs Ceres | Optimization s | Timing class | SSE/base DRS | DRS time/base |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| Ceres | left-SE3, I90, T16 | 15 | 1.000000 | 1.000000 | 0/15/0 | 219.521 | CPU native solve | 0.686334 | -- |
| DRS K1 BAE-style | best local diagnostic, I90, T1 | 15 | 0.993529 | 0.947561 | 5/0/10 | 2060.937 | CPU DRS optimization | 0.681892 | 3.600524 |
| DRS K1 Schur-PCG | secondary local diagnostic, I80, T24 | 15 | 1.022987 | 1.087127 | 11/0/4 | 1048.950 | CPU DRS optimization | 0.702110 | 1.832550 |
| Base DRS K24 | preserved quality baseline, I200, T1/cluster | 15 | 1.457017 | 1.434854 | 1/0/14 | 572.399 | CPU DRS optimization | 1.000000 | 1.000000 |
| DRS+Schur fast | K24/I30 + up to 10 corrections | 15 | 1.411625 | 1.273158 | 1/0/14 | 191.161 | CPU DRS + Schur optimization | 0.968846 | 0.333964 |
| DRS+Schur balanced | K24/I60 + up to 10 corrections | 15 | 1.272048 | 1.147988 | 1/0/14 | 289.496 | CPU DRS + Schur optimization | 0.873049 | 0.505759 |
| DRS+Schur quality | K24/I30 + up to 20 corrections | 15 | 1.267361 | 1.160986 | 2/0/13 | 385.234 | CPU DRS + Schur optimization | 0.869832 | 0.673017 |
| DRS K4 | frozen C1+C5 resource endpoint, I30, T1/cluster | 15 | 2.052536 | 1.924183 | 0/0/15 | 160.607 | CPU DRS optimization | 1.408725 | 0.280585 |
| DRS K16 | frozen C1+C5 latency endpoint, I30, T1/cluster | 15 | 1.957296 | 1.848750 | 0/0/15 | 93.793 | CPU DRS optimization | 1.343358 | 0.163860 |

## Complete BAL cohort (29/29)

| Method | Fixed specification | Scenes | SSE/Ceres | Summed SSE/Ceres | W/T/L vs Ceres | Optimization s | Timing class | SSE/base DRS | DRS time/base |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| Ceres | left-SE3, I90, T16 | 29 | 1.000000 | 1.000000 | 0/29/0 | 1803.780 | CPU native solve | 1.001599 | -- |
| Base DRS K24 | established quality baseline, I90, T1/cluster | 29 | 0.998404 | 0.995488 | 12/0/17 | 1271.575 | CPU DRS optimization | 1.000000 | 1.000000 |
| DRS K4 | frozen C1+C5 resource endpoint, I30, T1/cluster | 29 | 1.007309 | 1.010726 | 8/0/21 | 939.466 | CPU DRS optimization | 1.008920 | 0.738821 |
| DRS K16 | frozen C1+C5 latency endpoint, I30, T1/cluster | 29 | 1.010098 | 1.016017 | 8/0/21 | 479.087 | CPU DRS optimization | 1.011713 | 0.376767 |

K1 has no authoritative matched all-29 BAL artifact and is therefore omitted from the BAL panel.

## Verified BAE inset (six 1DSfM scenes)

| Method | Fixed specification | Scenes | SSE/Ceres | Summed SSE/Ceres | W/T/L vs Ceres | Optimization s | Timing class | SSE/BAE-CG |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Ceres | left-SE3, I90, T16 | 6 | 1.000000 | 1.000000 | 0/6/0 | 103.504 | CPU native solve | 1.058573 |
| DRS K1 BAE-style | best local diagnostic, I90, T1 | 6 | 0.953562 | 0.910225 | 4/0/2 | 1413.708 | CPU DRS optimization | 1.009415 |
| DRS K1 Schur-PCG | secondary local diagnostic, I80, T24 | 6 | 1.041629 | 1.135638 | 4/0/2 | 543.595 | CPU DRS optimization | 1.102641 |
| Base DRS K24 | preserved quality baseline, I200 | 6 | 1.304394 | 1.203876 | 1/0/5 | 315.889 | CPU DRS optimization | 1.380797 |
| DRS K4 | frozen C1+C5 resource endpoint, I30, T1/cluster | 6 | 1.922802 | 1.720695 | 0/0/6 | 93.262 | CPU DRS optimization | 2.035427 |
| DRS K16 | frozen C1+C5 latency endpoint, I30, T1/cluster | 6 | 1.745880 | 1.627073 | 0/0/6 | 52.823 | CPU DRS optimization | 1.848141 |
| BAE Schur-PCG CG | verified exported state, I90 | 6 | 0.944668 | 0.894484 | 4/0/2 | 56.170 | RTX 5090 GPU optimization | 1.000000 |
| BAE Schur-PCG Nesterov | verified exported state, I90 | 6 | 1.091773 | 1.147230 | 4/0/2 | 90.859 | RTX 5090 GPU optimization | 1.155722 |

## Interpretation

The best BAE-style K1 local diagnostic reaches near-Ceres aggregate quality on all 15 1DSfM scenes but is neither distributed nor a matched work budget. The preserved longer-horizon base DRS is better in endpoint quality than current K4/K16 on both families. K4 and K16 are therefore speed endpoints, not quality replacements: C1+C5 improves its matched I30 plain control, but that gain does not overcome the shorter horizon. The three DRS+Schur rows are separately labeled polishing workflows rather than DRS-only gains. Each improves both endpoint quality and measured optimization time relative to the preserved I200 base; fast, balanced, and quality expose distinct budget points. Verified BAE coverage is only six 1DSfM scenes, uses GPU hardware, and is basin-sensitive on Trafalgar; it is contextual evidence, not an all-scene or deterministic reference. No per-scene settings or solver routing are used.
