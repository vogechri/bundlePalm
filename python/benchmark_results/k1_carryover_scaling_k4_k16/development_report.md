# K1 Carryover Combined Stack At K4/K16

The copied mature direct-left-SE3 baseline is compared with the frozen combined shared-only plus guarded I60+I90 Krylov stack. No K24-selected setting changes. K4 is the resource deployment point and K16 the latency deployment point. This report covers the frozen `development` cohort.

| Family/K | Candidate/control | Candidate/Ceres | W/T/L control | Selected I60/I90 | Full trajectories control/candidate | Elapsed ratio | Max RSS GiB C/W |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm/K4 | 0.714243368 | 1.035312831 | 6/0/0 | 6/2 (6/6 attempted) | 6/6 | 1.352159 | 1.458/1.563 |
| 1dsfm/K16 | 0.788555765 | 1.054164556 | 5/0/1 | 6/4 (6/6 attempted) | 6/6 | 1.007481 | 1.767/1.759 |
| bal/K4 | 1.001972252 | 0.998188728 | 0/2/3 | 0/0 (5/5 attempted) | 5/5 | 1.108091 | 3.075/5.055 |
| bal/K16 | 0.997714436 | 0.991197118 | 3/2/0 | 0/0 (5/5 attempted) | 5/5 | 1.471676 | 3.887/5.077 |

| Family | K16/K4 SSE | W/T/L | Worst | K16/K4 elapsed control/candidate |
|---|---:|---:|---:|---:|
| 1dsfm | 1.018208724 | 3/0/3 | 1.128267544 | 0.666113/0.496314 |
| bal | 0.992995704 | 3/0/2 | 1.013101904 | 0.559871/0.743575 |

Recovery exhaustion: `none`.

This is the frozen six-1DSfM/BAL5 development gate. Advance unchanged to held-out/all15/all29 only if both families have bounded aggregate quality and no unsafe candidate completion failure. Do not tune K-specific settings.

Gate bounds: geometric and summed candidate/control at most `1.01x`, worst scene at most `1.02x`, and no candidate recovery exhaustion.

Gate failures:

- 1dsfm/K16 worst candidate/control 1.022191218 > 1.02

Gate status: **failed**.

Disposition: close this frozen common K4/K16 transfer without breadth expansion or K-specific retuning. Retain the accepted K24 combined stack and mandatory algebraic carryovers.
