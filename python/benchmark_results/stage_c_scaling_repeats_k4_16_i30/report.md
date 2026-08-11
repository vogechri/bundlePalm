# K4/K16 Scaling Repeatability Gate

One warm-up is excluded. 3 measured repeats use the frozen global L2 C1+C5 policy, fixed partition cache, and one thread per cluster.

| Family | Scene | K | SSE spread | Opt. s | Opt. CV | Overall s | Overall CV | Worker CPU s | CPU CV | RSS spread | Traffic spread |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm | roman_forum | 4 | 0.000e+00 | 15.978 | 0.0099 | 17.575 | 0.0111 | 42.807 | 0.0023 | 1.303e-03 | 0.000e+00 |
| 1dsfm | roman_forum | 16 | 0.000e+00 | 8.598 | 0.0015 | 9.942 | 0.0071 | 62.487 | 0.0170 | 2.307e-03 | 0.000e+00 |
| 1dsfm | trafalgar | 4 | 0.000e+00 | 36.903 | 0.0029 | 39.807 | 0.0037 | 94.987 | 0.0038 | 6.697e-03 | 0.000e+00 |
| 1dsfm | trafalgar | 16 | 2.956e-09 | 18.676 | 0.0010 | 21.381 | 0.0022 | 145.693 | 0.0020 | 2.807e-04 | 0.000e+00 |
| bal | bal52 | 4 | 0.000e+00 | 7.716 | 0.0037 | 8.487 | 0.0050 | 24.220 | 0.0044 | 5.554e-04 | 0.000e+00 |
| bal | bal52 | 16 | 0.000e+00 | 3.422 | 0.0175 | 4.432 | 0.1109 | 31.727 | 0.0187 | 5.298e-03 | 0.000e+00 |
| bal | bal3068 | 4 | 0.000e+00 | 45.783 | 0.0032 | 49.491 | 0.0030 | 127.233 | 0.0013 | 5.791e-03 | 0.000e+00 |
| bal | bal3068 | 16 | 0.000e+00 | 18.992 | 0.0029 | 22.414 | 0.0036 | 164.680 | 0.0045 | 1.344e-02 | 0.000e+00 |

## Endpoint ratios

| Family | K16/K4 opt. | Ratio CV | K16/K4 overall | Ratio CV | K16/K4 worker CPU | K16/K4 traffic |
|---|---:|---:|---:|---:|---:|---:|
| 1dsfm | 0.5158 | 0.0050 | 0.5459 | 0.0047 | 1.5108 | 1.6734 |
| bal | 0.4190 | 0.0039 | 0.4630 | 0.0192 | 1.2968 | 1.6895 |

## Decision

Endpoint SSE is bitwise identical in `7/8` cases; maximum relative spread is `2.956e-09`. Rejection and oracle counts are identical; maximum per-case optimization-time CV is `0.0175`. Repeatability is reported without changing either global operating point.
