# Mid-Run Schur Product-State Transport Gate

Fresh K24/I30 arms use no correction, the existing I10 reset, or the same accepted correction transported through local camera copies and centers with exact dual-offset and trust preservation.

| Family | Arm | SSE/control | SSE/reset | Summed/control | W/T/L vs control | Opt. s | I10--I14 accepts/rejects |
|---|---|---:|---:|---:|---:|---:|---:|
| 1dsfm | control | 1.000000000 | 1.047591748 | 1.000000000 | 0/2/0 | 20.735 | 8/2 |
| 1dsfm | reset | 0.954570329 | 1.000000000 | 0.947395771 | 2/0/0 | 22.198 | 3/7 |
| 1dsfm | transport | 0.973731923 | 1.020073528 | 0.933522735 | 1/0/1 | 23.356 | 6/4 |
| bal | control | 1.000000000 | 1.003186844 | 1.000000000 | 0/2/0 | 46.575 | 6/4 |
| bal | reset | 0.996823280 | 1.000000000 | 0.996562692 | 1/1/0 | 82.443 | 6/4 |
| bal | transport | 0.999466346 | 1.002651489 | 0.999421805 | 1/1/0 | 78.400 | 9/1 |

All settings are global; no scene selects an arm.

## Decision

Reject dual-preserving mid-Schur transport as a quality mechanism. It partially repairs the continuation failure: I10--I14 accepts/rejects improve from `3/7` to `6/4` on the 1DSfM pair and from `6/4` to `9/1` on BAL. Trafalgar and BAL3068 continue more smoothly. This does not transfer to endpoint quality. Transport/reset geometric SSE is `1.020074x` on 1DSfM and `1.002651x` on BAL. Roman remains at `1/4` post-correction accepts/rejects and ends `1.103501x` reset; BAL3068 ends `1.005310x` reset. BAL1490 rejects the global correction itself and therefore remains an exact reset/transport tie.

Product-space collapse explains part of the post-correction stall, but preserving dual offsets and trust state does not produce a safer common basin. Retain the default-off transport path as state-transition diagnostic infrastructure; do not broaden or tune the I10 trigger.
