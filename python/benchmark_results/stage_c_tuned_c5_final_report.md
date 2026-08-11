# Final Tuned Global C1+C5 Result

One global C5 policy is used everywhere: start I5, high/low thresholds `0.35/0.20`, rolling window 3, dwell 3, maximum depth 2. The setting was selected on six development 1DSfM plus five BAL sentinels, frozen for nine held-out 1DSfM scenes, then confirmed on all 29 BAL scenes.

| Family | Comparison | Geomean SSE | Summed SSE | W/T/L | Worst | Opt. |
|---|---|---:|---:|---:|---:|---:|
| 1DSFM | C1/plain | 0.865147004 | 0.864685169 | 13/0/2 | 1.098217105 | 1.557830 |
| 1DSFM | C5/plain | 1.002127516 | 1.003857238 | 8/1/6 | 1.209441402 | 1.456217 |
| 1DSFM | C1+C5/plain | 0.847890308 | 0.850831612 | 14/0/1 | 1.000458521 | 2.140832 |
| 1DSFM | C1+C5/C1 | 0.980053452 | 0.983978496 | 6/2/7 | 1.097713297 | 1.374240 |
| 1DSFM | C1+C5/C5 | 0.846090238 | 0.847562362 | 15/0/0 | 0.991202185 | 1.470132 |
| 1DSFM | C1+C5/Ceres | 2.141228969 | 1.941492754 | 0/0/15 | 3.429181865 | -- |
| BAL | C1/plain | 0.983780202 | 0.985830859 | 28/0/1 | 1.002733966 | 1.820914 |
| BAL | C5/plain | 1.000027001 | 1.000011213 | 0/28/1 | 1.000783337 | 1.322693 |
| BAL | C1+C5/plain | 0.983274885 | 0.984419239 | 28/0/1 | 1.002733966 | 2.294455 |
| BAL | C1+C5/C1 | 0.999486351 | 0.998568090 | 2/27/0 | 1.000000000 | 1.260057 |
| BAL | C1+C5/C5 | 0.983248336 | 0.984408200 | 28/0/1 | 1.002733966 | 1.734684 |
| BAL | C1+C5/Ceres | 1.009698445 | 1.013431877 | 7/0/22 | 1.144007515 | -- |

## Decision

Promote the tuned C1+C5 stack as the final Stage-C configuration. It improves the accepted C1 rung geometrically on both complete families and uses one common policy. Keep C1 and C5 independently switchable for ablations. No scene-specific settings are used.

This promotion is within the matched K24/I30 Stage-C ablation. It does not
supersede the longer-horizon best base DRS in endpoint quality.

## Deterministic repeats

After one excluded warm-up, three measured repeats on Roman, Trafalgar, BAL52,
and BAL3068 have zero endpoint SSE spread and identical rejection/oracle counts.
Optimization-time CV is `1.1%--2.4%`; overall-time CV is `0.9%--2.5%`. See
`stage_c_tuned_repeats_k24_i30/report.md`.

## K>1 scaling confirmation

The frozen policy was swept globally at K2/K4/K8/K16/K24 on four sentinels,
then K4 and K16 were run unchanged on all 15 1DSfM and all 29 BAL scenes. K4 is
the resource endpoint; K16 is the latency endpoint. K16/K4 geometric SSE is
`0.953599x` on 1DSfM and `1.002768x` on BAL, while optimization time is
`0.584x`/`0.510x`. No scene-specific K routing is used. See
`stage_c_scaling_confirmation_k4_16_i30/report.md`.

Three measured K4/K16 repeats after one excluded warm-up preserve identical
rejection/oracle counts and are SSE-bitwise identical in `7/8` cases; maximum
relative SSE spread is `2.956e-09` and maximum optimization-time CV is `1.75%`.
Mean K16/K4 optimization ratios are `0.5158x` on 1DSfM and `0.4190x` on BAL.
See `stage_c_scaling_repeats_k4_16_i30/report.md`.

## Final reference comparison

The cohort-explicit publication table compares K1/K4/K16 with Ceres on all 15
1DSfM scenes, K4/K16 with Ceres on all 29 BAL scenes, and verified BAE only on
its matched six-scene 1DSfM cohort. K1 and BAE coverage gaps are explicit, and
CPU/GPU timings are not turned into cross-hardware speedups. The corrected
table also includes preserved best base DRS. K4/K16 are faster, but reach
`1.408725x`/`1.343358x` base SSE on 1DSfM and
`1.008920x`/`1.011713x` on BAL. See
`stage_c_publication_comparison/report.md`.
