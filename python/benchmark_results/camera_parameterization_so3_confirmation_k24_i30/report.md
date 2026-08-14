# Direct-Tangent Camera Parameterization Gate

Fresh K24/I30 runs use the promoted Stage-C C1+C5 stack. The only change is the camera retraction and its matching direct tangent Jacobian: left SE(3), product SO(3) x R3, or right SE(3).

| Family | Mode | SSE/left-SE3 | Summed/left-SE3 | W/T/L | Worst | Time/left-SE3 | Rejections | Oracles |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm | se3_left | 1.000000000 | 1.000000000 | 0/9/0 | 1.000000000 | 1.000000 | 50 | 453 |
| 1dsfm | so3_left | 1.002319839 | 1.030994199 | 2/0/7 | 1.246462423 | 0.974534 | 53 | 443 |
| bal | se3_left | 1.000000000 | 1.000000000 | 0/29/0 | 1.000000000 | 1.000000 | 101 | 1571 |
| bal | so3_left | 0.996646096 | 0.995409597 | 23/0/6 | 1.001555315 | 0.987873 | 103 | 1576 |

No scene-specific parameterization is selected.
