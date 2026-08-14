# Direct-Tangent Camera Parameterization Gate

Fresh K24/I30 runs use the promoted Stage-C C1+C5 stack. The only change is the camera retraction and its matching direct tangent Jacobian: left SE(3), product SO(3) x R3, or right SE(3).

| Family | Mode | SSE/left-SE3 | Summed/left-SE3 | W/T/L | Worst | Time/left-SE3 | Rejections | Oracles |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm | se3_left | 1.000000000 | 1.000000000 | 0/2/0 | 1.000000000 | 1.000000 | 12 | 99 |
| 1dsfm | so3_center_left | 0.964637277 | 0.929393582 | 1/0/1 | 1.032978376 | 1.015529 | 11 | 101 |
| bal | se3_left | 1.000000000 | 1.000000000 | 0/2/0 | 1.000000000 | 1.000000 | 19 | 94 |
| bal | so3_center_left | 0.953493074 | 0.950503016 | 1/0/1 | 1.000073630 | 1.158953 | 10 | 106 |

No scene-specific parameterization is selected.
