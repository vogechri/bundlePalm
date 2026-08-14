# Direct-Tangent Camera Parameterization Gate

Fresh K24/I30 runs use the promoted Stage-C C1+C5 stack. The only change is the camera retraction and its matching direct tangent Jacobian: left SE(3), product SO(3) x R3, or right SE(3).

| Family | Mode | SSE/left-SE3 | Summed/left-SE3 | W/T/L | Worst | Time/left-SE3 | Rejections | Oracles |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm | se3_left | 1.000000000 | 1.000000000 | 0/6/0 | 1.000000000 | 1.000000 | 32 | 302 |
| 1dsfm | so3_left | 0.987431937 | 0.944768721 | 2/0/4 | 1.084378278 | 1.036779 | 34 | 300 |
| bal | se3_left | 1.000000000 | 1.000000000 | 0/5/0 | 1.000000000 | 1.000000 | 26 | 261 |
| bal | so3_left | 0.997954990 | 0.998523487 | 4/0/1 | 1.001255665 | 0.974646 | 25 | 264 |

No scene-specific parameterization is selected.
