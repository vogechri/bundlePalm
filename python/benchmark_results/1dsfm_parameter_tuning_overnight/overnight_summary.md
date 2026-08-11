# 1DSfM Parameter Tuning Overnight Summary

Date: 2026-08-04

## Timing Caveat

Endpoint SSE, rejection counts, fallback counts, and oracle-call counts are the
authoritative tuning signals. Wall-clock results are exploratory because the
machine was also used for other work and variants were run sequentially rather
than in randomized paired order. In particular, aggregate speedups driven by a
few large scenes must not be treated as confirmed performance claims.

Any promoted timing result should be repeated with baseline and candidate runs
paired per scene, order randomized, at least three repetitions, and no competing
CPU/GPU workload. Confirmation should report median optimization time and the
worker-operation counters in addition to wall time. Priority repeats are the
largest or most timing-sensitive scenes, including Trafalgar, 951, 1778, 3068,
and any grid scene showing a ratio outside `[0.8, 1.25]`.

## Protocol

- development scenes: Tower of London, Madrid Metropolis, Gendarmenmarkt,
	Notre Dame, and NYC Library;
- K24, clean original initialization, standard pixel objective;
- 24 parameter arms screened at I200;
- complete top-five arms promoted to I1000;
- BAL sentinels: 52, 245, 394, 871, and 1723 at I90;
- original baseline: metric `25`, regularization `5e-5`, curvature `0.4`,
	trust diagonal `1e-4`, proposal scale `0.6`, relaxation `1`, enhanced I30.

## I200 Screen

Sixteen arms completed all five scenes through I200. Ratios are geometric mean
pixel SSE relative to the baseline arm.

| Rank | Variant | Ratio | Best scene | Worst scene |
|---:|---|---:|---:|---:|
| 1 | regularization `1e-4` | 0.903722 | 0.813471 | 1.006287 |
| 2 | metric `100` | 0.903795 | 0.812198 | 0.974929 |
| 3 | metric `50` | 0.924795 | 0.876147 | 1.006384 |
| 4 | curvature `0.8` | 0.946171 | 0.857232 | 1.003922 |
| 5 | trust diagonal `2e-4` | 0.982180 | 0.959988 | 0.997750 |
| 6 | enhanced through I60 | 0.994252 | 0.900833 | 1.063526 |
| 7 | trust diagonal `5e-5` | 0.995289 | 0.969344 | 1.019729 |
| 8 | proposal scale `0.3` | 0.998571 | 0.887184 | 1.082482 |
| 9 | baseline | 1.000000 | 1.000000 | 1.000000 |
| 10 | curvature `0.2` | 1.011186 | 0.938607 | 1.058368 |
| 11 | proposal scale `1.0` | 1.013896 | 0.936121 | 1.121727 |
| 12 | curvature `1.6` | 1.020600 | 0.958248 | 1.115966 |
| 13 | trust diagonal `4e-4` | 1.033512 | 0.993727 | 1.081318 |
| 14 | proposal scale `0.8` | 1.042327 | 0.906243 | 1.185737 |
| 15 | two local steps | 1.069846 | 0.980001 | 1.144496 |
| 16 | regularization `2.5e-5` | 1.087991 | 0.985036 | 1.249337 |

Failure accounting:

- camera step `0.75` is incompatible with outer acceleration;
- camera step `1.25` is outside the supported `(0, 1]` range;
- both combination arms inherited one of those invalid camera steps;
- no enhanced window and metric `12.5` each produced a non-finite Notre Dame
	worker update;
- relaxation `1.2` produced non-finite updates on Notre Dame and NYC Library;
- relaxation `0.8` exhausted recovery on Tower at I78.

## Promoted I1000

Curvature `0.8` is disqualified because Madrid produced a non-finite worker
update at I1000. The other four promoted arms completed all five scenes.

| Variant | Geomean vs original I1000 | Geomean vs Ceres | Wins vs original | Time ratio |
|---|---:|---:|---:|---:|
| metric `100` | **0.934427** | **1.058373** | 5/5 | 1.023x |
| regularization `1e-4` | 0.940857 | 1.065656 | 5/5 | 1.024x |
| metric `50` | 0.958496 | 1.085634 | 4/5 | 1.003x |
| trust diagonal `2e-4` | 1.018649 | 1.153766 | 1/5 | 0.998x |

### Metric 100 By Scene

| Scene | Final SSE | Ratio vs original I1000 | Ratio vs Ceres |
|---|---:|---:|---:|
| Gendarmenmarkt | 641,804.767 | 0.966006 | 1.121174 |
| Madrid Metropolis | 202,399.679 | 0.908560 | 0.946014 |
| Notre Dame | 4,862,382.388 | 0.921291 | 1.025033 |
| NYC Library | 945,738.096 | 0.971170 | 1.033020 |
| Tower of London | 1,409,356.043 | 0.907198 | 1.182433 |

Metric `100` is the best long-horizon aggregate and improves every scene, but
it does not eliminate the Tower and Gendarmenmarkt gaps. Regularization `1e-4`
is better on Tower (`1.050x` Ceres versus `1.182x`) but worse in aggregate.

## Exact Metric-100 Plus Regularization-1e-4 Combination

The clean pair was not part of the original 24-arm matrix; the two original
combination arms changed several other parameters and were invalid because of
unsupported camera-step choices. The exact pair was therefore run separately.

At I200, the combination is `0.882802x` baseline SSE, improving all five
scenes. It is `0.976851x` regularization-only and `0.976772x` metric-100-only.

At I1000, the positive interaction strengthens:

| Comparison | Geomean SSE ratio | Scene wins |
|---|---:|---:|
| Combined / original I1000 | **0.896707** | 5/5 |
| Combined / regularization `1e-4` | 0.953075 | 5/5 |
| Combined / metric `100` | 0.959633 | 4/5 |
| Combined / Ceres | **1.015649** | 1/5 |

| Scene | Combined SSE | Ratio vs original | Ratio vs Ceres |
|---|---:|---:|---:|
| Gendarmenmarkt | 608,218.903 | 0.915454 | 1.062502 |
| Madrid Metropolis | 194,478.818 | 0.873004 | 0.908992 |
| Notre Dame | 4,979,983.980 | 0.943573 | 1.049824 |
| NYC Library | 935,833.533 | 0.960999 | 1.022202 |
| Tower of London | 1,242,853.338 | 0.800021 | 1.042739 |

The combined arm reduces the five-scene Ceres gap from `1.132643x` to
`1.015649x` and restores the strong Tower behavior of regularization-only while
retaining most metric-100 gains elsewhere.

## BAL Sentinels

Complete five-scene cohorts are available for the established additive BAL
control, the 1DSfM-family baseline, regularization `1e-4`, and metric `100`.

| Variant | Geomean SSE / additive BAL control | Wins vs additive | Total time |
|---|---:|---:|---:|
| additive BAL control | 1.000000 | baseline | 188.58 s |
| regularization `1e-4` | **1.002860** | 2/5 | 182.23 s |
| metric `100` | 1.006962 | 1/5 | 186.48 s |
| 1DSfM-family baseline | 1.011925 | 1/5 | 181.25 s |

Regularization `1e-4` and metric `100` improve the specialized SE3 1DSfM
baseline on BAL, but both remain worse than the established additive BAL
control. The dataset-family effect is therefore still present. Metric `50` and
trust diagonal `2e-4` lack complete BAL cohorts because repeated restarts hit
ZeroMQ address conflicts on large sentinels; they are not used for promotion.

The exact combined arm is `0.997111x` the specialized 1DSfM-family baseline but
`1.009001x` the additive BAL control. It is also `1.006124x`
regularization-only and `1.002025x` metric-100-only on BAL. Thus the interaction
is positive for hard 1DSfM and negative for BAL transfer.

### Combined Arm Per BAL Scene

| Scene | Additive SSE | Combined SSE | Change | Additive RMSE | Combined RMSE | Additive time | Combined time |
|---|---:|---:|---:|---:|---:|---:|---:|
| 52 | 472,512.976 | 484,851.844 | +2.6113% | 1.166632 | 1.181766 | 11.08 s | 11.90 s |
| 245 | 1,747,760.754 | 1,770,617.886 | +1.3078% | 1.265470 | 1.273718 | 31.81 s | 32.53 s |
| 394 | 602,287.237 | 601,943.621 | -0.0571% | 1.061611 | 1.061308 | 18.56 s | 19.52 s |
| 871 | 3,459,869.059 | 3,485,258.952 | +0.7338% | 1.114400 | 1.118481 | 105.21 s | 90.60 s |
| 1723 | 764,624.624 | 764,083.756 | -0.0707% | 1.061401 | 1.061025 | 21.93 s | 26.29 s |

The `0.9001%` aggregate regression is therefore not a uniform sub-one-percent
bound: scene 52 regresses `2.61%` and scene 245 regresses `1.31%`. Combined
aggregate execution time is `0.9590x` additive control. This five-scene result
is promising enough to test as one universal setting, but not enough to replace
the BAL default without the complete 29-scene suite.

### Complete 29-Scene BAL Validation

The exact combined arm completed all 29 BAL scenes at K24/I90. Relative to the
authoritative additive coordinator baseline:

- geometric-mean SSE ratio: `1.004063` (`+0.406%`);
- wins/losses: `11/18`;
- aggregate execution-time ratio: `0.947850` (`5.2%` faster);
- geometric-mean per-scene time ratio: `1.072518`.

The aggregate and geometric time ratios differ because the candidate is faster
on several of the largest scenes but slower on many small scenes.

| Scene | Additive SSE | Combined SSE | Change | Additive time | Combined time |
|---:|---:|---:|---:|---:|---:|
| 49 | 26,724.646 | 26,703.383 | -0.0796% | 5.14 s | 6.40 s |
| 52 | 471,962.332 | 484,851.844 | +2.7310% | 13.39 s | 14.43 s |
| 88 | 571,163.847 | 572,911.053 | +0.3059% | 12.64 s | 13.83 s |
| 89 | 569,732.381 | 572,821.511 | +0.5422% | 14.69 s | 21.33 s |
| 126 | 187,489.790 | 188,846.599 | +0.7237% | 8.01 s | 10.20 s |
| 135 | 728,651.076 | 732,504.872 | +0.5289% | 19.28 s | 17.39 s |
| 142 | 541,785.857 | 544,375.839 | +0.4780% | 17.94 s | 17.85 s |
| 173 | 515,027.070 | 517,587.888 | +0.4972% | 21.96 s | 20.41 s |
| 245 | 1,779,288.461 | 1,770,617.886 | -0.4873% | 28.95 s | 32.64 s |
| 253 | 686,891.381 | 689,820.081 | +0.4264% | 37.99 s | 29.96 s |
| 257 | 201,127.749 | 199,825.780 | -0.6473% | 8.73 s | 10.40 s |
| 287 | 705,798.685 | 710,458.892 | +0.6603% | 37.51 s | 32.18 s |
| 308 | 780,911.602 | 785,554.899 | +0.5946% | 41.26 s | 35.84 s |
| 356 | 1,017,595.243 | 1,038,907.601 | +2.0944% | 45.50 s | 41.21 s |
| 394 | 603,714.750 | 601,943.621 | -0.2934% | 17.64 s | 19.79 s |
| 427 | 2,118,102.080 | 2,114,675.279 | -0.1618% | 41.65 s | 50.31 s |
| 646 | 361,159.264 | 360,945.178 | -0.0593% | 11.51 s | 15.28 s |
| 744 | 3,064,672.876 | 3,088,667.668 | +0.7829% | 84.61 s | 100.72 s |
| 783 | 396,283.971 | 396,150.609 | -0.0337% | 13.49 s | 17.19 s |
| 871 | 3,457,482.503 | 3,485,258.952 | +0.8034% | 102.98 s | 92.18 s |
| 931 | 503,718.725 | 503,609.146 | -0.0218% | 15.53 s | 19.29 s |
| 951 | 3,139,819.245 | 3,228,220.028 | +2.8155% | 192.11 s | 126.36 s |
| 961 | 3,174,248.696 | 3,175,287.922 | +0.0327% | 59.84 s | 62.76 s |
| 1064 | 564,227.156 | 564,366.079 | +0.0246% | 15.71 s | 21.28 s |
| 1266 | 663,658.319 | 661,051.897 | -0.3927% | 18.73 s | 24.99 s |
| 1490 | 3,053,841.231 | 3,081,586.539 | +0.9085% | 144.08 s | 151.86 s |
| 1723 | 765,554.882 | 764,083.756 | -0.1922% | 21.87 s | 26.28 s |
| 1778 | 3,312,788.574 | 3,349,580.879 | +1.1106% | 241.92 s | 170.32 s |
| 3068 | 3,306,549.089 | 3,247,522.176 | -1.7852% | 54.83 s | 76.42 s |

The full suite supports a universal setting as a deliberate simplicity
tradeoff: it gives the large 1DSfM gain and slightly lower aggregate BAL time at
the cost of `0.41%` BAL geometric-mean SSE. It is not a Pareto improvement or a
uniform sub-one-percent change; three BAL scenes regress by more than `2%`. The
apparent aggregate time reduction is provisional under the timing caveat above.

## All-15 1DSfM Generalization

The provisional metric `100`, regularization `1e-4` pair was frozen after the
hard-five study and compared with the original metric `25`, regularization
`5e-5` setting on the other ten scenes at I1000. Across all 15 scenes:

- geometric-mean SSE ratio: `0.946879` (`5.31%` reduction);
- wins/losses: 13/2;
- hard rejections: 1696 to 1318 (`-22.3%`);
- nominal fallbacks: 826 to 889 (`+7.6%`);
- proximal oracle calls: 25,807 to 28,100 (`+8.9%`).

The two endpoint regressions are Piazza del Popolo (`+15.5%`) and Vienna
Cathedral (`+1.37%`). The apparent aggregate runtime ratio is `1.196x`, heavily
influenced by a `10.9x` Trafalgar timing outlier on the shared machine, and is
not treated as authoritative under the timing caveat.

## Focused Metric/Regularization Grid At I200

A 15-point grid was run across all 15 scenes around the provisional pair. It
varied metric scale along `25, 50, 75, 100, 150, 200`, regularization along
`5e-5, 7.5e-5, 1e-4, 1.5e-4, 2e-4`, and selected joint higher points.

Twelve variants completed all 15 scenes. Higher joint settings were unstable:
metric `150` with regularization `1.5e-4` failed on Piccadilly and Vienna;
metric `150` or `200` with regularization `2e-4` failed on Piccadilly. The
provisional `(100, 1e-4)` pair exhausted recovery on Piazza at I148 in this
repeat and is therefore not deterministic enough to retain as default.

| Candidate | SSE ratio | Wins/losses | Worst ratio | Rejections | Fallbacks | Oracle calls |
|---|---:|---:|---:|---:|---:|---:|
| metric `200`, reg `1e-4` | **0.964091** | 13/2 | 1.122447 | **205** | 190 | 5,653 |
| metric `100`, reg `1.5e-4` | 0.965355 | 11/4 | 1.063245 | 244 | 210 | 5,612 |
| metric `100`, reg `2e-4` | 0.967635 | 11/4 | 1.063068 | **200** | 193 | 5,661 |
| metric `75`, reg `1e-4` | 0.970050 | 12/3 | **1.047824** | 291 | 160 | 5,552 |
| metric `50`, reg `1e-4` | 0.976268 | 11/4 | 1.089699 | 303 | 166 | 5,531 |
| original metric `25`, reg `5e-5` | 1.000000 | baseline | 1.000000 | 378 | 129 | 5,421 |

Metric `200`, regularization `1e-4` is the endpoint-quality optimum and reduces
rejections most strongly, but its worst scene regresses `12.2%`. Metric `75`,
regularization `1e-4` is the balanced point: `3.0%` aggregate SSE gain, 12/15
wins, a `4.8%` worst regression, and complete deterministic execution. Both are
being promoted to all-15 I1000 before the default is revised.

## Final I1000 Candidate Validation

Metric `200`, regularization `1e-4` is disqualified: Gendarmenmarkt produced a
non-finite worker update during I1000. Metric `75`, regularization `1e-4`
completed all 15 scenes through I1000 without recovery exhaustion or non-finite
updates.

Relative to the original metric `25`, regularization `5e-5` I1000 trajectories,
metric `75`, regularization `1e-4` achieves:

- geometric-mean SSE ratio `0.972531` (`2.75%` reduction);
- 12 wins and 3 losses;
- worst regression `1.049079x` (Piazza del Popolo);
- hard rejections 1696 to 1540;
- nominal fallbacks 826 to 815;
- proximal oracle calls 25,807 to 27,783.

### Metric-75 I1000 By 1DSfM Scene

The baseline is the original metric `25`, regularization `5e-5` I1000 run.
Times are included for completeness but remain exploratory under the timing
caveat above.

| Scene | Original SSE | Metric-75 SSE | Change | Original time | Metric-75 time |
|---|---:|---:|---:|---:|---:|
| Alamo | 988,589.251 | 973,161.779 | -1.5606% | 276.74 s | 228.68 s |
| Ellis Island | 1,755,679.617 | 1,736,116.377 | -1.1143% | 141.13 s | 124.14 s |
| Gendarmenmarkt | 664,390.318 | 649,038.985 | -2.3106% | 86.33 s | 93.10 s |
| Madrid Metropolis | 222,769.657 | 205,020.060 | -7.9677% | 74.07 s | 78.87 s |
| Montreal Notre Dame | 10,759,122.239 | 10,751,636.299 | -0.0696% | 413.73 s | 284.36 s |
| Notre Dame | 5,277,791.440 | 4,873,415.982 | -7.6618% | 250.09 s | 266.28 s |
| NYC Library | 973,813.495 | 951,982.180 | -2.2418% | 88.28 s | 90.16 s |
| Piazza del Popolo | 1,030,946.987 | 1,081,544.956 | +4.9079% | 169.91 s | 95.03 s |
| Piccadilly | 5,431,099.154 | 5,391,376.705 | -0.7314% | 406.86 s | 281.66 s |
| Roman Forum | 3,383,852.785 | 3,420,872.379 | +1.0940% | 244.52 s | 244.91 s |
| Tower of London | 1,553,525.643 | 1,521,971.460 | -2.0311% | 100.58 s | 97.25 s |
| Trafalgar | 15,007,957.108 | 12,240,428.655 | -18.4404% | 66.40 s | 615.87 s |
| Union Square | 3,775,247.978 | 3,784,353.676 | +0.2412% | 137.86 s | 147.27 s |
| Vienna Cathedral | 4,299,586.236 | 4,295,752.559 | -0.0892% | 266.28 s | 274.51 s |
| Yorkminster | 4,596,710.457 | 4,551,032.017 | -0.9937% | 146.34 s | 141.56 s |
| **Geometric-mean ratio** |  |  | **0.972531x** |  |  |

Against Ceres, its geometric-mean SSE ratio is `0.926199`, with 8 wins and 7
losses. Compared with the provisional metric-100/reg-1e-4 pair, metric 75 is
`1.027091x` in endpoint SSE: it gives up `2.7%` aggregate quality in exchange
for complete repeatability, fewer fallbacks, and no Piazza recovery exhaustion.
The full 29-scene BAL gate for metric 75 is complete.

### Metric-75 Full BAL Gate

Metric `75`, regularization `1e-4` completed all 29 BAL scenes. Relative to the
authoritative additive BAL control:

- geometric-mean SSE ratio: `1.002980` (`+0.298%`);
- wins/losses: 12/17;
- hard rejections: 258 to 31 (`-88.0%`);
- nominal fallbacks: 198 to 36 (`-81.8%`);
- provisional aggregate execution-time ratio: `0.966502`;
- provisional geometric-mean time ratio: `1.112816`.

### Metric-75 By BAL Scene

The baseline is the authoritative additive coordinator result at K24/I90.
Times are included for completeness but remain exploratory under the timing
caveat above.

| Scene | Additive SSE | Metric-75 SSE | Change | Additive time | Metric-75 time |
|---:|---:|---:|---:|---:|---:|
| 49 | 26,724.646 | 26,703.726 | -0.0783% | 5.14 s | 6.18 s |
| 52 | 471,962.332 | 481,743.838 | +2.0725% | 13.39 s | 12.44 s |
| 88 | 571,163.847 | 572,298.357 | +0.1986% | 12.64 s | 15.45 s |
| 89 | 569,732.381 | 572,461.777 | +0.4791% | 14.69 s | 22.34 s |
| 126 | 187,489.790 | 188,733.831 | +0.6635% | 8.01 s | 8.94 s |
| 135 | 728,651.076 | 732,067.678 | +0.4689% | 19.28 s | 18.71 s |
| 142 | 541,785.857 | 544,182.504 | +0.4424% | 17.94 s | 19.55 s |
| 173 | 515,027.070 | 517,533.349 | +0.4866% | 21.96 s | 26.38 s |
| 245 | 1,779,288.461 | 1,767,882.983 | -0.6410% | 28.95 s | 41.60 s |
| 253 | 686,891.381 | 689,358.321 | +0.3591% | 37.99 s | 32.30 s |
| 257 | 201,127.749 | 199,978.883 | -0.5712% | 8.73 s | 11.99 s |
| 287 | 705,798.685 | 710,250.315 | +0.6307% | 37.51 s | 33.76 s |
| 308 | 780,911.602 | 786,561.655 | +0.7235% | 41.26 s | 36.93 s |
| 356 | 1,017,595.243 | 1,021,738.536 | +0.4072% | 45.50 s | 43.84 s |
| 394 | 603,714.750 | 602,863.732 | -0.1410% | 17.64 s | 20.76 s |
| 427 | 2,118,102.080 | 2,115,955.838 | -0.1013% | 41.65 s | 51.52 s |
| 646 | 361,159.264 | 360,928.425 | -0.0639% | 11.51 s | 16.23 s |
| 744 | 3,064,672.876 | 3,087,234.108 | +0.7362% | 84.61 s | 98.57 s |
| 783 | 396,283.971 | 396,122.443 | -0.0408% | 13.49 s | 17.31 s |
| 871 | 3,457,482.503 | 3,465,667.454 | +0.2367% | 102.98 s | 90.71 s |
| 931 | 503,718.725 | 503,608.372 | -0.0219% | 15.53 s | 19.48 s |
| 951 | 3,139,819.245 | 3,218,213.126 | +2.4968% | 192.11 s | 122.89 s |
| 961 | 3,174,248.696 | 3,174,101.770 | -0.0046% | 59.84 s | 62.83 s |
| 1064 | 564,227.156 | 564,254.168 | +0.0048% | 15.71 s | 22.24 s |
| 1266 | 663,658.319 | 661,050.419 | -0.3930% | 18.73 s | 25.75 s |
| 1490 | 3,053,841.231 | 3,081,081.463 | +0.8920% | 144.08 s | 149.47 s |
| 1723 | 765,554.882 | 765,090.192 | -0.0607% | 21.87 s | 27.03 s |
| 1778 | 3,312,788.574 | 3,340,339.213 | +0.8316% | 241.92 s | 169.05 s |
| 3068 | 3,306,549.089 | 3,263,766.254 | -1.2939% | 54.83 s | 80.06 s |
| **Geometric-mean ratio** |  |  | **1.002980x** |  |  |

The worst regressions are scene 951 (`+2.50%`) and scene 52 (`+2.07%`); the
largest gain is scene 3068 (`-1.29%`). Compared directly with the metric-100
pair, metric 75 reaches `0.998922x` geometric SSE and wins 22/29 BAL scenes.
It therefore improves both BAL quality and repeatability while retaining the
1DSfM gain.

## Camera Floor and Trust-Radius Sensitivity

The promoted metric-75/reg-1e-4 method was screened at K24/I200 on the hard
five while varying one worker threshold at a time. Camera diagonal floors from
`1e-54` through `1e-30` and minimum trust radii from `3e-5` through `3e-4`
were exact endpoint and counter no-ops. Raising the minimum radius to `1e-2`
or `1e-1` only affected NYC and did not improve the aggregate. Initial radii
`1`, `3`, `5`, and `30` changed convergence basins but produced inconsistent
scene tradeoffs.

Maximum trust radius was materially active. The balanced I200 cap is `3e3`:

| Maximum radius | Geomean SSE ratio | Wins/losses | Worst ratio | Rejections | Fallbacks |
|---:|---:|---:|---:|---:|---:|
| `3e3` | **0.966527** | 4/1 | **1.048919** | **49** | 42 |
| `1e4` | 0.949392 | 4/1 | 1.082504 | 59 | 47 |
| `3e4` | 0.978075 | 3/2 | 1.075172 | 56 | 53 |
| `1e5` | 0.966913 | 3/2 | 1.059459 | 79 | 47 |
| `1e6` baseline | 1.000000 | baseline | 1.000000 | 94 | **40** |

At I1000 on the same hard five, cap `3e3` remains favorable:

- geometric-mean SSE ratio: `0.959595` (`4.04%` reduction);
- wins/losses: 4/1;
- worst ratio: `1.052318` on NYC Library;
- hard rejections: 521 to 182;
- nominal fallbacks: 276 to 197.

| Scene | Baseline SSE | Max-3e3 SSE | Ratio | Baseline rejections | Max-3e3 rejections |
|---|---:|---:|---:|---:|---:|
| Gendarmenmarkt | 649,038.985 | 588,939.607 | 0.907403 | 98 | 4 |
| Madrid Metropolis | 205,020.060 | 192,800.322 | 0.940397 | 66 | 2 |
| Notre Dame | 4,873,415.982 | 4,700,426.188 | 0.964503 | 78 | 6 |
| NYC Library | 951,982.180 | 1,001,788.080 | 1.052318 | 126 | 104 |
| Tower of London | 1,521,971.460 | 1,429,831.653 | 0.939460 | 153 | 66 |

The frozen ten-scene BAL K24/I90 transfer gate is mildly positive: `0.999640x`
geometric SSE, 4 wins/6 losses, worst ratio `1.004562`, rejections 13 to 4,
and fallbacks 19 to 13. The largest BAL gain is scene 3068 (`0.990660x`); the
largest regression is scene 245 (`1.004562x`). A transient ZeroMQ port-rebind
failure before baseline scene 3068 was resumed successfully and is not a
solver failure.

The remaining-ten I1000 gate also completed. It reaches `0.991500x` geometric
SSE with 9 wins and 1 loss. Combining both cohorts, cap `3e3` achieves:

- all-15 geometric SSE ratio versus the metric-75/max-1e6 baseline: `0.980749`;
- all-15 geometric SSE ratio versus original DRS: `0.953809`;
- all-15 geometric SSE ratio versus Ceres: `0.908369`;
- wins/losses versus max `1e6`: 13/2;
- hard rejections: 1,540 to 994;
- nominal fallbacks: 815 to 624;
- proximal oracle calls: 27,783 to 28,502.

The two regressions are Yorkminster (`1.082109x`) and NYC Library
(`1.052318x`). Cap `3e3` is therefore promoted as the current universal method
parameter: its aggregate and robustness gains survive all 15 1DSfM scenes and
the frozen BAL gate is slightly positive. It remains a tradeoff rather than a
uniform improvement because of those two 1DSfM regressions.

### Full I200 Cap Comparison

A subsequent matched full benchmark compared cap `3e3` directly with `1e4` on
all 15 1DSfM and all 29 BAL scenes, recording cumulative best SSE and timing at
I50/I100/I150/I200. This broader comparison changes the cap recommendation.
Both arms keep camera floor `1e-48` and proposal disagreement scale `0.6`; only
the maximum trust-radius cap changes. All 29 BAL scenes complete I200 in both
arms. In particular, BAL 3068 reaches SSE `3,151,147.653` with cap `3e3` and
`3,157,973.987` with cap `1e4` at I200.

| Family | Checkpoint | SSE ratio 3e3/1e4 | Wins/losses | Optimization-time ratio |
|---|---:|---:|---:|---:|
| 1DSfM | I50 | 0.995128 | 6/9 | 0.997524 |
| 1DSfM | I100 | 1.008407 | 8/7 | 0.995476 |
| 1DSfM | I150 | 1.019274 | 6/9 | 0.986101 |
| 1DSfM | I200 | 1.017034 | 6/8 | 0.990402 |
| BAL | I50 | 1.000627 | 12/17 | 0.985068 |
| BAL | I100 | 1.000667 | 8/21 | 0.979561 |
| BAL | I150 | 1.000357 | 13/16 | 0.979129 |
| BAL | I200 | 1.000501 | 10/19 | 0.980665 |

The 1DSfM I200 comparison covers 14 matched scenes because cap `3e3`
reproducibly stopped Piazza at I150; cap `1e4` completed it through I200. Among
the matched scenes, `1e4` has `1.70%` better geometric SSE. Roman Forum is the
largest cap-3e3 regression (`1.149582x`), while NYC is its largest gain
(`0.968974x`). BAL is effectively tied in quality (`+0.050%` for `3e3`) with an
exploratory `1.93%` aggregate optimization-time advantage for `3e3`.

Use maximum radius `1e4`: it is less restrictive, completes all 15 1DSfM
scenes, and has the stronger full-cohort I200 endpoint. Keep `3e3` as an
ablation showing that aggressive caps can improve selected long-horizon basins
and reduce recovery counts, but not as the universal cap.

This cap table does not contradict the later BAL 3068 failure. That failure
appears only after raising the camera floor from `1e-48` to `3e-10` (and in the
proposal experiments, also varying proposal damping). At floor `1e-48`, changing
the cap between `3e3`, `1e4`, and `1e6` does not create the large 3068
regression.

### Effect-Producing Camera-Floor Screen

The five-scene K24/I200 development screen then raised the relative camera
diagonal floor toward the measured Tower q01. Complete arms ranked as follows
against the `1e-48` baseline:

| Camera floor | Complete scenes | Geomean SSE ratio | Wins/losses | Worst ratio |
|---:|---:|---:|---:|---:|
| `3e-10` | 5/5 | **0.979373** | 2/3 | 1.021987 |
| `1e-48` | 5/5 | 1.000000 | baseline | 1.000000 |
| `1e-8` | 5/5 | 1.007022 |  |  |
| `3e-9` | 5/5 | 1.013143 |  |  |
| `1e-10` | 4/5 | incomplete |  |  |
| `1e-9` | 4/5 | incomplete |  |  |

The incomplete arms initially stopped on Notre Dame with a non-finite camera
back-transform. An isolated floor-by-maximum-radius matrix showed that the
classification is not monotone in the camera floor:

| Camera floor | Max radius `1e4`, original inverse | Max radius `1e6`, original inverse |
|---:|---:|---:|
| `1e-10` | failed | **5,083,124.295 SSE, completed** |
| `3e-10` | **5,095,656.566 SSE, completed** | failed |
| `1e-9` | failed | failed |

Detailed diagnostics show that the camera back-transform is not the source:
its transform is finite and nonsingular, while the incoming camera step is
already NaN. The first Nesterov iterate starts invalid because landmark block
3804 produces a non-finite inverse. The block is finite and positive definite,
with entries around `1e-102`; its explicit 3x3 determinant underflows to
`3.38e-310` even though the mathematical inverse is finite.

`BlockInverse` now retains the direct inverse fast path and retries only a
non-finite result by scaling the block to unit magnitude and solving with LDLT.
With the baseline POBA floor `0`, both disputed max-`1e4` cells then complete:

| Camera floor | Repaired Notre Dame SSE |
|---:|---:|
| `1e-10` | **5,103,900.528** |
| `1e-9` | **5,297,874.778** |

The user was correct that both values work on Notre Dame. The earlier failures
were numerical inversion underflow exposed by their trajectories, not evidence
that either camera floor is invalid. A POBA relative block floor of `1e-12`
also completes `1e-9` at `5,128,504.537` SSE, but it changes the trajectory and
remains a separate untuned parameter. Rerun the five-scene ranking with the
fixed worker before selecting values for the all-15 gate.

### All-15 Camera-Floor Gate

The top two complete hard-five arms, `3e-10` and `1e-8`, were evaluated against
the `1e-48` baseline on all 15 1DSfM scenes at K24/I200. All 45 cases completed.

| Camera floor | Geomean SSE ratio | Wins/losses | Worst ratio | Rejections | Fallbacks | Oracle calls | Optimization s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1e-48` baseline | 1.000000 | baseline | 1.000000 | 232 | 161 | 5,626 | 574.726 |
| `3e-10` | **0.996823** | 6/9 | 1.050433 | 220 | 156 | 5,635 | 571.320 |
| `1e-8` | 1.010222 | 3/12 | 1.080252 | **217** | **155** | 5,632 | 572.793 |

Floor `3e-10` passes the aggregate gate with a `0.318%` geometric SSE
reduction. Its largest gains are NYC Library (`0.910940x`), Tower of London
(`0.955595x`), and Trafalgar (`0.973853x`); its largest regressions are Piazza
del Popolo (`1.050433x`) and Roman Forum (`1.044573x`). Floor `1e-8` fails the
quality gate at `1.010222x`, with 12 losses and a worst ratio of `1.080252x`.

The best evaluated candidate is therefore `3e-10`, but the gain is modest and
comes with a 5.04% worst-scene regression. Promote it when aggregate I200 SSE is
the objective; retain `1e-48` for the conservative universal baseline.

### POBA Block Relative Floor

The next one-parameter hard-five K24/I200 screen held the camera floor at
`3e-10` and varied `BUNDLE_PALM_POBA_BLOCK_RELATIVE_FLOOR`:

| POBA floor | Geomean SSE ratio | Wins/losses | Worst ratio | Rejections | Fallbacks | Optimization s |
|---:|---:|---:|---:|---:|---:|---:|
| `0` baseline | **1.000000** | baseline | 1.000000 | 63 | 50 | **141.656** |
| `1e-14` | 1.010570 | 2/3 | 1.039451 | 63 | 53 | 148.116 |
| `1e-10` | 1.015578 | 1/4 | 1.043446 | **58** | **37** | 146.258 |
| `1e-12` | 1.018297 | 1/4 | 1.046047 | 60 | 64 | 149.922 |

All 20 cases completed. Every positive floor worsens aggregate SSE, so this
parameter is rejected and remains `0`. The `1e-10` arm demonstrates a quality
versus recovery-count tradeoff, but its `1.56%` SSE regression is too large to
promote. The scale-normalized block-inverse fallback remains the numerical fix;
it does not require a positive POBA floor.

### Remaining One-Parameter Overnight Screens

The remaining five candidate parameters were screened sequentially on the hard
five at K24/I200 with camera floor `3e-10` and all other selected settings held
fixed:

| Parameter | Winner | Geomean ratio | Runner-up | Runner-up ratio |
|---|---:|---:|---:|---:|
| Nesterov Schur Lipschitz | `0.9` | **1.000000** | `1.1` | 1.001093 |
| camera trust diagonal scale | `1e-4` | **1.000000** | `3e-5` | 1.001051 |
| block curvature multiplier | `0.4` | **1.000000** | `0.6` | 1.001275 |
| metric-proposal disagreement scale | `0.8` | **0.990073** | `0.6` | 1.000000 |
| local acceptance ratio | `0.9999` | **1.000000** | `1.0` | 1.001756 |

Thus four current values survive their screens. Proposal disagreement `0.8` is
the sole quality winner and was promoted. Robustness failures were also
informative: Lipschitz `0.85` failed Alamo because a landmark eigenvalue reached
`8.94e-312`; curvature `0.8` exhausted recovery on Tower at I71; proposal `1.0`
exhausted recovery on Tower at I170.

A follow-up hard-five test added proposal scale `0.7`, which was not part of the
original current-method grid. The previous baseline was `0.6`. Scale `0.7`
completed all five but reached `1.013251x` geometric SSE (2 wins/3 losses), with
a `1.062754x` Notre Dame regression. It is inferior to `0.6` and `0.8` and was
not promoted to the all-15 gate.

### Proposal Disagreement 0.8 All-15 Gate

Proposal disagreement scale `0.8` completed all 15 1DSfM scenes at K24/I200
against the matched camera-floor-`3e-10`, proposal-`0.6` baseline:

| Proposal scale | Geomean SSE ratio | Wins/losses | Worst ratio | Rejections | Fallbacks | Oracle calls | Optimization s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.6` baseline | 1.000000 | baseline | 1.000000 | 220 | 156 | 5,635 | 571.320 |
| `0.8` | **0.983872** | 10/5 | 1.086993 | **151** | 188 | 5,708 | 607.407 |

The `1.61%` aggregate SSE gain is substantial relative to the other remaining
parameter screens. Piazza improves by `11.18%`, Roman Forum by `8.30%`, Tower
by `4.81%`, and Yorkminster by `4.33%`. Madrid regresses by `8.70%`, NYC by
`3.26%`, and total optimization time rises by `6.3%`. Promote proposal scale
`0.8` for the quality-oriented method; retain `0.6` as the conservative/runtime
alternative.

Proposal scale `0.7` was then run on the same all-15 gate. It reaches
`0.998027x` geometric SSE, 8 wins/7 losses, a `1.092370x` worst ratio, and
`1.011030x` optimization time. Its I50/I100/I150/I200 ratios are `1.026064`,
`1.006303`, `0.997367`, and `0.998027`; `0.8` reaches `1.020014`, `0.992546`,
`0.985081`, and `0.983872`. Thus `0.8` separates clearly after I50.

The existing corrected-DRE grid over `0.6,0.7,0.8` was also tested on the hard
five. It selected the three scales on 664/92/244 of 1,000 iterations but reached
`1.015055x` geometric SSE and required `171.876` optimization seconds, versus
`0.990073x` and `141.070` seconds for fixed `0.8`. This adaptive selector is
rejected. A cheaper hysteretic rule based on the already-computed pre-damping
disagreement ratio is more promising: use `0.8` normally, switch to `0.6` only
for high-disagreement iterations, and test threshold pairs `(0.10,0.03)` and
`(0.20,0.10)`.

The hysteretic rule was subsequently implemented and screened:

| Low/high threshold | Geomean SSE ratio | Wins/losses | Worst ratio | `0.6`/`0.8` selections | Optimization s |
|---:|---:|---:|---:|---:|---:|
| fixed `0.8` | **0.990073** | 3/2 | 1.032597 | 0/1,000 | **141.070** |
| `0.03/0.10` | 1.009347 | 2/3 | 1.101933 | 113/887 | 142.592 |
| `0.10/0.20` | 1.054110 | 1/4 | 1.154653 | 85/915 | 144.748 |

Neither pair passes the hard-five gate. The `0.03/0.10` rule improves Tower
beyond fixed `0.8` but strongly regresses NYC, while the higher pair is broadly
worse. This falsifies the simple global-ratio switching hypothesis: adaptation
needs additional state, such as safeguard/recovery pressure or proposal energy
split by rotation, translation, and intrinsics.

### BAL Proposal-Disagreement Full Benchmark

The missing BAL half was run for proposal scales `0.6`, `0.7`, and `0.8` on all
29 scenes at K24/I90 with camera floor `3e-10`. BAL 3068 exhausted recovery for
all three scales at I30, I33, and I44; the matched I90 aggregate therefore uses
28 scenes.

| Proposal scale | Geomean SSE ratio | Mean ratio | Median ratio | Wins/losses | Worst ratio | Rejections | Fallbacks | Oracle calls | Optimization-time ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.6` | **1.000000** | 1.000000 | 1.000000 | baseline | 1.000000 | 16 | 24 | 4,955 | 1.000000 |
| `0.7` | 1.000262 | 1.000262 | 1.000063 | 12/16 | 1.002353 | 18 | **22** | 4,948 | **0.986408** |
| `0.8` | 1.000620 | 1.000623 | 1.000111 | 12/16 | 1.010597 | 22 | 23 | **4,943** | 1.006734 |

At I30, using all 29 scenes, `0.7/0.6=1.003357` and `0.8/0.6=1.003180`.
At I60 and I90 on the 28 complete scenes, `0.7` reaches `1.000085` and
`1.000262`; `0.8` reaches `1.000634` and `1.000620`. BAL therefore favors the
existing `0.6` quality baseline. Scale `0.7` is a near-tie with a 1.36% timing
advantage, while `0.8` is not transferable as a universal setting.

#### BAL Per-Scene Results

Ratios are relative to proposal scale `0.6`. Rejections, fallbacks, and timing
are listed in `0.6/0.7/0.8` order.

| BAL ID | Iterations | SSE 0.6 | SSE 0.7 | Ratio 0.7 | SSE 0.8 | Ratio 0.8 | Rejections | Fallbacks | Optimization s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 49 | 90/90/90 | 26,765.860 | 26,774.199 | 1.000312 | 26,780.835 | 1.000559 | 0/0/0 | 0/0/0 | 6.01/5.72/5.57 |
| 52 | 90/90/90 | 481,797.241 | 481,695.472 | 0.999789 | 481,550.193 | 0.999487 | 0/0/0 | 0/0/0 | 10.95/11.13/11.25 |
| 88 | 90/90/90 | 598,478.160 | 598,817.745 | 1.000567 | 598,850.903 | 1.000623 | 0/0/0 | 0/0/0 | 12.01/12.07/12.46 |
| 89 | 90/90/90 | 575,540.159 | 575,567.129 | 1.000047 | 575,588.563 | 1.000084 | 0/0/0 | 0/0/0 | 16.13/16.18/17.31 |
| 126 | 90/90/90 | 188,892.657 | 188,911.796 | 1.000101 | 188,919.892 | 1.000144 | 0/0/0 | 0/0/0 | 8.41/8.50/8.31 |
| 135 | 90/90/90 | 759,227.633 | 759,131.693 | 0.999874 | 759,350.436 | 1.000162 | 0/0/0 | 0/0/0 | 16.40/16.53/16.44 |
| 142 | 90/90/90 | 563,861.147 | 564,426.211 | 1.001002 | 564,526.104 | 1.001179 | 0/0/0 | 0/0/0 | 16.89/17.22/17.06 |
| 173 | 90/90/90 | 531,324.735 | 531,046.037 | 0.999475 | 531,211.764 | 0.999787 | 0/0/0 | 0/0/0 | 20.07/19.75/20.10 |
| 245 | 90/90/90 | 1,820,899.124 | 1,817,251.341 | 0.997997 | 1,840,194.443 | 1.010597 | 3/0/6 | 9/7/13 | 30.63/30.76/29.02 |
| 253 | 90/90/90 | 696,740.720 | 698,129.922 | 1.001994 | 697,970.036 | 1.001764 | 0/1/1 | 0/0/0 | 29.29/29.01/29.44 |
| 257 | 90/90/90 | 199,855.283 | 199,878.587 | 1.000117 | 199,882.988 | 1.000139 | 0/0/0 | 0/0/0 | 10.26/10.16/10.23 |
| 287 | 90/90/90 | 713,177.245 | 714,666.514 | 1.002088 | 713,341.147 | 1.000230 | 0/0/0 | 0/0/0 | 32.56/31.09/31.60 |
| 308 | 90/90/90 | 788,968.164 | 790,824.813 | 1.002353 | 791,374.583 | 1.003050 | 0/3/1 | 0/0/0 | 35.37/32.44/33.70 |
| 356 | 90/90/90 | 1,023,111.431 | 1,024,085.362 | 1.000952 | 1,024,818.915 | 1.001669 | 0/0/0 | 0/0/0 | 40.94/39.45/39.96 |
| 394 | 90/90/90 | 600,936.556 | 600,550.289 | 0.999357 | 600,757.102 | 0.999701 | 0/0/0 | 1/1/1 | 20.19/19.64/20.07 |
| 427 | 90/90/90 | 2,151,471.605 | 2,151,148.269 | 0.999850 | 2,151,953.604 | 1.000224 | 0/0/0 | 4/5/0 | 51.17/48.98/50.09 |
| 646 | 90/90/90 | 360,821.099 | 360,871.928 | 1.000141 | 360,883.055 | 1.000172 | 0/0/0 | 0/0/0 | 14.81/14.88/14.73 |
| 744 | 90/90/90 | 3,100,548.925 | 3,100,792.307 | 1.000078 | 3,100,594.022 | 1.000015 | 0/0/0 | 0/1/0 | 88.11/88.59/87.49 |
| 783 | 90/90/90 | 396,133.728 | 396,098.512 | 0.999911 | 396,084.976 | 0.999877 | 0/0/0 | 0/0/0 | 16.90/16.26/16.61 |
| 871 | 90/90/90 | 3,458,632.175 | 3,458,708.236 | 1.000022 | 3,458,628.747 | 0.999999 | 0/0/0 | 4/4/4 | 81.48/81.02/80.81 |
| 931 | 90/90/90 | 503,608.405 | 503,587.083 | 0.999958 | 503,586.544 | 0.999957 | 0/0/0 | 0/0/0 | 18.61/18.52/18.82 |
| 951 | 90/90/90 | 3,382,467.316 | 3,384,340.028 | 1.000554 | 3,378,220.961 | 0.998745 | 0/0/0 | 2/0/0 | 112.74/111.38/114.34 |
| 961 | 90/90/90 | 3,173,510.622 | 3,173,297.879 | 0.999933 | 3,173,327.001 | 0.999942 | 0/0/0 | 0/0/0 | 59.87/57.92/60.83 |
| 1064 | 90/90/90 | 564,276.756 | 564,088.374 | 0.999666 | 564,082.194 | 0.999655 | 0/0/0 | 3/0/0 | 21.23/20.47/21.50 |
| 1266 | 90/90/90 | 661,059.947 | 661,030.407 | 0.999955 | 661,034.407 | 0.999961 | 0/0/0 | 0/0/0 | 24.62/23.78/25.08 |
| 1490 | 90/90/90 | 3,101,342.949 | 3,103,075.610 | 1.000559 | 3,100,580.622 | 0.999754 | 0/0/0 | 0/3/0 | 139.82/136.65/140.22 |
| 1723 | 90/90/90 | 764,763.475 | 765,494.373 | 1.000956 | 765,216.256 | 1.000592 | 13/14/14 | 1/1/5 | 25.25/25.65/25.20 |
| 1778 | 90/90/90 | 3,363,550.180 | 3,362,662.154 | 0.999736 | 3,361,410.809 | 0.999364 | 0/0/0 | 0/0/0 | 149.82/151.68/159.74 |
| 3068 | 30/33/44 | 3,954,623.582 | 4,279,687.472 | 1.082198 | 4,176,564.551 | 1.056122 | 10/11/12 | 0/0/4 | 19.22/21.29/30.09 |

BAL 3068 is not a matched I90 comparison: each arm terminated with
`recovery_exhausted`, and the displayed ratios compare different horizons.
They are included to expose the failure rather than folded into the aggregate.

#### BAL 3068 Failure Diagnosis

This is not a worker crash and is not caused by the scale-normalized LDLT
fallback. All worker logs contain only normal startup, with no non-finite block
inverse, Nesterov initialization, or camera back-transform diagnostics. The
coordinator intentionally terminates after safeguard rejection when recovery
curvature has reached its configured maximum and can no longer increase.

The earlier Metric-75 BAL table and the later proposal experiment do not use
the same camera numerics. The old `3,263,766.254` result used camera floor
`1e-48`, proposal `0.6`, and maximum trust radius `1e6`; it completed I90. The
later partial `3,954,623.582` result used camera floor `3e-10`, proposal `0.6`,
and maximum trust radius `1e4`; it stopped at I30. Therefore those two numbers
must not be interpreted as a proposal-scale regression or an I90-to-I90
comparison.

Matched controls quantify the two changed worker settings:

| Camera floor | Maximum radius | I30 best SSE | I90 best SSE | Ratio vs old I90 |
|---:|---:|---:|---:|---:|
| `1e-48` | `1e6` | 3,408,233.412 | **3,263,766.254** | 1.000000 |
| `1e-48` | `1e4` | 3,410,455.169 | 3,270,255.309 | 1.001988 |
| `3e-10` | `1e4`, curvature cap `256` | 3,954,623.582 | 3,909,069.271 | **1.197105** |

Reducing the maximum radius from `1e6` to `1e4` changes the I90 endpoint by
only `+0.20%`. Raising the camera floor from `1e-48` to `3e-10` changes the
matched cap-`1e4` endpoint by `+19.53%` and creates the repeated safeguard
rejections. The camera floor is the dominant regression source; the curvature
cap only determines whether that poor trajectory stops early or reaches I90.

#### Joint 5+5 Camera-Floor Sweep

Camera-floor evaluation was repeated with the repaired LDLT worker on five
1DSfM scenes and five floor-sensitive BAL scenes (`88`, `142`, `245`, `951`,
and catastrophic sentinel `3068`). Fixed settings were metric `75`, regularizer
`1e-4`, proposal `0.6`, maximum radius `1e4`, POBA floor `0`, and K24. The grid
covered `1e-48`, `1e-14`, `1e-12`, `3e-12`, `1e-11`, `2e-11`, `3e-11`,
`6e-11`, `1e-10`, `2e-10`, and `3e-10`. Promotion remains deferred.

| Camera floor | 1DSfM geomean ratio | 1DSfM complete | BAL geomean ratio | BAL complete |
|---:|---:|---:|---:|---:|
| `1e-48` | 1.000000 | 5/5 | **1.000000** | 5/5 |
| `1e-14` | 0.982448 | 5/5 | 1.014048 | 5/5 |
| `1e-12` | 0.989141 | 5/5 | 1.018339 | 5/5 |
| `3e-12` | 0.992215 | 5/5 | incomplete | 4/5 |
| `1e-11` | incomplete | 4/5 | incomplete | 4/5 |
| `2e-11` | 1.010564 | 5/5 | incomplete | 4/5 |
| `3e-11` | 1.007568 | 5/5 | incomplete | 4/5 |
| `6e-11` | 0.993744 | 5/5 | incomplete | 4/5 |
| `1e-10` | 1.007710 | 5/5 | incomplete | 4/5 |
| `2e-10` | 1.002742 | 5/5 | 1.067533 | 5/5 |
| `3e-10` | **0.979373** | 5/5 | incomplete | 4/5 |

The previously incomplete 1DSfM `1e-10` arm now completes all five scenes with
the fixed worker. The omitted `1e-9` arm is no longer relevant because
`3e-10` is the largest floor retained for consideration. NYC Library at
`1e-11` is a genuine failure (`DRE inputs must be finite`), not an operational
miss.

No raised fixed floor passes both family gates. Floor `3e-10` is the best
1DSfM aggregate (`-2.06%`) but again stops BAL 3068. Floor `1e-14` is the most
balanced raised 1DSfM value (`-1.76%`, 4/5 wins, worst `+0.17%`) but regresses
the sensitive BAL cohort by `1.40%`. BAL scenes `88`, `142`, `245`, and `951`
all select `1e-48`; only 3068 benefits from a raised floor, selecting `1e-12`
at `0.977799x`.

Per-scene best complete 1DSfM floors are heterogeneous:

| Scene | Best floor | Ratio vs `1e-48` |
|---|---:|---:|
| Alamo | `3e-12` | 0.981475 |
| Gendarmenmarkt | `2e-10` | 0.971075 |
| Notre Dame | `1e-14` | 0.971965 |
| NYC Library | `2e-11` | 0.894633 |
| Tower of London | `1e-12` | 0.933325 |

Selective nominal-oracle instrumentation measured the raw camera trust-diagonal
q01 at I1 and I10. Family medians are stable over those iterations: about
`1.13e-9` for 1DSfM and `2.4e-10` for BAL. However, the retrospective ratio of
best floor to baseline q01 is not scene-invariant. For 1DSfM it ranges from
`1.37e-5` (Notre Dame) through roughly `1e-3` (Alamo, NYC, Tower) to `0.299`
(Gendarmenmarkt). Four BAL scenes prefer the effectively zero baseline, while
3068 prefers about `0.0046 * q01`.

Therefore a universal rule such as `floor = alpha * q01` is not supported by
this cohort. I1 and I10 give nearly the same calibration signal, so waiting to
I10 adds little. The complete detailed endpoints, checkpoints, q01 distributions,
floor-hit fractions, and retrospective ratios are in
`benchmark_results/camera_floor_joint_sweep/report.md`. No value is promoted at
this stage.

Additional I1/I10 diagnostics measured q0.1%, bottom-eight means, parameter-group
hits, affected cameras, and hit concentration at a probe floor `1e-12`. The
dominant weak coordinates are usually intrinsics. Helpful probe-floor cases are
NYC (`0.943876x`), Tower (`0.933325x`), and BAL 3068 (`0.977799x`); their hit
shares are respectively 92.9%, 100%, and 86.2% intrinsics, with at most 2.41
hits per affected camera. Harmful BAL cases have denser hits (3.08-6.11 per
affected camera) and 21-44% translation hits.

This is a meaningful correlation but not a complete rule: Notre Dame also has
87.9% intrinsics hits and sparse activity, yet floor `1e-12` regresses it by
6.34%. Per-cluster q0.1% varies strongly, but BAL 245 demonstrates that an
extreme lower tail alone does not justify flooring. The next focused experiment
should therefore compare intrinsics-only flooring and a conservative gate based
on intrinsics share, pose-hit share, and hits per affected camera. Detailed
per-scene/per-cluster tables are in
`benchmark_results/camera_floor_structure_diagnostics/report.md`.

Independent subspace-floor validation confirms that the raised floor should
primarily target intrinsics. Intrinsics-only `3e-10` is the best 1DSfM arm at
`0.972248x` (4/5 wins, worst `1.006014x`), outperforming the all-coordinate
floor. Rotation-only and translation-only `1e-12` both regress 1DSfM.

BAL remains different: four sensitive scenes prefer the baseline. Rotation-only
`1e-12` is effectively neutral (`0.998177x`), while every intrinsics-only arm
exhausts recovery on 3068. A tiny translation support floor resolves that
specific instability: intrinsics `1e-12` plus translation `1e-14` completes all
five BAL scenes, although its BAL aggregate remains worse at `1.018467x`.
Rotation should therefore stay at `1e-48`; translation should generally stay at
`1e-48` except as a possible stability support; intrinsics is the group worth
raising. Full results are in `benchmark_results/camera_subspace_floor_grid/report.md`.

Focused controls separate the two interactions:

| Camera floor | Proposal scale | Maximum curvature | Completed iterations | Termination | Best SSE | Rejections/fallbacks |
|---:|---:|---:|---:|---|---:|---:|
| `1e-48` | `0.6` | `64` | 90 | iteration limit | **3,270,255.309** | 1/8 |
| `1e-48` | `0.7` | `64` | 29 | recovery exhausted | 4,039,933.729 | 9/0 |
| `1e-48` | `0.8` | `64` | 27 | recovery exhausted | 3,954,872.151 | 10/0 |
| `3e-10` | `0.6` | `128` | 83 | recovery exhausted | 3,904,233.085 | 17/0 |
| `3e-10` | `0.6` | `256` | 90 | iteration limit | 3,909,069.271 | 18/0 |

For floor `3e-10`, increasing the curvature cap from `64` to `128` delays the
stop from I30 to I83; cap `256` reaches I90. However, its I90 SSE remains
`19.5%` above the floor-`1e-48` control. Thus the cap causes the early
termination, while the raised camera floor causes the poor trajectory basin.
Separately, proposal `0.7` and `0.8` exhaust recovery even with floor `1e-48`,
so proposal damping is another independent 3068 sensitivity. Keep floor
`1e-48` and proposal `0.6` for BAL; do not use the 1DSfM `3e-10`/`0.8` pair as
a universal setting.

## DABA Clustering

DABA Louvain clustering was already tested and was not rerun:

- the compatible exclusive-owner 1DSfM gate lost all three scenes: Montreal
	`+3.74%`, Piazza `+1.37%`, Yorkminster `+5.25%` SSE at K24/I30;
- on five BAL scenes across K10/K20/K30, native DABA endpoint ownership was
	`6.57x` faster to partition and reduced camera replication, but created
	`1.415x` local residual work and `44.21x` observation-load CV;
- exact DABA duplicated-point ownership requires shared-point consensus absent
	from the current product-space DRS formulation.

## Decision

Use metric `75`, regularization `1e-4`, and maximum trust radius `1e4` as the
provisional universal default. The full I200 comparison favors `1e4` over
`3e3` by `1.70%` geometric SSE on 14 matched 1DSfM scenes, and `1e4` completes
Piazza where `3e3` reproducibly stops at I150. Across all 29 BAL scenes the two
caps are effectively tied (`3e3/1e4 = 1.000501x`). The earlier all-29 BAL
metric-75/max-1e6 result remains `1.002980x` the additive control; the I200 cap
study compares `3e3` and `1e4` directly rather than rerunning that control.
The metric-100 pair has better aggregate 1DSfM endpoints but exhausted Piazza
recovery in a repeat; metric 200 failed Gendarmenmarkt at I1000. The default is
not set in stone: endpoint quality remains the authoritative signal, and speed
claims require paired idle-machine confirmation.
