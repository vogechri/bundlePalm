# ADMM Innovation Matrix

Optimization and evaluation use the standard BAL/Snavely pixel
reprojection objective: sum over observations of dx^2 + dy^2.
Reported SSE is twice the corresponding Ceres half-cost.
DABA's weighted 3D-ray metric is not used in this matrix.

Variants are paired by dataset and cluster count. Final SSE ratios below
1 are better. Time speedup is measured to the baseline's final SSE; a
variant that does not reach that target is not eligible for greedy
combination ranking.

| Variant | Cases | Paired | Targets solved | Geomean final SSE ratio | Geomean speedup | Greedy eligible |
|---|---:|---:|---:|---:|---:|---|
| baseline | 30 | 30 | 30 | 1.0000 | 1.000 | no |

## Completed Rows

| Variant | Dataset | K | Best pixel SSE <=30 | Best pixel SSE <=60 | Mean px | Time s | MiB sent+received |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | problem-1064-113655-pre.txt | 5 | 13,290,537 | 3,557,338 | 0.780638 | 102.290 | 209.973 |
| baseline | problem-1064-113655-pre.txt | 15 | 41,467,351 | 41,467,351 | 0.800084 | 109.628 | 263.618 |
| baseline | problem-1064-113655-pre.txt | 30 | 41,467,351 | 41,467,351 | 3.459896 | 88.917 | 305.279 |
| baseline | problem-1266-132593-pre.txt | 5 | 18,983,789 | 3,546,806 | 0.774745 | 110.740 | 233.492 |
| baseline | problem-1266-132593-pre.txt | 15 | 96,745,297 | 46,759,916 | 0.793587 | 119.994 | 296.257 |
| baseline | problem-1266-132593-pre.txt | 30 | 96,745,297 | 96,745,297 | 0.776620 | 122.920 | 360.095 |
| baseline | problem-1723-156502-pre.txt | 5 | 124,050,155 | 5,052,460 | 0.787146 | 174.349 | 279.749 |
| baseline | problem-1723-156502-pre.txt | 15 | 124,050,155 | 124,050,155 | 3.888063 | 140.073 | 362.271 |
| baseline | problem-1723-156502-pre.txt | 30 | 124,050,155 | 124,050,155 | 3.888063 | 150.402 | 426.279 |
| baseline | problem-245-198739-pre.txt | 5 | 3,004,119 | 3,003,957 | 0.940452 | 113.372 | 277.385 |
| baseline | problem-245-198739-pre.txt | 15 | 3,202,072 | 3,201,565 | 0.980845 | 92.665 | 319.544 |
| baseline | problem-245-198739-pre.txt | 30 | 3,317,309 | 3,316,835 | 1.000949 | 84.185 | 364.349 |
| baseline | problem-3068-310854-pre.txt | 5 | 22,785,128 | 22,785,128 | 1.320250 | 265.495 | 595.953 |
| baseline | problem-3068-310854-pre.txt | 15 | 181,986,684 | 24,255,814 | 1.024211 | 272.994 | 817.239 |
| baseline | problem-3068-310854-pre.txt | 30 | 22,349,327 | 4,463,429 | 0.971522 | 271.948 | 1031.232 |
| baseline | problem-394-100368-pre.txt | 5 | 3,267,767 | 732,791 | 0.652138 | 71.103 | 167.996 |
| baseline | problem-394-100368-pre.txt | 15 | 780,011 | 754,117 | 0.661378 | 57.569 | 239.112 |
| baseline | problem-394-100368-pre.txt | 30 | 9,091,475 | 9,091,475 | 0.673172 | 57.196 | 340.022 |
| baseline | problem-52-64053-pre.txt | 5 | 22,304,126 | 2,761,023 | 1.517423 | 35.002 | 87.258 |
| baseline | problem-52-64053-pre.txt | 15 | 1,965,261 | 1,965,134 | 1.656392 | 25.209 | 97.924 |
| baseline | problem-52-64053-pre.txt | 30 | 2,469,787 | 2,127,703 | 1.733550 | 22.638 | 113.371 |
| baseline | problem-744-543562-pre.txt | 5 | 4,086,934 | 4,078,602 | 0.721086 | 376.731 | 761.641 |
| baseline | problem-744-543562-pre.txt | 15 | 4,289,385 | 4,267,597 | 0.743072 | 326.690 | 853.657 |
| baseline | problem-744-543562-pre.txt | 30 | 4,876,016 | 4,876,016 | 0.759329 | 303.993 | 967.920 |
| baseline | problem-871-527480-pre.txt | 5 | 5,849,707 | 4,326,166 | 0.771979 | 394.630 | 747.596 |
| baseline | problem-871-527480-pre.txt | 15 | 10,682,439 | 4,366,316 | 0.784223 | 351.813 | 855.100 |
| baseline | problem-871-527480-pre.txt | 30 | 5,433,664 | 4,431,205 | 0.796739 | 338.029 | 991.437 |
| baseline | problem-931-102699-pre.txt | 5 | 33,858,245 | 4,259,879 | 0.770622 | 85.699 | 191.548 |
| baseline | problem-931-102699-pre.txt | 15 | 33,858,245 | 1,845,001 | 0.779710 | 80.741 | 240.109 |
| baseline | problem-931-102699-pre.txt | 30 | 33,858,245 | 33,858,245 | 0.773099 | 76.105 | 295.557 |
