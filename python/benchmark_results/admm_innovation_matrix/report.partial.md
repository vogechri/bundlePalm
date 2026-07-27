# ADMM Innovation Matrix

Variants are paired by dataset and cluster count. Final SSE ratios below
1 are better. Time speedup is measured to the baseline's final SSE; a
variant that does not reach that target is not eligible for greedy
combination ranking.

| Variant | Cases | Paired | Targets solved | Geomean final SSE ratio | Geomean speedup | Greedy eligible |
|---|---:|---:|---:|---:|---:|---|
| baseline | 15 | 15 | 15 | 1.0000 | 1.000 | no |

## Completed Rows

| Variant | Dataset | K | SSE | Mean px | Time s | MiB sent+received |
|---|---|---:|---:|---:|---:|---:|
| baseline | problem-1723-156502-pre.txt | 5 | 5052459.816 | 0.787146 | 174.349 | 279.749 |
| baseline | problem-1723-156502-pre.txt | 15 | 124050154.625 | 3.888063 | 140.073 | 362.271 |
| baseline | problem-1723-156502-pre.txt | 30 | 124050154.625 | 3.888063 | 150.402 | 426.279 |
| baseline | problem-245-198739-pre.txt | 5 | 3003957.261 | 0.940452 | 113.372 | 277.385 |
| baseline | problem-245-198739-pre.txt | 15 | 3201565.159 | 0.980845 | 92.665 | 319.544 |
| baseline | problem-245-198739-pre.txt | 30 | 3316834.876 | 1.000949 | 84.185 | 364.349 |
| baseline | problem-394-100368-pre.txt | 5 | 732791.178 | 0.652138 | 71.103 | 167.996 |
| baseline | problem-394-100368-pre.txt | 15 | 754116.606 | 0.661378 | 57.569 | 239.112 |
| baseline | problem-394-100368-pre.txt | 30 | 783804.541 | 0.673172 | 57.196 | 340.022 |
| baseline | problem-52-64053-pre.txt | 5 | 1827204.915 | 1.517423 | 35.002 | 87.258 |
| baseline | problem-52-64053-pre.txt | 15 | 1965133.481 | 1.656392 | 25.209 | 97.924 |
| baseline | problem-52-64053-pre.txt | 30 | 2127702.948 | 1.733550 | 22.638 | 113.371 |
| baseline | problem-871-527480-pre.txt | 5 | 4326165.851 | 0.771979 | 394.630 | 747.596 |
| baseline | problem-871-527480-pre.txt | 15 | 4366316.416 | 0.784223 | 351.813 | 855.100 |
| baseline | problem-871-527480-pre.txt | 30 | 4431203.226 | 0.796739 | 338.029 | 991.437 |
