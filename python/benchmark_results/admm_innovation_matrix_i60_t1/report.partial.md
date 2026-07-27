# Standard Pixel Reprojection Results

Every value below is independently evaluated on the global camera and
landmark state using the standard BAL/Snavely pixel reprojection error.
Pixel SSE is sum(dx^2 + dy^2); mean px is the mean Euclidean pixel error.
ADMM proximal/consensus terms and worker-local surrogate costs are not
reported. DABA's weighted 3D-ray metric is not used.

For each checkpoint, the table reports the best global pixel-SSE state
reached through that many outer iterations and its matching mean pixel error.

Completed rows: 16

| Method | Dataset | K | Pixel SSE <=30 | Mean px <=30 | Pixel SSE <=60 | Mean px <=60 | Runtime s | Communication MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | problem-1266-132593-pre.txt | 10 | 96,745,297 | 3.6061 | 96,745,297 | 3.6061 | 120.185 | 171.479 |
| baseline | problem-1266-132593-pre.txt | 30 | 96,745,297 | 3.6061 | 96,745,297 | 3.6061 | 106.893 | 224.375 |
| baseline | problem-1723-156502-pre.txt | 10 | 124,050,155 | 3.8881 | 124,050,155 | 3.8881 | 175.090 | 198.456 |
| baseline | problem-1723-156502-pre.txt | 30 | 124,050,155 | 3.8881 | 124,050,155 | 3.8881 | 124.428 | 265.500 |
| baseline | problem-245-198739-pre.txt | 10 | 3,116,626 | 0.9628 | 3,116,288 | 0.9627 | 79.889 | 189.537 |
| baseline | problem-245-198739-pre.txt | 30 | 3,317,309 | 1.0011 | 3,316,835 | 1.0009 | 61.578 | 230.520 |
| baseline | problem-3068-310854-pre.txt | 10 | 181,986,684 | 5.7518 | 5,023,205 | 1.0347 | 273.179 | 430.655 |
| baseline | problem-3068-310854-pre.txt | 30 | 22,349,327 | 0.9982 | 4,463,429 | 0.9715 | 228.540 | 642.425 |
| baseline | problem-394-100368-pre.txt | 10 | 768,338 | 0.6715 | 744,621 | 0.6578 | 49.653 | 129.571 |
| baseline | problem-394-100368-pre.txt | 30 | 9,091,475 | 2.2166 | 9,091,475 | 2.2166 | 45.222 | 211.679 |
| baseline | problem-52-64053-pre.txt | 10 | 22,304,126 | 4.9411 | 1,894,252 | 1.6235 | 26.392 | 59.082 |
| baseline | problem-52-64053-pre.txt | 30 | 2,469,787 | 1.8737 | 2,127,703 | 1.7336 | 16.090 | 71.744 |
| baseline | problem-744-543562-pre.txt | 10 | 5,693,311 | 0.7871 | 4,197,652 | 0.7333 | 269.102 | 520.177 |
| baseline | problem-744-543562-pre.txt | 30 | 4,876,016 | 0.7912 | 4,876,016 | 0.7912 | 225.394 | 614.162 |
| baseline | problem-871-527480-pre.txt | 10 | 7,779,174 | 0.9276 | 4,283,675 | 0.7798 | 291.280 | 513.422 |
| baseline | problem-871-527480-pre.txt | 30 | 5,433,664 | 0.7979 | 4,431,205 | 0.7967 | 256.116 | 626.486 |
