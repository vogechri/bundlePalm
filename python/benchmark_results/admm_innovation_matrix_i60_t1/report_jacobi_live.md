# Standard Pixel Reprojection Results

Every value below is independently evaluated on the global camera and
landmark state using the standard BAL/Snavely pixel reprojection error.
Pixel SSE is sum(dx^2 + dy^2); mean px is the mean Euclidean pixel error.
ADMM proximal/consensus terms and worker-local surrogate costs are not
reported. DABA's weighted 3D-ray metric is not used.

For each checkpoint, the table reports the best global pixel-SSE state
reached through that many outer iterations and its matching mean pixel error.

Completed rows: 14

| Method | Dataset | K | Pixel SSE <=30 | Mean px <=30 | Pixel SSE <=60 | Mean px <=60 | Runtime s | Communication MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | problem-1064-113655-pre.txt | 30 | 41,467,351 | 3.4599 | 41,467,351 | 3.4599 | 74.736 | 190.284 |
| baseline | problem-1266-132593-pre.txt | 30 | 96,745,297 | 3.6061 | 96,745,297 | 3.6061 | 106.893 | 224.375 |
| baseline | problem-1723-156502-pre.txt | 30 | 124,050,155 | 3.8881 | 124,050,155 | 3.8881 | 124.428 | 265.500 |
| baseline | problem-245-198739-pre.txt | 30 | 3,317,309 | 1.0011 | 3,316,835 | 1.0009 | 61.578 | 230.520 |
| baseline | problem-3068-310854-pre.txt | 30 | 22,349,327 | 0.9982 | 4,463,429 | 0.9715 | 228.540 | 642.425 |
| baseline | problem-394-100368-pre.txt | 30 | 9,091,475 | 2.2166 | 9,091,475 | 2.2166 | 45.222 | 211.679 |
| baseline | problem-52-64053-pre.txt | 30 | 2,469,787 | 1.8737 | 2,127,703 | 1.7336 | 16.090 | 71.744 |
| baseline | problem-744-543562-pre.txt | 30 | 4,876,016 | 0.7912 | 4,876,016 | 0.7912 | 225.394 | 614.162 |
| baseline | problem-871-527480-pre.txt | 30 | 5,433,664 | 0.7979 | 4,431,205 | 0.7967 | 256.116 | 626.486 |
| baseline | problem-931-102699-pre.txt | 30 | 33,858,245 | 3.5038 | 33,858,245 | 3.5038 | 64.990 | 183.954 |
| jacobi_scaling | problem-1723-156502-pre.txt | 30 | 34,134,957 | 1.7953 | 34,134,957 | 1.7953 | 61.456 | 265.500 |
| jacobi_scaling | problem-245-198739-pre.txt | 30 | 2,975,506 | 0.9906 | 2,974,057 | 0.9904 | 53.630 | 230.520 |
| jacobi_scaling | problem-394-100368-pre.txt | 30 | 2,305,279 | 0.7226 | 2,305,279 | 0.7226 | 25.795 | 211.679 |
| jacobi_scaling | problem-52-64053-pre.txt | 30 | 1,901,916 | 1.6233 | 1,894,063 | 1.6193 | 13.237 | 71.744 |
