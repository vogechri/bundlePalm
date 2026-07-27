# Standard Pixel Reprojection Results

Every value below is independently evaluated on the global camera and
landmark state using the standard BAL/Snavely pixel reprojection error.
Pixel SSE is sum(dx^2 + dy^2); mean px is the mean Euclidean pixel error.
ADMM proximal/consensus terms and worker-local surrogate costs are not
reported. DABA's weighted 3D-ray metric is not used.

The table reports the best global pixel-SSE state reached through 30
outer iterations, its matching mean pixel error, and the best iteration.

Completed rows: 10

| Method | Dataset | K | Best iteration <=30 | Best pixel SSE <=30 | Mean px | Pixel SSE at iteration 30 | End / best | Time at 30 s | Communication at 30 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | problem-1723-156502-pre.txt | 30 | -1 | 124,050,155 | 3.8881 | 56,487,006,829,122,224 | 455356198.46 | 97.130 | 144.915 |
| baseline | problem-245-198739-pre.txt | 30 | 29 | 3,317,309 | 1.0011 | 3,317,309 | 1.00 | 42.100 | 130.149 |
| baseline | problem-394-100368-pre.txt | 30 | -1 | 9,091,475 | 2.2166 | 1,344,748,397,578,174,976 | 147913109868.71 | 34.688 | 115.421 |
| baseline | problem-52-64053-pre.txt | 30 | 12 | 2,469,787 | 1.8737 | 1,633,704,008,885,200 | 661475774.39 | 10.945 | 40.523 |
| baseline | problem-871-527480-pre.txt | 30 | 29 | 5,433,664 | 0.7979 | 5,433,664 | 1.00 | 197.234 | 352.772 |
| jacobi_scaling | problem-1723-156502-pre.txt | 30 | 3 | 34,134,957 | 1.7953 | 1,986,432,842,987,691 | 58193506.53 | 48.090 | 144.915 |
| jacobi_scaling | problem-245-198739-pre.txt | 30 | 29 | 2,975,506 | 0.9906 | 2,975,506 | 1.00 | 33.628 | 130.149 |
| jacobi_scaling | problem-394-100368-pre.txt | 30 | 3 | 2,305,279 | 0.7226 | 117,539,478 | 50.99 | 17.580 | 115.421 |
| jacobi_scaling | problem-52-64053-pre.txt | 30 | 29 | 1,901,916 | 1.6233 | 1,901,916 | 1.00 | 8.232 | 40.523 |
| jacobi_scaling | problem-871-527480-pre.txt | 30 | 22 | 7,031,564 | 0.8591 | 2,389,697,906,458,428,964,864 | 339852972860356.31 | 118.651 | 352.772 |
