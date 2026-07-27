# Standard Pixel Reprojection Results

Every value below is independently evaluated on the global camera and
landmark state using the standard BAL/Snavely pixel reprojection error.
Pixel SSE is sum(dx^2 + dy^2); mean px is the mean Euclidean pixel error.
ADMM proximal/consensus terms and worker-local surrogate costs are not
reported. DABA's weighted 3D-ray metric is not used.

The table reports the best global pixel-SSE state reached through 12
outer iterations, its matching mean pixel error, and the best iteration.

Completed rows: 5

| Method | Dataset | K | Best iteration <=12 | Best pixel SSE <=12 | Mean px | Pixel SSE at iteration 12 | End / best | Time at 12 s | Communication at 12 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic_jacobi_alpha1 | problem-1723-156502-pre.txt | 20 | 11 | 3,841,564 | 1.0477 | 3,841,564 | 1.00 | 8.939 | 63.575 |
| diagnostic_jacobi_cap1e12 | problem-1723-156502-pre.txt | 20 | 5 | 49,466,531 | 1.7294 | 621,636,798,467 | 12566.82 | 8.829 | 63.575 |
| diagnostic_jacobi_cap1e6 | problem-1723-156502-pre.txt | 20 | -1 | 124,050,155 | 3.8881 | 5,991,263,305,295,969,508,130,816 | 48297104694389528.00 | 8.872 | 63.575 |
| diagnostic_jacobi_cap1e9 | problem-1723-156502-pre.txt | 20 | 4 | 9,883,864 | 1.6841 | 11,564,218,617,706 | 1170009.92 | 8.940 | 63.575 |
| jacobi_scaling | problem-1723-156502-pre.txt | 20 | 11 | 3,568,965 | 1.0062 | 3,568,965 | 1.00 | 8.998 | 63.575 |
