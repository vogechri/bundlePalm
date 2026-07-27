# Standard Pixel Reprojection Results

Every value below is independently evaluated on the global camera and
landmark state using the standard BAL/Snavely pixel reprojection error.
Pixel SSE is sum(dx^2 + dy^2); mean px is the mean Euclidean pixel error.
ADMM proximal/consensus terms and worker-local surrogate costs are not
reported. DABA's weighted 3D-ray metric is not used.

The table reports the best global pixel-SSE state reached through 12
outer iterations, its matching mean pixel error, and the best iteration.

Completed rows: 6

| Method | Dataset | K | Best iteration <=12 | Best pixel SSE <=12 | Mean px | Pixel SSE at iteration 12 | End / best | Time at 12 s | Communication at 12 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic_jacobi_alpha1_rho100 | problem-1723-156502-pre.txt | 20 | 11 | 2,589,324 | 0.8616 | 2,589,324 | 1.00 | 8.651 | 63.575 |
| diagnostic_jacobi_alpha1_rho100 | problem-52-64053-pre.txt | 20 | 11 | 8,288,532 | 3.3984 | 8,288,532 | 1.00 | 3.351 | 20.347 |
| diagnostic_jacobi_alpha1_rho30 | problem-1723-156502-pre.txt | 20 | 8 | 3,556,929 | 0.9547 | 5,889,726 | 1.66 | 8.678 | 63.575 |
| diagnostic_jacobi_alpha1_rho30 | problem-52-64053-pre.txt | 20 | 11 | 4,384,957 | 2.5824 | 4,384,957 | 1.00 | 3.368 | 20.347 |
| diagnostic_jacobi_alpha1_rho300 | problem-1723-156502-pre.txt | 20 | 11 | 2,746,462 | 0.8669 | 2,746,462 | 1.00 | 8.656 | 63.575 |
| diagnostic_jacobi_alpha1_rho300 | problem-52-64053-pre.txt | 20 | 11 | 13,162,300 | 4.0801 | 13,162,300 | 1.00 | 3.423 | 20.347 |
