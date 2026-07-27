# Standard Pixel Reprojection Results

Every value below is independently evaluated on the global camera and
landmark state using the standard BAL/Snavely pixel reprojection error.
Pixel SSE is sum(dx^2 + dy^2); mean px is the mean Euclidean pixel error.
ADMM proximal/consensus terms and worker-local surrogate costs are not
reported. DABA's weighted 3D-ray metric is not used.

The table reports the best global pixel-SSE state reached through 12
outer iterations, its matching mean pixel error, and the best iteration.

Completed rows: 4

| Method | Dataset | K | Best iteration <=12 | Best pixel SSE <=12 | Mean px | Pixel SSE at iteration 12 | End / best | Time at 12 s | Communication at 12 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic_jacobi_alpha1 | problem-1723-156502-pre.txt | 20 | 11 | 3,841,564 | 1.0477 | 3,841,564 | 1.00 | 9.214 | 63.575 |
| diagnostic_jacobi_alpha1_rho10 | problem-1723-156502-pre.txt | 20 | 11 | 4,491,424 | 0.9474 | 4,491,424 | 1.00 | 8.621 | 63.575 |
| diagnostic_jacobi_alpha1_rho100 | problem-1723-156502-pre.txt | 20 | 11 | 2,589,324 | 0.8616 | 2,589,324 | 1.00 | 8.575 | 63.575 |
| diagnostic_jacobi_alpha1_rho1000 | problem-1723-156502-pre.txt | 20 | 9 | 2,658,484 | 0.9572 | 621,668,666 | 233.84 | 8.558 | 63.575 |
