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
| diagnostic_jacobi_bounded_alpha1 | problem-1723-156502-pre.txt | 20 | -1 | 124,050,155 | 3.8881 | 97,794,224,857,019,040 | 788344239.90 | 9.011 | 63.575 |
| diagnostic_jacobi_bounded_alpha1 | problem-52-64053-pre.txt | 20 | 11 | 1,802,246 | 1.5589 | 1,802,246 | 1.00 | 3.419 | 20.347 |
| jacobi_scaling | problem-1723-156502-pre.txt | 20 | 11 | 3,568,965 | 1.0062 | 3,568,965 | 1.00 | 8.930 | 63.575 |
| jacobi_scaling | problem-52-64053-pre.txt | 20 | 11 | 1,837,247 | 1.5868 | 1,837,247 | 1.00 | 3.448 | 20.347 |
