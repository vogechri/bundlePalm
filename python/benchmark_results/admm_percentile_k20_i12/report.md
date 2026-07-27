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
| diagnostic_jacobi_alpha1 | problem-1723-156502-pre.txt | 20 | 11 | 3,841,564 | 1.0477 | 3,841,564 | 1.00 | 8.864 | 63.578 |
| diagnostic_jacobi_alpha1 | problem-52-64053-pre.txt | 20 | 11 | 1,851,308 | 1.5929 | 1,851,308 | 1.00 | 3.272 | 20.349 |
| diagnostic_jacobi_percentile_alpha1 | problem-1723-156502-pre.txt | 20 | 3 | 18,800,642 | 1.9291 | 109,637,378,747,328,118,784 | 5831576382181.50 | 8.920 | 63.578 |
| diagnostic_jacobi_percentile_alpha1 | problem-52-64053-pre.txt | 20 | 11 | 1,848,830 | 1.5913 | 1,848,830 | 1.00 | 3.381 | 20.349 |
