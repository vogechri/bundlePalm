# Standard Pixel Reprojection Results

Every value below is independently evaluated on the global camera and
landmark state using the standard BAL/Snavely pixel reprojection error.
Pixel SSE is sum(dx^2 + dy^2); mean px is the mean Euclidean pixel error.
ADMM proximal/consensus terms and worker-local surrogate costs are not
reported. DABA's weighted 3D-ray metric is not used.

The table reports the best global pixel-SSE state reached through 30
outer iterations, its matching mean pixel error, and the best iteration.

Completed rows: 2

| Method | Dataset | K | Best iteration <=30 | Best pixel SSE <=30 | Mean px | Pixel SSE at iteration 30 | End / best | Time at 30 s | Communication at 30 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic_jacobi_alpha1 | problem-1723-156502-pre.txt | 20 | 27 | 2,805,429 | 0.7843 | 2,809,739 | 1.00 | 16.169 | 125.019 |
| diagnostic_jacobi_alpha1 | problem-52-64053-pre.txt | 20 | 29 | 1,840,911 | 1.5878 | 1,840,911 | 1.00 | 6.142 | 37.312 |
