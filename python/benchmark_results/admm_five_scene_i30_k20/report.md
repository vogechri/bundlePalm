# Standard Pixel Reprojection Results

Every value below is independently evaluated on the global camera and
landmark state using the standard BAL/Snavely pixel reprojection error.
Pixel SSE is sum(dx^2 + dy^2); mean px is the mean Euclidean pixel error.
ADMM proximal/consensus terms and worker-local surrogate costs are not
reported. DABA's weighted 3D-ray metric is not used.

The table reports the best global pixel-SSE state reached through 30
outer iterations, its matching mean pixel error, and the best iteration.

Completed rows: 15

| Method | Dataset | K | Local LM steps | Alpha | Best iteration <=30 | Best pixel SSE <=30 | Mean px | Pixel SSE at iteration 30 | End / best | Time at 30 s | Communication at 30 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | problem-1723-156502-pre.txt | 20 | 1 | 1.5 | -1 | 124,050,155 | 3.8881 | 2,997,591,130,546,531,467,264 | 24164348199319.77 | 16.529 | 125.019 |
| baseline | problem-245-198739-pre.txt | 20 | 1 | 1.5 | 29 | 3,390,104 | 1.0242 | 3,390,104 | 1.00 | 32.271 | 120.575 |
| baseline | problem-394-100368-pre.txt | 20 | 1 | 1.5 | 2 | 2,695,707 | 1.0533 | 43,816,337,839,366,987,776 | 16254114821406.95 | 22.108 | 92.728 |
| baseline | problem-52-64053-pre.txt | 20 | 1 | 1.5 | 29 | 2,111,382 | 1.7232 | 2,111,382 | 1.00 | 6.298 | 37.312 |
| baseline | problem-871-527480-pre.txt | 20 | 1 | 1.5 | 28 | 4,703,812 | 0.8317 | 5,185,401 | 1.10 | 85.286 | 321.828 |
| daba_split_penalty_adaptation | problem-1723-156502-pre.txt | 20 | 1 | 1.0 | 27 | 2,805,429 | 0.7843 | 2,809,739 | 1.00 | 15.952 | 125.020 |
| daba_split_penalty_adaptation | problem-245-198739-pre.txt | 20 | 1 | 1.0 | 29 | 3,079,341 | 1.0093 | 3,079,341 | 1.00 | 23.910 | 120.576 |
| daba_split_penalty_adaptation | problem-394-100368-pre.txt | 20 | 1 | 1.0 | 28 | 720,580 | 0.6579 | 720,649 | 1.00 | 12.484 | 92.729 |
| daba_split_penalty_adaptation | problem-52-64053-pre.txt | 20 | 1 | 1.0 | 29 | 1,840,502 | 1.5876 | 1,840,502 | 1.00 | 6.296 | 37.314 |
| daba_split_penalty_adaptation | problem-871-527480-pre.txt | 20 | 1 | 1.0 | 24 | 4,933,865 | 0.8756 | 4,981,989 | 1.01 | 79.642 | 321.829 |
| diagnostic_jacobi_alpha1 | problem-1723-156502-pre.txt | 20 | 1 | 1.0 | 27 | 2,805,429 | 0.7843 | 2,809,739 | 1.00 | 16.288 | 125.019 |
| diagnostic_jacobi_alpha1 | problem-245-198739-pre.txt | 20 | 1 | 1.0 | 29 | 3,079,341 | 1.0093 | 3,079,341 | 1.00 | 24.519 | 120.575 |
| diagnostic_jacobi_alpha1 | problem-394-100368-pre.txt | 20 | 1 | 1.0 | 28 | 720,582 | 0.6579 | 720,652 | 1.00 | 12.551 | 92.728 |
| diagnostic_jacobi_alpha1 | problem-52-64053-pre.txt | 20 | 1 | 1.0 | 29 | 1,840,911 | 1.5878 | 1,840,911 | 1.00 | 6.263 | 37.312 |
| diagnostic_jacobi_alpha1 | problem-871-527480-pre.txt | 20 | 1 | 1.0 | 24 | 4,933,865 | 0.8756 | 5,037,226 | 1.02 | 83.297 | 321.828 |
