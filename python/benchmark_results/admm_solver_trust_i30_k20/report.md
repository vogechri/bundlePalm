# Standard Pixel Reprojection Results

Every value below is independently evaluated on the global camera and
landmark state using the standard BAL/Snavely pixel reprojection error.
Pixel SSE is sum(dx^2 + dy^2); mean px is the mean Euclidean pixel error.
ADMM proximal/consensus terms and worker-local surrogate costs are not
reported. DABA's weighted 3D-ray metric is not used.

The table reports the best global pixel-SSE state reached through 30
outer iterations, its matching mean pixel error, and the best iteration.

Completed rows: 25

| Method | Dataset | K | Local LM steps | Alpha | Best iteration <=30 | Best pixel SSE <=30 | Mean px | Pixel SSE at iteration 30 | End / best | Time at 30 s | Communication at 30 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local_ceres_reference | problem-1723-156502-pre.txt | 20 | 1 | 1.0 | 27 | 2,805,429 | 0.7843 | 2,809,739 | 1.00 | 15.954 | 125.019 |
| local_ceres_reference | problem-245-198739-pre.txt | 20 | 1 | 1.0 | 29 | 3,079,341 | 1.0093 | 3,079,341 | 1.00 | 23.877 | 120.575 |
| local_ceres_reference | problem-394-100368-pre.txt | 20 | 1 | 1.0 | 28 | 720,582 | 0.6579 | 720,652 | 1.00 | 12.304 | 92.728 |
| local_ceres_reference | problem-52-64053-pre.txt | 20 | 1 | 1.0 | 29 | 1,840,911 | 1.5878 | 1,840,911 | 1.00 | 6.379 | 37.312 |
| local_ceres_reference | problem-871-527480-pre.txt | 20 | 1 | 1.0 | 24 | 4,933,865 | 0.8756 | 5,037,226 | 1.02 | 79.657 | 321.828 |
| nesterov_daba_tr | problem-1723-156502-pre.txt | 20 | 1 | 1.0 | 1 | 16,667,483 | 1.3518 | 36,821,417,865,468,755,968 | 2209176901357.84 | 12.946 | 197.965 |
| nesterov_daba_tr | problem-245-198739-pre.txt | 20 | 1 | 1.0 | 29 | 2,906,069 | 0.9753 | 2,906,069 | 1.00 | 16.913 | 155.496 |
| nesterov_daba_tr | problem-394-100368-pre.txt | 20 | 1 | 1.0 | 29 | 724,074 | 0.6544 | 724,074 | 1.00 | 9.035 | 154.253 |
| nesterov_daba_tr | problem-52-64053-pre.txt | 20 | 1 | 1.0 | 29 | 1,791,898 | 1.5631 | 1,791,898 | 1.00 | 4.613 | 46.677 |
| nesterov_daba_tr | problem-871-527480-pre.txt | 20 | 1 | 1.0 | 29 | 4,635,978 | 0.8445 | 4,635,978 | 1.00 | 54.418 | 417.744 |
| nesterov_drs_tr | problem-1723-156502-pre.txt | 20 | 1 | 1.0 | 0 | 52,281,868 | 1.7271 | 66,567,532,929,774 | 1273243.20 | 13.396 | 197.963 |
| nesterov_drs_tr | problem-245-198739-pre.txt | 20 | 1 | 1.0 | 29 | 2,936,244 | 0.9810 | 2,936,244 | 1.00 | 17.130 | 155.495 |
| nesterov_drs_tr | problem-394-100368-pre.txt | 20 | 1 | 1.0 | 29 | 720,096 | 0.6543 | 720,096 | 1.00 | 9.384 | 154.252 |
| nesterov_drs_tr | problem-52-64053-pre.txt | 20 | 1 | 1.0 | 29 | 1,867,157 | 1.6046 | 1,867,157 | 1.00 | 4.314 | 46.676 |
| nesterov_drs_tr | problem-871-527480-pre.txt | 20 | 1 | 1.0 | 29 | 4,574,347 | 0.8376 | 4,574,347 | 1.00 | 56.406 | 417.743 |
| pcg_daba_tr | problem-1723-156502-pre.txt | 20 | 1 | 1.0 | -1 | 124,050,155 | 3.8881 | 1,994,926,454 | 16.08 | 14.630 | 197.966 |
| pcg_daba_tr | problem-245-198739-pre.txt | 20 | 1 | 1.0 | 29 | 2,961,929 | 0.9896 | 2,961,929 | 1.00 | 17.623 | 155.497 |
| pcg_daba_tr | problem-394-100368-pre.txt | 20 | 1 | 1.0 | 29 | 741,877 | 0.6642 | 741,877 | 1.00 | 9.351 | 154.254 |
| pcg_daba_tr | problem-52-64053-pre.txt | 20 | 1 | 1.0 | 29 | 1,836,434 | 1.5871 | 1,836,434 | 1.00 | 4.297 | 46.678 |
| pcg_daba_tr | problem-871-527480-pre.txt | 20 | 1 | 1.0 | 29 | 4,729,235 | 0.8560 | 4,729,235 | 1.00 | 52.725 | 417.745 |
| pcg_drs_tr | problem-1723-156502-pre.txt | 20 | 1 | 1.0 | 1 | 22,158,530 | 1.4452 | 1,248,375,263,237 | 56338.36 | 11.546 | 197.965 |
| pcg_drs_tr | problem-245-198739-pre.txt | 20 | 1 | 1.0 | 29 | 2,995,577 | 0.9937 | 2,995,577 | 1.00 | 16.483 | 155.496 |
| pcg_drs_tr | problem-394-100368-pre.txt | 20 | 1 | 1.0 | 29 | 737,202 | 0.6626 | 737,202 | 1.00 | 9.290 | 154.253 |
| pcg_drs_tr | problem-52-64053-pre.txt | 20 | 1 | 1.0 | 29 | 1,915,464 | 1.6289 | 1,915,464 | 1.00 | 4.396 | 46.677 |
| pcg_drs_tr | problem-871-527480-pre.txt | 20 | 1 | 1.0 | 29 | 4,683,011 | 0.8514 | 4,683,011 | 1.00 | 53.935 | 417.744 |
