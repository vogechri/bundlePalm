# Standard Pixel Reprojection Results

Every value below is independently evaluated on the global camera and
landmark state using the standard BAL/Snavely pixel reprojection error.
Pixel SSE is sum(dx^2 + dy^2); mean px is the mean Euclidean pixel error.
ADMM proximal/consensus terms and worker-local surrogate costs are not
reported. DABA's weighted 3D-ray metric is not used.

The table reports the best global pixel-SSE state reached through 30
outer iterations, its matching mean pixel error, and the best iteration.

Completed rows: 19

| Method | Dataset | K | Local LM steps | Alpha | Best iteration <=30 | Best pixel SSE <=30 | Mean px | Pixel SSE at iteration 30 | End / best | Time at 30 s | Communication at 30 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local_ceres_reference | problem-1723-156502-pre.txt | 20 | 1 | 1.0 | 27 | 2,805,429 | 0.7843 | 2,809,739 | 1.00 | 16.551 | 125.019 |
| local_ceres_reference | problem-245-198739-pre.txt | 20 | 1 | 1.0 | 29 | 3,079,341 | 1.0093 | 3,079,341 | 1.00 | 24.793 | 120.575 |
| local_ceres_reference | problem-394-100368-pre.txt | 20 | 1 | 1.0 | 28 | 720,582 | 0.6579 | 720,652 | 1.00 | 12.671 | 92.728 |
| local_ceres_reference | problem-52-64053-pre.txt | 20 | 1 | 1.0 | 29 | 1,840,911 | 1.5878 | 1,840,911 | 1.00 | 6.742 | 37.312 |
| local_ceres_reference | problem-871-527480-pre.txt | 20 | 1 | 1.0 | 24 | 4,933,865 | 0.8756 | 5,037,226 | 1.02 | 81.721 | 321.828 |
| local_ceres_reference_catastrophic_guard | problem-1723-156502-pre.txt | 20 | 1 | 1.0 | 24 | 2,290,486 | 0.8132 | 2,319,061 | 1.01 | 16.013 | 125.019 |
| local_ceres_reference_catastrophic_guard | problem-245-198739-pre.txt | 20 | 1 | 1.0 | 29 | 3,079,341 | 1.0093 | 3,079,341 | 1.00 | 23.868 | 120.575 |
| local_ceres_reference_catastrophic_guard | problem-394-100368-pre.txt | 20 | 1 | 1.0 | 28 | 720,582 | 0.6579 | 720,652 | 1.00 | 12.332 | 92.728 |
| local_ceres_reference_catastrophic_guard | problem-52-64053-pre.txt | 20 | 1 | 1.0 | 29 | 1,840,911 | 1.5878 | 1,840,911 | 1.00 | 6.225 | 37.312 |
| local_ceres_reference_catastrophic_guard | problem-871-527480-pre.txt | 20 | 1 | 1.0 | 24 | 4,933,865 | 0.8756 | 5,037,226 | 1.02 | 81.796 | 321.828 |
| nesterov_daba_tr | problem-1723-156502-pre.txt | 20 | 1 | 1.0 | 1 | 16,667,483 | 1.3518 | 36,821,417,865,468,755,968 | 2209176901357.84 | 12.913 | 197.965 |
| nesterov_daba_tr | problem-245-198739-pre.txt | 20 | 1 | 1.0 | 29 | 2,906,069 | 0.9753 | 2,906,069 | 1.00 | 16.852 | 155.496 |
| nesterov_daba_tr | problem-394-100368-pre.txt | 20 | 1 | 1.0 | 29 | 724,074 | 0.6544 | 724,074 | 1.00 | 8.916 | 154.253 |
| nesterov_daba_tr | problem-52-64053-pre.txt | 20 | 1 | 1.0 | 29 | 1,791,898 | 1.5631 | 1,791,898 | 1.00 | 4.452 | 46.677 |
| nesterov_daba_tr | problem-871-527480-pre.txt | 20 | 1 | 1.0 | 29 | 4,635,978 | 0.8445 | 4,635,978 | 1.00 | 54.872 | 417.744 |
| nesterov_daba_tr_catastrophic_guard | problem-1723-156502-pre.txt | 20 | 1 | 1.0 | 1 | 16,667,483 | 1.3518 | 30,842,287 | 1.85 | 13.256 | 197.965 |
| nesterov_daba_tr_catastrophic_guard | problem-394-100368-pre.txt | 20 | 1 | 1.0 | 29 | 724,074 | 0.6544 | 724,074 | 1.00 | 8.917 | 154.253 |
| nesterov_daba_tr_catastrophic_guard | problem-52-64053-pre.txt | 20 | 1 | 1.0 | 29 | 1,791,898 | 1.5631 | 1,791,898 | 1.00 | 4.533 | 46.677 |
| nesterov_daba_tr_catastrophic_guard | problem-871-527480-pre.txt | 20 | 1 | 1.0 | 29 | 4,635,978 | 0.8445 | 4,635,978 | 1.00 | 54.758 | 417.744 |
