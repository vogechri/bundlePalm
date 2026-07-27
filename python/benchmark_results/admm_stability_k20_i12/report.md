# Standard Pixel Reprojection Results

Every value below is independently evaluated on the global camera and
landmark state using the standard BAL/Snavely pixel reprojection error.
Pixel SSE is sum(dx^2 + dy^2); mean px is the mean Euclidean pixel error.
ADMM proximal/consensus terms and worker-local surrogate costs are not
reported. DABA's weighted 3D-ray metric is not used.

The table reports the best global pixel-SSE state reached through 12
outer iterations, its matching mean pixel error, and the best iteration.

Completed rows: 10

| Method | Dataset | K | Best iteration <=12 | Best pixel SSE <=12 | Mean px | Pixel SSE at iteration 12 | End / best | Time at 12 s | Communication at 12 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | problem-1723-156502-pre.txt | 20 | -1 | 124,050,155 | 3.8881 | 3,408,050,439,671,874,284,683,264 | 27473165591486476.00 | 77.669 | 63.575 |
| baseline | problem-52-64053-pre.txt | 20 | 6 | 6,416,929 | 2.2324 | 135,494,546,887,626,653,696 | 21115169183301.71 | 6.747 | 20.347 |
| diagnostic_jacobi_bounded | problem-1723-156502-pre.txt | 20 | -1 | 124,050,155 | 3.8881 | 1,401,645,894,125,997,260,800 | 11299025771947.83 | 42.174 | 63.575 |
| diagnostic_jacobi_bounded | problem-52-64053-pre.txt | 20 | 11 | 1,659,748 | 1.4777 | 1,659,748 | 1.00 | 5.922 | 20.347 |
| diagnostic_jacobi_bounded_alpha1 | problem-1723-156502-pre.txt | 20 | -1 | 124,050,155 | 3.8881 | 6,804,156,294,607,140,864 | 54850042832.73 | 38.648 | 63.575 |
| diagnostic_jacobi_bounded_alpha1 | problem-52-64053-pre.txt | 20 | 11 | 1,675,181 | 1.4848 | 1,675,181 | 1.00 | 5.823 | 20.347 |
| diagnostic_jacobi_bounded_fixed | problem-1723-156502-pre.txt | 20 | -1 | 124,050,155 | 3.8881 | 90,270,651,012,827,398,144 | 727694788333.43 | 47.837 | 63.575 |
| diagnostic_jacobi_bounded_fixed | problem-52-64053-pre.txt | 20 | 1 | 4,049,417 | 1.8886 | 22,151,239,695,388,142,206,976 | 5470229316707683.00 | 7.219 | 20.347 |
| jacobi_scaling | problem-1723-156502-pre.txt | 20 | 6 | 18,841,151 | 1.3841 | 794,131,304,568 | 42148.77 | 32.843 | 63.575 |
| jacobi_scaling | problem-52-64053-pre.txt | 20 | 11 | 1,717,390 | 1.5198 | 1,717,390 | 1.00 | 6.725 | 20.347 |
