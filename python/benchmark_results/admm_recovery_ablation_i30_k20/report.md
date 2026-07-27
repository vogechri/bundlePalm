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
| nesterov_daba_tr_catastrophic_guard | problem-1723-156502-pre.txt | 20 | 1 | 1.0 | 1 | 16,667,483 | 1.3518 | 30,842,287 | 1.85 | 12.855 | 197.970 |
| nesterov_daba_tr_catastrophic_guard | problem-245-198739-pre.txt | 20 | 1 | 1.0 | 29 | 2,906,069 | 0.9753 | 2,906,069 | 1.00 | 16.902 | 155.502 |
| nesterov_daba_tr_catastrophic_guard | problem-394-100368-pre.txt | 20 | 1 | 1.0 | 29 | 724,074 | 0.6544 | 724,074 | 1.00 | 9.323 | 154.259 |
| nesterov_daba_tr_catastrophic_guard | problem-52-64053-pre.txt | 20 | 1 | 1.0 | 29 | 1,791,898 | 1.5631 | 1,791,898 | 1.00 | 4.418 | 46.683 |
| nesterov_daba_tr_catastrophic_guard | problem-871-527480-pre.txt | 20 | 1 | 1.0 | 29 | 4,635,978 | 0.8445 | 4,635,978 | 1.00 | 54.877 | 417.749 |
| nesterov_daba_tr_temporary_proximal_guard | problem-1723-156502-pre.txt | 20 | 1 | 1.0 | 1 | 16,667,483 | 1.3518 | 45,652,681 | 2.74 | 13.198 | 197.970 |
| nesterov_daba_tr_temporary_proximal_guard | problem-245-198739-pre.txt | 20 | 1 | 1.0 | 29 | 2,906,069 | 0.9753 | 2,906,069 | 1.00 | 17.110 | 155.502 |
| nesterov_daba_tr_temporary_proximal_guard | problem-394-100368-pre.txt | 20 | 1 | 1.0 | 29 | 724,074 | 0.6544 | 724,074 | 1.00 | 9.148 | 154.259 |
| nesterov_daba_tr_temporary_proximal_guard | problem-52-64053-pre.txt | 20 | 1 | 1.0 | 29 | 1,791,898 | 1.5631 | 1,791,898 | 1.00 | 4.461 | 46.683 |
| nesterov_daba_tr_temporary_proximal_guard | problem-871-527480-pre.txt | 20 | 1 | 1.0 | 29 | 4,635,978 | 0.8445 | 4,635,978 | 1.00 | 55.163 | 417.749 |
| nesterov_daba_tr_trust_region_guard | problem-1723-156502-pre.txt | 20 | 1 | 1.0 | 1 | 16,667,483 | 1.3518 | 253,143,019 | 15.19 | 13.452 | 197.972 |
| nesterov_daba_tr_trust_region_guard | problem-245-198739-pre.txt | 20 | 1 | 1.0 | 29 | 2,906,069 | 0.9753 | 2,906,069 | 1.00 | 16.817 | 155.503 |
| nesterov_daba_tr_trust_region_guard | problem-394-100368-pre.txt | 20 | 1 | 1.0 | 29 | 724,074 | 0.6544 | 724,074 | 1.00 | 9.152 | 154.260 |
| nesterov_daba_tr_trust_region_guard | problem-52-64053-pre.txt | 20 | 1 | 1.0 | 29 | 1,791,898 | 1.5631 | 1,791,898 | 1.00 | 4.377 | 46.684 |
| nesterov_daba_tr_trust_region_guard | problem-871-527480-pre.txt | 20 | 1 | 1.0 | 29 | 4,635,978 | 0.8445 | 4,635,978 | 1.00 | 54.924 | 417.750 |
