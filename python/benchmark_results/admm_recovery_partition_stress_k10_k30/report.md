# Standard Pixel Reprojection Results

Every value below is independently evaluated on the global camera and
landmark state using the standard BAL/Snavely pixel reprojection error.
Pixel SSE is sum(dx^2 + dy^2); mean px is the mean Euclidean pixel error.
ADMM proximal/consensus terms and worker-local surrogate costs are not
reported. DABA's weighted 3D-ray metric is not used.

The table reports the best global pixel-SSE state reached through 30
outer iterations, its matching mean pixel error, and the best iteration.

Completed rows: 20

| Method | Dataset | K | Local LM steps | Alpha | Best iteration <=30 | Best pixel SSE <=30 | Mean px | Pixel SSE at iteration 30 | End / best | Time at 30 s | Communication at 30 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nesterov_daba_tr_catastrophic_guard | problem-1723-156502-pre.txt | 10 | 1 | 1.0 | 1 | 21,566,178 | 1.3206 | 785,061,559,103 | 36402.44 | 13.522 | 162.088 |
| nesterov_daba_tr_catastrophic_guard | problem-1723-156502-pre.txt | 30 | 1 | 1.0 | 17 | 5,160,080 | 0.8172 | 6,202,749 | 1.20 | 13.408 | 245.131 |
| nesterov_daba_tr_catastrophic_guard | problem-245-198739-pre.txt | 10 | 1 | 1.0 | 29 | 2,614,943 | 0.9080 | 2,614,943 | 1.00 | 17.338 | 127.510 |
| nesterov_daba_tr_catastrophic_guard | problem-245-198739-pre.txt | 30 | 1 | 1.0 | 29 | 3,055,715 | 1.0065 | 3,055,715 | 1.00 | 17.889 | 178.297 |
| nesterov_daba_tr_catastrophic_guard | problem-394-100368-pre.txt | 10 | 1 | 1.0 | 29 | 695,503 | 0.6354 | 695,503 | 1.00 | 9.041 | 106.372 |
| nesterov_daba_tr_catastrophic_guard | problem-394-100368-pre.txt | 30 | 1 | 1.0 | 29 | 745,344 | 0.6680 | 745,344 | 1.00 | 10.305 | 208.055 |
| nesterov_daba_tr_catastrophic_guard | problem-52-64053-pre.txt | 10 | 1 | 1.0 | 29 | 1,502,636 | 1.4003 | 1,502,636 | 1.00 | 4.412 | 38.656 |
| nesterov_daba_tr_catastrophic_guard | problem-52-64053-pre.txt | 30 | 1 | 1.0 | 29 | 1,970,132 | 1.6600 | 1,970,132 | 1.00 | 4.584 | 54.307 |
| nesterov_daba_tr_catastrophic_guard | problem-871-527480-pre.txt | 10 | 1 | 1.0 | 29 | 4,304,508 | 0.8004 | 4,304,508 | 1.00 | 51.842 | 351.369 |
| nesterov_daba_tr_catastrophic_guard | problem-871-527480-pre.txt | 30 | 1 | 1.0 | 29 | 4,920,265 | 0.8802 | 4,920,265 | 1.00 | 55.085 | 491.634 |
| nesterov_daba_tr_temporary_proximal_guard | problem-1723-156502-pre.txt | 10 | 1 | 1.0 | 1 | 21,566,178 | 1.3206 | 785,061,559,103 | 36402.44 | 13.465 | 162.088 |
| nesterov_daba_tr_temporary_proximal_guard | problem-1723-156502-pre.txt | 30 | 1 | 1.0 | 17 | 5,160,080 | 0.8172 | 6,202,749 | 1.20 | 13.448 | 245.131 |
| nesterov_daba_tr_temporary_proximal_guard | problem-245-198739-pre.txt | 10 | 1 | 1.0 | 29 | 2,614,943 | 0.9080 | 2,614,943 | 1.00 | 17.379 | 127.510 |
| nesterov_daba_tr_temporary_proximal_guard | problem-245-198739-pre.txt | 30 | 1 | 1.0 | 29 | 3,055,715 | 1.0065 | 3,055,715 | 1.00 | 17.512 | 178.297 |
| nesterov_daba_tr_temporary_proximal_guard | problem-394-100368-pre.txt | 10 | 1 | 1.0 | 29 | 695,503 | 0.6354 | 695,503 | 1.00 | 9.112 | 106.372 |
| nesterov_daba_tr_temporary_proximal_guard | problem-394-100368-pre.txt | 30 | 1 | 1.0 | 29 | 745,344 | 0.6680 | 745,344 | 1.00 | 10.158 | 208.055 |
| nesterov_daba_tr_temporary_proximal_guard | problem-52-64053-pre.txt | 10 | 1 | 1.0 | 29 | 1,502,636 | 1.4003 | 1,502,636 | 1.00 | 4.467 | 38.656 |
| nesterov_daba_tr_temporary_proximal_guard | problem-52-64053-pre.txt | 30 | 1 | 1.0 | 29 | 1,970,132 | 1.6600 | 1,970,132 | 1.00 | 4.674 | 54.307 |
| nesterov_daba_tr_temporary_proximal_guard | problem-871-527480-pre.txt | 10 | 1 | 1.0 | 29 | 4,304,508 | 0.8004 | 4,304,508 | 1.00 | 51.936 | 351.369 |
| nesterov_daba_tr_temporary_proximal_guard | problem-871-527480-pre.txt | 30 | 1 | 1.0 | 29 | 4,920,265 | 0.8802 | 4,920,265 | 1.00 | 54.992 | 491.634 |
