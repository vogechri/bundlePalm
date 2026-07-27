# Standard Pixel Reprojection Results

Every value below is independently evaluated on the global camera and
landmark state using the standard BAL/Snavely pixel reprojection error.
Pixel SSE is sum(dx^2 + dy^2); mean px is the mean Euclidean pixel error.
ADMM proximal/consensus terms and worker-local surrogate costs are not
reported. DABA's weighted 3D-ray metric is not used.

The table reports the best global pixel-SSE state reached through 30
outer iterations, its matching mean pixel error, and the best iteration.

Completed rows: 5

| Method | Dataset | K | Local LM steps | Alpha | Best iteration <=30 | Best pixel SSE <=30 | Mean px | Pixel SSE at iteration 30 | End / best | Time at 30 s | Communication at 30 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nesterov_daba_tr_catastrophic_guard | problem-1723-156502-pre.txt | 20 | 1 | 1.0 | 1 | 16,667,483 | 1.3518 | 30,842,287 | 1.85 | 12.882 | 197.965 |
| nesterov_daba_tr_catastrophic_guard | problem-245-198739-pre.txt | 20 | 1 | 1.0 | 29 | 2,906,069 | 0.9753 | 2,906,069 | 1.00 | 17.103 | 155.496 |
| nesterov_daba_tr_catastrophic_guard | problem-394-100368-pre.txt | 20 | 1 | 1.0 | 29 | 724,074 | 0.6544 | 724,074 | 1.00 | 9.214 | 154.253 |
| nesterov_daba_tr_catastrophic_guard | problem-52-64053-pre.txt | 20 | 1 | 1.0 | 29 | 1,791,898 | 1.5631 | 1,791,898 | 1.00 | 4.399 | 46.677 |
| nesterov_daba_tr_catastrophic_guard | problem-871-527480-pre.txt | 20 | 1 | 1.0 | 29 | 4,635,978 | 0.8445 | 4,635,978 | 1.00 | 54.417 | 417.744 |
