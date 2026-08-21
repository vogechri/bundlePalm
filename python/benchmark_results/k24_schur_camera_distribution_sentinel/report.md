# K24 Projected-Camera Alignment Sentinel

Roman Forum and BAL52 were rerun through I200 with behavior-neutral projected
camera alignment telemetry. Both reproduce the reference trajectories,
endpoint cameras, and endpoint points exactly.

| Scene | I | Global cosine | Median camera cosine | Positive cameras | Signed action balance | Positive action fraction |
|---|---:|---:|---:|---:|---:|---:|
| Roman | 90 | 0.063400 | 0.283221 | 0.740106 | 0.572812 | 0.786406 |
| Roman | 120 | 0.134127 | 0.219008 | 0.745383 | 0.779166 | 0.889583 |
| Roman | 160 | 0.127315 | 0.222366 | 0.713720 | 0.795961 | 0.897981 |
| Roman | 200 | 0.043603 | 0.206597 | 0.728232 | 0.419285 | 0.709642 |
| BAL52 | 90 | 0.554641 | 0.467625 | 0.923077 | 0.995394 | 0.997697 |
| BAL52 | 120 | 0.412323 | 0.392253 | 0.807692 | 0.961821 | 0.980911 |
| BAL52 | 160 | 0.480670 | 0.452014 | 0.884615 | 0.998482 | 0.999241 |
| BAL52 | 200 | 0.128226 | 0.591273 | 1.000000 | 1.000000 | 1.000000 |

Roman's weak global alignment is broad: roughly one quarter of shared cameras
move against Schur, median camera cosine remains low, and unfavorable action is
material. BAL52 is broadly aligned even though its reflected-copy coherence is
lower than many 1DSfM rows. A small outlier-camera mechanism is rejected.

The next diagnostic should compute, but not apply, a cross-camera coupled
consensus projection from the current reflected copies and distributed Schur
factors. Improvement on Roman with preservation on BAL52 would justify
integrating factorized cross-camera curvature consistently into the DRS local
metric and consensus projection. It must not substitute a Ceres step.