# K24 Metric-Contribution Alignment Sentinel

Roman Forum and BAL52 were rerun through I200 with behavior-neutral telemetry.
Both reproduce the reference trajectories, endpoint cameras, and endpoint
points exactly.

| Scene | I | Projected cosine | Raw-copy cosine | Metric-copy cosine | Signed action balance | Positive action fraction | Median best-copy cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| Roman | 90 | 0.063400 | -0.019086 | 0.001940 | 0.017152 | 0.508576 | 0.414702 |
| Roman | 120 | 0.134128 | -0.010830 | 0.006459 | 0.054498 | 0.527249 | 0.362770 |
| Roman | 160 | 0.127310 | -0.016698 | 0.012547 | 0.109107 | 0.554554 | 0.411395 |
| Roman | 200 | 0.043521 | -0.009833 | 0.000798 | 0.006824 | 0.503412 | 0.327837 |
| BAL52 | 90 | 0.554641 | 0.068306 | 0.002160 | 0.056320 | 0.528160 | 0.496302 |
| BAL52 | 120 | 0.412323 | -0.003851 | 0.000969 | 0.060176 | 0.530088 | 0.501421 |
| BAL52 | 160 | 0.480670 | -0.003760 | 0.000526 | 0.022969 | 0.511485 | 0.484377 |
| BAL52 | 200 | 0.128226 | 0.005751 | 0.000234 | 0.019079 | 0.509540 | 0.516466 |

Neither raw reflected copies nor their exact metric contributions contain a
globally coherent Schur direction in either scene. Their signed balances and
best-copy cosines do not separate the poor Roman endpoint from the stronger
BAL52 control. Consensus projection improves alignment in both, so copy-level
signal loss and metric weighting are closed as primary mechanisms.

The next diagnostic must decompose the projected shared-camera direction by
camera. It should distinguish widespread misalignment from a small set of
high-energy cameras dominating the global inner product.