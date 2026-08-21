# K24 Schur Proposal Landmark-Response Cohort

The frozen seven-scene I120 cohort applies the retained I90 shared-camera residual proposal after evaluating each fixed camera scale with the existing three-step rollback-safe landmark response. Selected cameras and refined landmarks commit atomically, followed by canonical product-state restart and selected-only I91 trust rebase. The scale grid and `1e-3` floor are unchanged.

| Scene | Applied | Scale | Delivered/control | Camera-only/control | Rejections new/old |
|---|---|---:|---:|---:|---:|
| Montreal Notre Dame | yes | 1 | 0.571810112 | 0.716422367 | 8/8 |
| Piazza del Popolo | yes | 1 | 0.580405726 | 0.724137901 | 15/14 |
| Roman Forum | yes | 1 | 0.649632297 | 0.797755358 | 7/10 |
| Trafalgar | yes | 1 | 0.895394379 | 0.971947952 | 17/19 |
| Yorkminster | yes | 1 | 0.820583428 | 0.935148057 | 10/10 |
| BAL52 | no | declined | 1.000000000 | 1.000000000 | 0/0 |
| BAL3068 | no | declined | 1.000000000 | 1.000000000 | 3/3 |

The five-scene 1DSfM geometric delivered/control ratio is `0.691763505`, versus `0.822387889` for the retained camera-only proposal; W/T/L is `5/0/0`. BAL52 and BAL3068 decline and reproduce their camera-only trajectories and endpoint states exactly. All seven complete under serialized 14 GiB-capped execution. This passes the frozen transfer gate unchanged and warrants all-15 breadth before BAL29.
