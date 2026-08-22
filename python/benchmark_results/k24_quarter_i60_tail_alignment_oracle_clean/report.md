# K24 Quarter-Damped I60 Tail Alignment

A behavior-neutral full-Schur oracle at I60 is compared with the promoted one-action camera plus three-step landmark-response proposal. All 120 oracle trajectory rows exactly reproduce the ordinary control.

| Scene | Nominal cosine | Nominal/Schur norm | One-action cosine | One-action/Schur norm | One-action/Schur model gain | Immediate proposal ratio | Continuation/proposal | Delivered/Ceres |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tower_of_london | 0.154430718 | 0.031127916 | 0.584828500 | 0.381565519 | 0.623107243 | 0.914278391 | 0.846803398 | 1.790270661 |
| yorkminster | 0.304742619 | 0.015952563 | 0.535583466 | 0.437460242 | 0.727929004 | 0.816113873 | 0.761366238 | 2.219245186 |

The proposal materially repairs direction alignment, but still captures less than half of the converged shared-camera Schur norm and only 62-73% of its predicted camera-model gain. Continuation improves both accepted states further, so the remaining tail is not caused by the I61 restart. The next mechanism must improve cross-camera direction quality without adding frozen Jacobi depth.

Gate status: **passed**.
