# K24 Landmark-Response DRS Plus One Terminal Correction

The promoted K24/I120 landmark-response proposal component is followed by exactly one frozen safeguarded distributed Schur correction: camera/landmark damping `0.005859375`, `bsr_low_memory`, Jacobi PCG, relative tolerance `1e-6`, and `1e-3` progress floor. No DRS or proposal setting changes.

| Scene | Correction/raw I120 | Corrected/Ceres | Accepted | Attempts | Final damping |
|---|---:|---:|---|---:|---:|
| Roman Forum | 0.951065924 | 1.122558651 | yes | 1 | 0.0029296875 |
| Trafalgar | 0.976092668 | 0.926136664 | yes | 1 | 0.0029296875 |
| BAL52 | 0.999412357 | 0.970114618 | yes | 1 | 0.0029296875 |
| BAL3068 | 0.998706718 | 0.989642398 | yes | 2 | 0.005859375 |

The incremental geomean is `0.963498041x` on Roman/Trafalgar and `0.999059475x` on BAL52/3068. All four correction solves converge and accept. The composition passes the frozen sentinel and should expand unchanged to all-15/all-29 as a separately labeled polished endpoint, not as an additional DRS iteration mechanism.
