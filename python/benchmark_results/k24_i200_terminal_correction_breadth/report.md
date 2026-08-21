# Integrated K24/I200 Terminal-Correction Breadth

| Family | Raw I200/I90 | Corrected I200/I90 | Summed | Correction/raw I200 | Corrected I90/Ceres | Corrected I200/Ceres | Ceres wins | W/L | Time s | Max RSS GiB C/W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm | 0.941290950 | 0.959794524 | 0.974472500 | 0.807688850 | 1.209546466 | 1.160916074 | 2/15 | 13/2 | 891.535 | 2.370/2.227 |
| bal | 0.996496300 | 0.996820171 | 0.996893508 | 0.999847726 | 1.000112950 | 0.996932763 | 14/29 | 29/0 | 3644.830 | 12.363/8.494 |

Maximum accepted residual: `9.967e-07`. Damping counts across both arms: `{'0.005859375': 73, '0.01171875': 1, '0.0234375': 5, '0.09375': 2, 'noop': 7}`. Completion failures: `['piazza_del_popolo']`. Gate status: **failed**.

## Decision

The continuation is a strong failure-inclusive quality ceiling but does not
promote as the common method. Corrected I200 improves corrected I90 by
`0.959795x` on 1DSfM and `0.996820x` on BAL, reaching `1.160916x` and
`0.996933x` Ceres. Piazza recovery-exhausts at I188, which violates the frozen
completion gate. Montreal and Yorkminster are the only corrected-I200 1DSfM
losses (`1.025943x` and `1.066442x`) even though their raw I200 states improve.

Do not tune the duration, curvature ceiling, or correction damping from these
tails. Retain I200 plus one correction as a separately labeled diagnostic
quality ceiling. The next justified experiment is behavior-neutral alignment
telemetry comparing the nominal DRS camera direction with the global Schur
direction at fixed I90/I120/I160/I200 checkpoints.
