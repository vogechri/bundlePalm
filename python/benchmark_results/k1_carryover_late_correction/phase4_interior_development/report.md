# Shared-Fixed Interior Trial Development Gate

| Family | SSE/control | Summed | W/L | Worst | Time/control |
|---|---:|---:|---:|---:|---:|
| 1dsfm | 0.921785057 | 0.942959309 | 5/1 | 1.000900978 | 1.133425 |
| bal | 1.071668367 | 1.102603988 | 2/3 | 1.353559911 | 0.930854 |

Accepted/rejected local trials: `13782/90`; accepted fraction `0.993512`; maximum shared-camera change `0.0`; summed local cost ratio `0.971187050`.

| Scene | SSE/control | Time/control | Completion |
|---|---:|---:|---|
| gendarmenmarkt | 0.872976000 | 1.132348 | 30/30 iteration_limit |
| piccadilly | 0.858205459 | 1.124390 | 30/30 iteration_limit |
| roman_forum | 0.940223909 | 1.059055 | 30/30 iteration_limit |
| trafalgar | 1.000900978 | 1.145917 | 30/30 iteration_limit |
| union_square | 0.967058071 | 1.104337 | 30/30 iteration_limit |
| vienna_cathedral | 0.899724289 | 1.242474 | 30/30 iteration_limit |
| bal52 | 1.043312936 | 1.130445 | 30/30 iteration_limit |
| bal245 | 0.999858545 | 1.196814 | 30/30 iteration_limit |
| bal1490 | 1.001146899 | 1.179178 | 30/30 iteration_limit |
| bal1778 | 0.999939247 | 1.207905 | 30/30 iteration_limit |
| bal3068 | 1.353559911 | 0.362677 | 15/30 recovery_exhausted |

## Decision

The frozen development gate fails. Do not tune the trial or expand it to held-out/all-29 breadth.

The safety probe separately forced all 48 Roman I2 trials to reject by setting
the maximum backtracking attempts to zero. The rejected arm reproduced endpoint
SSE, all behavioral trajectory fields, saved cameras, and saved points exactly.
Accepted and rejected development trials report maximum shared-camera change
exactly zero. Thus rollback and shared-camera freezing work as designed; the
failure is the outer distributed basin response, not mutation leakage.
