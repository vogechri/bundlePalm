# DRS Parameter Sensitivity Gate

Fresh K24/I30 runs use the promoted Stage-C C1+C5 stack. Each arm changes exactly one global parameter from the shared control.

| Family | Arm | Geomean SSE/control | Summed SSE/control | W/T/L | Worst | Time/control | Oracles | Accepts | Fallbacks | Step-cap hits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1dsfm | control | 1.000000000 | 1.000000000 | 0/2/0 | 1.000000000 | 1.000000 | 99 | 33 | 4 | 0 |
| 1dsfm | block_2p5em5 | 0.976870337 | 0.902568886 | 1/0/1 | 1.140033385 | 1.007107 | 99 | 29 | 7 | 0 |
| 1dsfm | block_1em4 | 0.937564038 | 0.907587472 | 2/0/0 | 0.994952004 | 0.961189 | 100 | 32 | 5 | 0 |
| 1dsfm | restart_1 | 1.032178669 | 1.043635654 | 0/0/2 | 1.052219741 | 0.940812 | 95 | 31 | 2 | 0 |
| 1dsfm | restart_5 | 1.000000000 | 1.000000000 | 0/2/0 | 1.000000000 | 1.035856 | 99 | 33 | 4 | 0 |
| 1dsfm | cap_3 | 1.005335068 | 1.002313267 | 0/1/1 | 1.010698599 | 1.029838 | 99 | 34 | 3 | 1 |
| 1dsfm | exponent_2 | 1.000000000 | 1.000000000 | 0/2/0 | 1.000000000 | 1.035353 | 99 | 33 | 4 | 0 |
| 1dsfm | exponent_8 | 1.000000000 | 1.000000000 | 0/2/0 | 1.000000000 | 1.031252 | 99 | 33 | 4 | 0 |
| 1dsfm | dre_0p005 | 1.000000000 | 1.000000000 | 0/2/0 | 1.000000000 | 1.022347 | 99 | 33 | 4 | 0 |
| 1dsfm | dre_0p02 | 1.000000000 | 1.000000000 | 0/2/0 | 1.000000000 | 1.019526 | 99 | 33 | 4 | 0 |
| bal | control | 1.000000000 | 1.000000000 | 0/2/0 | 1.000000000 | 1.000000 | 94 | 26 | 3 | 3 |
| bal | block_2p5em5 | 1.018592939 | 1.020302449 | 0/0/2 | 1.036165038 | 0.887867 | 89 | 21 | 2 | 1 |
| bal | block_1em4 | 0.991877078 | 0.991209831 | 2/0/0 | 0.999699158 | 1.100672 | 101 | 34 | 2 | 5 |
| bal | restart_1 | 0.999272148 | 0.999257637 | 2/0/0 | 0.999434331 | 1.050971 | 94 | 28 | 3 | 4 |
| bal | restart_5 | 1.000000000 | 1.000000000 | 0/2/0 | 1.000000000 | 1.021032 | 94 | 26 | 3 | 3 |
| bal | cap_3 | 1.000254281 | 1.000231535 | 0/1/1 | 1.000508626 | 1.118816 | 100 | 34 | 2 | 25 |
| bal | exponent_2 | 0.997276287 | 0.997036378 | 1/1/0 | 1.000000000 | 1.017815 | 96 | 29 | 2 | 3 |
| bal | exponent_8 | 1.000000000 | 1.000000000 | 0/2/0 | 1.000000000 | 1.003732 | 94 | 26 | 3 | 3 |
| bal | dre_0p005 | 1.000000000 | 1.000000000 | 0/2/0 | 1.000000000 | 1.004678 | 94 | 26 | 3 | 3 |
| bal | dre_0p02 | 0.996988486 | 0.996723699 | 1/1/0 | 1.000000000 | 1.002493 | 94 | 26 | 3 | 3 |

This gate is a safety filter, not a promotion cohort. A parameter advances only as one frozen global value to the established six-1DSfM plus five-BAL development gate.

## Decision

Only block regularization `1e-4` passes the four-scene safety filter: it reaches
`0.937564x` control on the 1DSfM pair and `0.991877x` on the BAL pair, winning
all four scenes. It advances unchanged to the frozen development cohort.

Block regularization `2.5e-5`, restart-after `1`, and acceleration cap `3` are
rejected for cross-family loss. Restart-after `5` is trajectory-identical to
the control. Annealing exponents `2/8` and DRE allowances `0.005/0.02` are
mostly trajectory-neutral; the isolated BAL gains do not justify breadth.
