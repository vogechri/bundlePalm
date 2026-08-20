# Late-Correction Phase 2 Restart Oracle

Mature DRS runs to I60, accepts the frozen one correction, then restarts a fresh I30 DRS process from the corrected physical state.

Stage A requests the full I90 horizon but stops after 60 completed ordinary DRS
iterations, so its first 60 trajectory rows match the fresh same-executable I90
control exactly. The correction then uses the frozen damping and acceptance
policy. This avoids the horizon-dependent trajectory mismatch observed when
the process itself requested only 60 iterations.

| Family | Restart/mature I90 | Restart/endpoint one | Restart/I60 corrected |
|---|---:|---:|---:|
| 1dsfm | 0.733780103 | 0.933819317 | 0.893217476 |
| bal | 1.009823225 | 1.010694939 | 0.988324143 |

| Scene | Restart/mature | Restart/endpoint | Restart/I60 corrected |
|---|---:|---:|---:|
| roman_forum | 0.620417206 | 0.912246185 | 0.853595793 |
| trafalgar | 0.867856717 | 0.955902618 | 0.934678294 |
| bal52 | 1.007908434 | 1.008474393 | 0.993659195 |
| bal3068 | 1.011741653 | 1.012920376 | 0.983017737 |

## Decision

Canonical restart produces further descent from the I60-corrected state on all
four scenes. It also beats the terminal I90 correction by `0.933819x` on the
1DSfM pair, but regresses to `1.010695x` on the BAL pair. The frozen Phase 2
cross-family gate therefore fails. Do not tune the correction iteration or
stagnation window and do not proceed to adaptive late scheduling. Retain the
single terminal correction as the common policy; continue only with the
independent shared-fixed interior trial.
