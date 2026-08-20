# Mature I90 Endpoint Correction

A single frozen distributed shared-camera Schur correction was applied to the
integrated mature K24/I90 endpoints. The global policy was selected on the
Roman/Trafalgar development pair and then used unchanged on nine held-out
1DSfM scenes and all 29 BAL scenes.

The correction uses camera and landmark damping `0.005859375`, geometric
fallback by doubling after rejection, `bsr_low_memory`, Jacobi
preconditioning, relative PCG tolerance `1e-6`, and physical-SSE/model-gain
acceptance. This is one safeguarded distributed correction, not Ceres/BAE and
not a repeated polishing tail.

| Cohort | Corrected / mature endpoint, geomean | Summed ratio | W/T/L |
|---|---:|---:|---:|
| 1DSfM development 6 | 0.813048282 | -- | 6/0/0 |
| 1DSfM held-out 9 | 0.781399122 | 0.687469911 | 9/0/0 |
| 1DSfM all 15 | 0.793908197 | 0.770181052 | 15/0/0 |
| BAL all 29, corrected / actual pre-correction state | 0.999518398 | 0.999652399 | 26/3/0 |
| BAL all 29, corrected / recorded mature endpoint | 0.999507331 | 0.999624348 | 28/0/1 |

## Reference quality

On all 15 1DSfM scenes, the correction changes the integrated mature endpoint
from `1.596631x` Ceres to `1.267579x` Ceres geometrically and `1.177933x` by
summed SSE. It wins against Ceres on 1/15 scenes.

On all 29 BAL scenes, the recorded mature endpoint is `1.000821880x` Ceres
geometrically and `0.996915443x` by summed SSE. The corrected endpoint is
`1.000328806x` and `0.996540949x`, with Ceres W/T/L `12/0/17`.

## Safety and transfer

All 15 1DSfM cases improve. Trafalgar alone uses fallback, accepting damping
`0.046875`; every accepted PCG solve terminates successfully with relative
residual below `1e-6` and positive accepted model gain.

On BAL, 26 corrections are accepted and improve their actual pre-correction
state; three cases (`951`, `1490`, and `1778`) reject all nonconverged attempts
and preserve the input state. Of the accepted cases, 24 use the frozen initial
damping and BAL `427`/`744` accept fallback damping `0.09375`. The maximum
accepted residual is `9.94e-7`, the minimum accepted damped gain ratio is
`0.8027`, and no nonconverged solve is accepted. Final-Schur time totals
`354.85s` across BAL29, with median `1.34s` and maximum `81.77s`.

Six large BAL checkpoints (`245`, `427`, `744`, `951`, `1490`, and `1778`) are
not reloadable: their saved camera arrays contain values as large as
approximately `1e28`, despite normal recorded endpoint SSE. Those cases were
rerun with the exact mature I90 configuration and the frozen correction applied
in-process. BAL49 first verified this path, reproducing the recorded endpoint
to `9.8e-15` relative and the same corrected SSE. Four recovered starts match
the recorded endpoint within `8.1e-8`; BAL951 and BAL1778 differ by
`1.91e-4` and `1.30e-4`. Therefore correction/prestate is the clean BAL effect,
while correction/recorded-endpoint is reported separately.

## Decision

The global one-correction policy passes the cross-family safety gate: it has no
regression against any actual input state, accepts no nonconverged linear solve,
and uses no scene-specific routing. Its quality effect is large on 1DSfM and
small but positive on BAL. Retain it as the coordinated distributed endpoint
correction candidate and investigate integrating the same safeguarded move into
late DRS iterations rather than adding interior local steps.

Artifacts:

- 1DSfM development: `damping_grid/`
- 1DSfM held-out: `heldout_confirmation/`
- BAL reload campaign: `bal_all29/`
- BAL corrupt-checkpoint recovery: `bal_inprocess_recovery/`
