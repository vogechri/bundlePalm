# I60+30 Four-Scene Quality Ceiling

Date: 2026-08-12

## Scope

The promoted I60 handoff and all Schur controls are unchanged. Only the maximum
correction budget is extended to 30 for NYC Library, Piazza del Popolo, Roman
Forum, and Tower of London, the four I60+20 scenes with substantial remaining
progress.

| Cap | SSE / base-I200 | SSE / Ceres | Schur seconds |
|---:|---:|---:|---:|
| 16 | 0.821415 | 1.315213 | 20.429 |
| 20 | 0.764555 | 1.224171 | 29.757 |
| 25 | 0.723698 | 1.158751 | 39.696 |
| 30 | 0.706069 | 1.130525 | 47.032 |

Corrections 21--30 improve the four-scene cap-20 endpoint by `0.923502x`
geometrically and `0.930037x` summed SSE in `17.274s` additional Schur work.
Piazza stops at correction 30 under the `1e-3` progress rule, and Roman stops at
23. NYC and Tower remain cap-limited.

| Scene | Final / cap-20 | Final / Ceres | Stop |
|---|---:|---:|---|
| NYC Library | 0.906856 | 1.052492 | cap 30; last gain 0.151% |
| Piazza del Popolo | 0.956134 | 0.840288 | progress at 30 |
| Roman Forum | 0.974984 | 1.158233 | progress at 23 |
| Tower of London | 0.860395 | 1.594694 | cap 30; last gain 0.563% |

## Decision

Deeper polishing remains useful as a targeted quality ceiling, especially for
Tower and NYC, but it is not a new global preset. Roman and Piazza have reached
the frozen progress criterion; NYC is close. Tower remains materially
under-polished even after 30 corrections and is the only scene in this cohort
that justifies a further ceiling extension.
