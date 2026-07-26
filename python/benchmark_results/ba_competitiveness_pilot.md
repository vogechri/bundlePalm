# BA Competitiveness Pilot

DRS source section: `# new full eval`.

This is a diagnostic pilot, not publication evidence. Negative cost gaps favor DRS; time ratios below 1 favor DRS.

| Problem | DRS K / iters | Schur K / iters | DRS cost gap | DRS time ratio | Result |
|---|---:|---:|---:|---:|---|
| 52 | 30 / 90 | 2 / 20 | -1.20% | 7.21x | tradeoff |
| 52 | 30 / 90 | 9 / 20 | -1.20% | 3.71x | tradeoff |
| 52 | 30 / 90 | 19 / 20 | -1.20% | 1.96x | tradeoff |
| 52 | 30 / 90 | 19 / 20 | -1.27% | 1.82x | tradeoff |
| 52 | 30 / 90 | 30 / 20 | -1.20% | 1.36x | tradeoff |
| 52 | 30 / 90 | 60 / 20 | -1.20% | 0.72x | DRS dominates |
| 871 | 30 / 90 | 60 / 20 | -0.33% | 5.82x | tradeoff |
| 3068 | 30 / 90 | 20 / 20 | +0.75% | 23.35x | Schur dominates |
| 3068 | 30 / 90 | 20 / 20 | +0.75% | 22.94x | Schur dominates |

## Claim Status

tradeoff: 6, DRS dominates: 1, Schur dominates: 2.
Current evidence can identify candidate regimes, but cannot support comparative solver claims until the blockers below are resolved.

## Blockers

- objective unverified: 9/9 comparisons
- Schur cross terms unverified: 9/9 comparisons
- iteration budgets differ: 9/9 comparisons
- partition counts differ: 8/9 comparisons
- Communication bytes and peak memory are not recorded for both methods.
- Wall times are single runs on unlike execution paths; repeated matched-hardware runs are required.

## Next Experiment Gate

Run matched 20-iteration, matched-partition cohorts on problems 52, 871, and 3068; independently evaluate every saved state under the same squared reprojection objective; then add byte and peak-memory accounting before interpreting wall time.
