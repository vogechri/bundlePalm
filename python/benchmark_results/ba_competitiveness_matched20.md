# BA Competitiveness Pilot

DRS source: `benchmark_results/drs_matched20.jsonl`.

This is a diagnostic pilot, not publication evidence. Negative cost gaps favor DRS; time ratios below 1 favor DRS.

| Problem | DRS K / iters | Schur K / iters | DRS cost gap | DRS total ratio | DRS solve ratio | Result |
|---|---:|---:|---:|---:|---:|---|
| 52 | 30 / 20 | 2 / 20 | +4.81% | 3.06x | 2.72x | Schur dominates |
| 52 | 30 / 20 | 9 / 20 | +4.81% | 1.57x | 1.40x | Schur dominates |
| 52 | 30 / 20 | 19 / 20 | +4.81% | 0.83x | 0.74x | tradeoff |
| 52 | 30 / 20 | 19 / 20 | +4.74% | 0.77x | 0.69x | tradeoff |
| 52 | 30 / 20 | 30 / 20 | +4.81% | 0.58x | 0.51x | tradeoff |
| 52 | 30 / 20 | 60 / 20 | +4.81% | 0.31x | 0.27x | tradeoff |
| 871 | 30 / 20 | 60 / 20 | +1.09% | 2.20x | 1.60x | Schur dominates |
| 3068 | 30 / 20 | 20 / 20 | +14.78% | 7.13x | 3.91x | Schur dominates |
| 3068 | 30 / 20 | 20 / 20 | +14.78% | 7.00x | 3.84x | Schur dominates |

## Claim Status

Schur dominates: 5, tradeoff: 4.
Current evidence can identify candidate regimes, but cannot support comparative solver claims until the blockers below are resolved.

## Blockers

- objective unverified: 9/9 comparisons
- Schur cross terms unverified: 9/9 comparisons
- partition counts differ: 8/9 comparisons
- Communication bytes and peak memory are not recorded for both methods.
- Wall times are single runs on unlike execution paths; repeated matched-hardware runs are required.

## Next Experiment Gate

Run matched 20-iteration, matched-partition cohorts on problems 52, 871, and 3068; independently evaluate every saved state under the same squared reprojection objective; then add byte and peak-memory accounting before interpreting wall time.
