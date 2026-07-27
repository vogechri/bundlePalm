# Safeguarded Outer Nesterov Overnight Matrix

Date: 2026-07-27

## Protocol

- all 29 top-level BAL problems
- K10, K20, and K30
- 30 outer iterations
- exactly one local Schur-Nesterov nonlinear update
- DABA trust-region policy and one CPU thread per cluster
- initial Jacobi camera substitution, standard Snavely pixel objective
- adaptive scalar ADMM penalty and validated catastrophic rollback
- paired methods:
  - unaccelerated guarded control;
  - outer Nesterov center proposal with relative pixel/augmented-merit
    fallback, `1e4` best-basin cap, momentum restart, and the same hard guard

Every case uses a fresh worker and a one-hour timeout. The matrix is resumable
by result key. All 174 planned cases completed successfully.

## Aggregate Results

| K | Method | Geomean best SSE | Best-SSE wins/ties/losses | Stable endpoints | Total time s | Outer rejections |
|---:|---|---:|---:|---:|---:|---:|
| 10 | Control | 1,726,992 | --- | 25/29 | 683.90 | 38 |
| 10 | Accelerated | **1,633,145** | 23/3/3 | 25/29 | 878.68 | 60 |
| 20 | Control | 2,346,179 | --- | 25/29 | 792.31 | 93 |
| 20 | Accelerated | **2,144,125** | 23/4/2 | 25/29 | 979.78 | 71 |
| 30 | Control | 1,866,188 | --- | 25/29 | 765.64 | 40 |
| 30 | Accelerated | **1,681,732** | 24/2/3 | 26/29 | 979.85 | 64 |

Across all 87 paired configurations, acceleration improves best SSE in 70,
ties in 9, and loses in 8. Median accelerated/control best-SSE ratios are:

- K10: `0.9358` (6.4% median improvement);
- K20: `0.9131` (8.7% median improvement);
- K30: `0.8802` (12.0% median improvement).

The accelerated method attempted 2,156 outer proposals; 114 (5.29%) invoked
the plain fallback. Total runtime is 26.6% higher than the control. The extra
cost comes from fallback local solves and coordinator-side global evaluation.

## Largest Quality Gains

| Scene | K | Accelerated/control best SSE | Improvement |
|---:|---:|---:|---:|
| 173 | 30 | 0.7440 | 25.6% |
| 173 | 20 | 0.7695 | 23.0% |
| 1266 | 30 | 0.7696 | 23.0% |
| 142 | 20 | 0.7701 | 23.0% |
| 52 | 10 | 0.7734 | 22.7% |
| 52 | 20 | 0.7777 | 22.2% |
| 52 | 30 | 0.7786 | 22.1% |
| 135 | 20 | 0.8068 | 19.3% |
| 88 | 30 | 0.8200 | 18.0% |

The largest best-SSE regression is scene 1266/K10 (`+46.8%`). Other notable
regressions are 1723/K30 (`+12.5%`) and 1064/K20 (`+6.0%`). Thus acceleration
is broadly beneficial but not uniformly safe at a fixed 30-iteration budget.

## Endpoint Robustness And Failure Cohort

Stable means endpoint SSE is at most `1.05` times the saved best. Acceleration
does not materially change the stable-case count: 76/87 versus 75/87 for the
control. The same small cohort dominates failures.

### Scene 646: immediate local/outer incompatibility

All six runs (both methods and all K) reject every one of 30 candidates and
never improve initialization. The first K10 candidate jumps from `1.91e7` to
`5.80e13`, followed by candidates up to `1e45` despite penalty doubling.
Acceleration never activates because there are no accepted center updates.
This is a baseline local solve, scaling, partition, or proximal-center failure,
not an acceleration failure.

### Scene 931: accepted-state basin loss

All K values are severely unstable. At K20 the control reaches best SSE
`3.31e7` and later converges in consensus around endpoint `1.46e12`; the
accelerated endpoint is `1.96e12`. Relative acceleration fallback catches many
proposals, but plain fallback steps still leave the good basin. K10/K30 are
more severe, with endpoint/best ratios up to `5.68e51`.

### Scene 1064: partition-sensitive recovery

K10 and K20 have endpoint/best ratios above `2e7` and `5e4`, respectively,
under acceleration. At K30, however, acceleration plus repeated hard recovery
ends near its best (`1.023`), while the control endpoint/best ratio is
`2.01e8`. This is a useful recovery case and a strong target for studying
partition/guard interaction.

### Scene 1266: highly non-monotone and K-sensitive

Acceleration helps best SSE by 23.0% at K30 but ends `2.18e5` times above that
best. K10 regresses by 46.8%; K20 never improves initialization. This scene is
the clearest test for a stateful relative accepted-state safeguard and adaptive
metric relaxation.

### Ladybug-1723

K10 remains gradually divergent for both methods (endpoint/best `3.64e4`). At
K20 acceleration preserves the same best but has a somewhat worse endpoint
(`2.93` versus `1.85`). K30 remains near-best for both, with the accelerated
best 12.5% worse at this budget. The line search prevents catastrophic
accelerated proposals but does not solve plain-fallback basin loss.

### Non-improving controls

Scene 646 stalls at all K. Scene 951/K20 also rejects every iteration and never
improves. These cases should be debugged before increasing iteration count;
more outer iterations only repeat the same failed transition.

## Interpretation

The outer Nesterov proposal is a successful quality accelerator on the broad
cohort: it wins 80.5% of paired best-SSE cases, with larger median gains at
higher K. The relative two-signal fallback is active but inexpensive in count,
rejecting about 5% of proposals.

It is not yet the complete robustness mechanism. The line search protects
against a bad accelerated center, while the plain fallback itself can still be
accepted into a poor basin. The next guard refinement must compare both trials
against an established accepted/best-state reference using a relative physical
objective and a splitting merit. It should not merely tighten the one-step
catastrophic threshold.

The result supports presenting acceleration after full block consensus and
preconditioning, followed by a separate safeguard/restoration component. It
also supplies the failure cohort needed to develop that component without
tuning only on Ladybug-1723.

## Reproducibility

- planned/completed cases: 174/174
- saved states: 174
- every saved state independently reproduces recorded pixel SSE exactly
- result rows: 87 per variant
- artifact size: 548 MiB
- summed per-case elapsed time: 5,305 s
- maximum single-case elapsed time: 167 s (scene 951/K20 accelerated)
- no experiment process remains active

Artifacts include result JSONL, NPZ states, coordinator/worker logs, GNU time
memory records, and `status.tsv`.