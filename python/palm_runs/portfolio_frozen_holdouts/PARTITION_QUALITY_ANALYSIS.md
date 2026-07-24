# Good and Bad Partition Analysis

## Scope

The analysis compares all 40 frozen two-part portfolio candidates on ten BAL
problems at a common 30-epoch horizon. All four candidates on problems 135,
356, and 427 were also audited through epoch 90. Features and costs are ranked
or standardized within each problem so that problem scale cannot create a
spurious correlation.

Exact initial Schur features are available for 36 candidates on nine problems.
Problem 961 was excluded from that feature family because its landmarks induce
87.4 million camera-pair contributions; one landmark is observed by 839
cameras. Topology, balance, and trajectory features still include problem 961.

## Static Evidence

| Criterion | Within-problem Spearman vs. epoch-30 cost | Pair ordering | Oracle hits | Mean regret | Worst regret |
|---|---:|---:|---:|---:|---:|
| Shared-point count, minimize | +0.309 | 66.7% | 6/10 | 0.205% | 0.893% |
| Cut observations, minimize | +0.225 | 65.0% | 6/10 | 0.205% | 0.893% |
| Schur cut fraction, minimize | +0.153 | 63.0% | 6/9 | 0.211% | 0.893% |
| Schur boundary max/mean, minimize | +0.153 | 59.3% | 3/9 | 0.228% | 0.765% |
| Maximum load ratio | -0.281 | 65.0% in the inverse direction | 6/10 | 0.205% | 0.893% |

No static feature is a reliable selector by itself. Lower cut and fewer shared
landmarks are useful weak tendencies, but problems 427, 646, and 961 provide
direct counterexamples where the best candidate cuts more than the worst.

Within the prefiltered 1.05 load cap, better candidates generally used more of
the available imbalance: the best candidate had a higher maximum load ratio
than the worst on 8 of 10 problems. This does not argue for unbounded
imbalance. It means balance should remain a feasibility constraint rather than
the objective minimized after feasibility has been reached.

The maximum degree of a shared landmark is constant across candidates within a
problem and has no selection value. The mean shared-landmark degree moves in
the opposite direction from shared-point count: good candidates tend to expose
fewer, higher-degree landmarks rather than many low-degree landmarks. This
suggests that the number of shared variables matters separately from the number
of cut observations.

## Candidate Static Score

An exploratory rank score combines the three complementary tendencies:

$$
R = \operatorname{rank}(N_{\mathrm{shared}})
  + \operatorname{rank}(-\overline{d}_{\mathrm{shared}})
  + \operatorname{rank}\left(
      \frac{\max_i b_i^{\mathrm{Schur}}}
           {\operatorname{mean}_i b_i^{\mathrm{Schur}}}
    \right).
$$

Here $b_i^{\mathrm{Schur}}$ is the total cut Schur weight incident to camera
$i$. The score prefers fewer shared landmarks, concentrates unavoidable sharing
in higher-degree landmarks, and avoids concentrating numerical boundary load
on one camera.

This score selected the epoch-30 winner on 8 of 9 feasible problems, with
0.059% mean regret and 0.531% worst regret. Leave-one-problem-out rule selection
chose the same score and produced the same result. This remains exploratory:
the feature family was developed after seeing these data, and the three-problem
90-epoch audit yielded only one oracle hit, 0.138% mean regret, and 0.361% worst
regret. It is suitable for shortlisting, not final selection.

## Solver Signal

| Selection horizon | Epoch-30 oracle hits | Mean epoch-30 regret | Worst epoch-30 regret | 90-epoch audit hits | Mean audit regret | Worst audit regret |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4/10 | 0.731% | 4.791% | 0/3 | 0.299% | 0.483% |
| 3 | 3/10 | 0.295% | 0.893% | 0/3 | 0.147% | 0.361% |
| 5 | 3/10 | 0.289% | 0.893% | 1/3 | 0.027% | 0.053% |
| 10 | 3/10 | 0.262% | 1.250% | 0/3 | 0.050% | 0.071% |
| 20 | 8/10 | 0.022% | 0.201% | 1/3 | 0.027% | 0.053% |
| 30 | 10/10 | 0% | 0% | 2/3 | 0.018% | 0.053% |

Twenty epochs is the first consistently strong solver signal. Thirty epochs
does not improve the available 90-epoch audit regret because problem 135 still
changes winner after epoch 30.

These horizons count completed PALM updates. An earlier analysis accidentally
treated trajectory record zero, which stores the initial objective and the
first update's diagnostics, as the result of one completed epoch. The table
above uses record $h$ as the result after $h$ updates.

## Cheap Solver-Informed Score

Exact winner recovery is not the engineering objective. For a selected
candidate $j$ and the best observed candidate in the pool, use

$$
R_j = 100\frac{F_j-\min_i F_i}{\min_i F_i},
$$

and provisionally accept $R_j \le 0.5\%$. Retrospective results at epoch 30 are:

| Selector | Within 0.5% | Mean regret | Worst regret |
|---|---:|---:|---:|
| Static topology: shared points and shared-point degree | 8/10 | 0.205% | 0.893% |
| Static topology plus Schur concentration | 8/9 | 0.059% | 0.531% |
| One-update objective reduction | 7/10 | 0.710% | 4.791% |
| One-update reduction plus model quality | 8/10 | 0.706% | 4.791% |
| Static topology plus one-update reduction | 8/10 | 0.229% | 0.893% |
| Three-update objective | 7/10 | 0.295% | 0.893% |
| Five-update objective | 8/10 | 0.289% | 0.893% |
| Twenty-update objective | 10/10 | 0.022% | 0.201% |

The available solver diagnostics do not improve the cheap decision. A single
update can introduce a 4.791% miss, and combining it with static topology does
not change which problems exceed the 0.5% target. Static-score margin is not a
useful uncertainty trigger either: problem 427 has a strong apparent static
margin but still misses the Schur-composite oracle by 0.531%. The 90-epoch
candidate audit contains only three complete pools; every tested selector stays
within 0.5% there, which is reassuring but not discriminating.

## One-Part Reference

A one-part result is useful only when the numerical solver configuration is
matched. In particular, compare one-part PCG with partitioned PCG, and one-part
Nesterov with partitioned Nesterov, using the same outer accelerator, damping,
preconditioner, inner tolerance and iteration cap, and outer-iteration budget.
The archived frozen candidate pools use inner Nesterov, so they cannot be
bounded by the available standalone CG results. No CG-to-CG one-versus-many
partition conclusion is reported here yet.

The separate standalone full-problem experiment is informative for tuning, not
partition evaluation. It used 400 inner iterations with tolerance $10^{-4}$ for
CG but tolerance $10^{-2}$ for Nesterov. Nesterov was about 15% faster, while CG
reached a lower final cost on 24/28 problems. Median Nesterov excess was only
0.075%, but 7/28 problems exceeded 0.5% and the worst excess was 1.886%. Because
the stopping tolerances differ, these results do not isolate an inherent
CG-versus-Nesterov quality gap.

For the seven cases above 0.5%, all Nesterov runs reached their best value at
the final outer iteration. Problems 89 and 1778 still improved by 0.150% and
0.171%, respectively, from iterations 50 to 60, so a larger outer budget may
help those cases. The other five improved by at most 0.071% over the same
interval. Tightening the Nesterov inner tolerance is therefore the first
controlled test; increase the inner cap only if the tighter runs hit 400, and
increase outer iterations only where the late objective slope remains material.

## Recommended Policy

1. Enforce the existing maximum-load cap; do not continue optimizing balance
   once all candidates are feasible.
2. Generate candidates from multiple seeds and shallow refinement checkpoints.
3. When approximately 0.5% regret is acceptable, select one candidate directly
   with the static composite. Fall back to shared-point count and degree when
   exact Schur concentration is unavailable.
4. Include at most one Schur-refined candidate in research portfolios. Total
   Schur cut alone is too weak and exact construction is expensive on
   high-degree graphs.
5. Do not run one- to five-update probes by default; they did not improve the
   0.5% success rate. Reserve the 20-update portfolio for audits or cases where
   the tighter observed 0.201% worst regret justifies its compute cost.
6. Report regret against the candidate-pool oracle. Add a one-part quality bound
   only when the one- and many-part numerical solver configurations match. Do
   not use exact winner count as the primary success criterion.

## Direct BAE Baseline

The broader systems claim is different from partition-oracle recovery: a fixed,
cheap partitioning policy should preserve the quality of the direct solver at
acceptable overhead. The clean archived comparison uses one predetermined
two-part overlap partition per problem, with 20 refinement passes and no
portfolio-search cost. Partitioned PALM uses outer Nesterov acceleration and
inner Nesterov local solves; direct BAE solves the full problem without PALM.

Against direct BAE CG after 60 iterations, on 10 matched problems:

| PALM budget | Better or equal | Within 0.5% or better | Worse by more than 1% | Median cost change | Aggregate runtime change |
|---:|---:|---:|---:|---:|---:|
| 60 epochs | 5/10 | 8/10 | 1/10 | +0.014% | +52.8% |
| 90 epochs | 7/10 | 8/10 | 1/10 | -0.122% | +124.0% |

The median supports the quality-preservation claim, but the tail does not yet:
problem 356 is 4.433% worse at epoch 60 and 4.085% worse at epoch 90. Problem
126 is the other case above 0.5%, at 0.946% and 0.707%, respectively. Reporting
only a mean would hide this. The defensible statement is therefore:

> Fixed two-part accelerated PALM stayed within 0.5% of direct BAE CG on 8/10
> problems at 60 epochs, with a +0.014% median objective change and +52.8%
> aggregate sequential wall time. One problem exceeded 1%, reaching 4.433%.

The extra 30 PALM epochs improve the median and the number of wins but do not
remove the outlier, while more than doubling aggregate runtime relative to the
direct baseline. A larger outer budget is therefore not a general remedy.

This does not establish a best partitioning algorithm. At the 30-update
horizon, even the best observed candidate from each frozen pool remains more
than 0.5% behind direct CG on 4/10 problems, with problem 356 still 3.724%
behind. That separates two questions: the static selector is often close to the
best candidate generated, but the current candidate family itself does not
always preserve direct-solver quality. Future partition objectives should be
judged first by baseline-relative tail loss, not only cut statistics or regret
within the candidate pool.

Wall times are archived elapsed times and assume the runs were made on the same
workstation. They measure sequential execution; parallel block execution is a
separate scalability result. `analyze_partitioned_baseline.py` regenerates the
full per-problem comparison in `baseline_comparison.json`.

The same claim is desirable for DRS, but the current DRS archive is not suitable
for it. `results_drs.json` contains duplicate exploratory configurations, and
the runs wrote free-form logs rather than machine-readable trajectories with
elapsed time and complete solver metadata. A valid DRS audit needs one frozen
configuration, one result per problem, objective checkpoints, cumulative wall
time, partition ownership, and an explicitly named direct BAE baseline. It
should report the same threshold counts, median and worst objective changes,
and aggregate runtime change as the PALM table above.

## Early Selection Deployment

`palm_partition_portfolio.py` now applies the policy before launching solver
pilots:

1. Reject candidates outside the maximum-load cap.
2. Generate both overlap and analysis-refined candidate families. Analysis
   refinement applies its swaps after the corresponding overlap refinement.
3. Rank the remaining candidates by shared-point count and mean shared-point
   degree, reserving the first slot for the best static candidate.
4. Reserve a slot for the minimum-cut candidate when it is distinct.
5. Fill remaining slots by label-invariant camera-ownership distance.
6. Pilot the shortlist for 20 epochs, reduced from the previous 30-epoch
   default.

In a retrospective test on the frozen four-candidate pools, retaining the best
three candidates by shared-point count and selecting with the epoch-20 cost
matched the full-pool epoch-20 selector: 8/10 epoch-30 winners, 0.019% mean
regret, and 0.174% worst regret. This test does not cover candidates previously
discarded before piloting, so the revised three-slot generation policy should
be audited on future runs rather than treated as established out-of-sample
performance.

## Steering Partition Refinement

The same evidence can define an experimental refinement cost, but the observed
rank score should not be inserted directly into a swap loop. Ranks exist only
relative to a candidate pool, and the Schur concentration term is global and
expensive to update. A more suitable lexicographic objective is

$$
J(P) = \left(
  V_{\mathrm{load}}(P),
  \sum_{p\in S(P)} \frac{1}{d_p},
  \frac{\max_i b_i^{\mathrm{Schur}}(P)}
       {\operatorname{mean}_i b_i^{\mathrm{Schur}}(P)},
  N_{\mathrm{cut}}(P)
\right),
$$

where $V_{\mathrm{load}}$ is zero inside the load cap and positive outside it,
$S(P)$ is the set of shared landmarks, and $d_p$ is landmark degree. The
inverse-degree sum penalizes exposing many low-degree landmarks without using
the unstable mean degree as a standalone scalar. Schur concentration and raw
cut count act as later tie-breakers.

The implemented `analysis` partitioner currently uses the load violation and
inverse-degree shared-landmark terms, with overlap gain and maximum load as
tie-breakers. It reuses the overlap swap data structures and updates only the
landmarks affected by each trial swap. Schur concentration is intentionally not
in the swap objective yet: exact edges remain too expensive on high-degree
problems and the available evidence does not support making that complexity a
default.

## Analysis Refinement Experiment

A matched two-part experiment compared each frozen problem's epoch-20 winning
overlap candidate with the same candidate followed by analysis refinement. Both
methods used the same seed, refinement depth, and solver configuration for 90
epochs. Problems 52 and 142 retain their earlier matched configurations.

| Problem | Shared points | Inverse degree | Epoch-20 cost | Epoch-90 cost |
|---:|---:|---:|---:|---:|
| 52 | -816 | -6.511% | -0.012% | +0.014% |
| 126 | -384 | -1.925% | -0.619% | -0.697% |
| 135 | -2,229 | -4.933% | +0.088% | +0.004% |
| 142 | -2,789 | -5.011% | -1.370% | -1.232% |
| 173 | -632 | -1.022% | -0.091% | +0.066% |
| 253 | -1,003 | -1.077% | +0.040% | +0.006% |
| 356 | -2,072 | -1.772% | -0.386% | +0.162% |
| 394 | -1,261 | -2.616% | -0.126% | +0.017% |
| 427 | -2,423 | -1.829% | +0.615% | -0.088% |
| 646 | -115 | -0.341% | -0.061% | +0.016% |
| 931 | -340 | -0.670% | +0.093% | +0.060% |

Analysis improves its static target on all 11 problems, averaging a 2.519%
reduction in inverse-degree boundary cost. It wins 7/11 comparisons at epoch 20
but only 3/11 at epoch 90. The mean epoch-90 change is -0.152%, dominated by the
large improvements on problems 126 and 142; the median is a 0.014% regression.
Runtime is effectively neutral at +0.563% on average.

The raw 20-epoch pair winner agrees with the 90-epoch winner on only 5/11
problems. A conservative post-hoc rule switches from overlap to analysis only
when analysis leads by more than 0.5% at epoch 20. It selects analysis for
problems 126 and 142, reaches 10/11 epoch-90 oracle choices, and has 0.008% mean
and 0.088% worst regret. The portfolio exposes this as
`--analysis-switch-threshold-percent`, defaulting to 0.5. This threshold must be
confirmed on future holdouts because it was chosen after inspecting this audit.

Problem 961 is excluded: its degree-839 landmark induces 87.4 million camera
pairs in the current overlap candidate construction. A scalable implementation
needs capped or sampled high-degree landmark pairs before that case is useful as
a quality experiment.

The objective is therefore a useful candidate generator, not a generally
superior replacement for overlap refinement. The reproducible experiment and
complete results are in `run_analysis_partition_experiment.py` and
`palm_runs/analysis_partition_experiment/summary.json`.

The machine-readable feature table, correlations, selector regrets, and
per-problem choices are in `partition_quality_analysis.json`. The reproducible
analysis is implemented by `analyze_partition_quality.py`.