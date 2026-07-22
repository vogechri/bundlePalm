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
| 1 | 3/10 | 0.743% | 4.791% | 0/3 | 0.202% | 0.483% |
| 3 | 5/10 | 0.198% | 0.857% | 0/3 | 0.147% | 0.361% |
| 5 | 4/10 | 0.282% | 0.893% | 1/3 | 0.026% | 0.053% |
| 10 | 4/10 | 0.140% | 0.857% | 1/3 | 0.032% | 0.071% |
| 20 | 8/10 | 0.019% | 0.174% | 2/3 | 0.018% | 0.053% |
| 30 | 10/10 | 0% | 0% | 2/3 | 0.018% | 0.053% |

Twenty epochs is the first consistently strong solver signal. Thirty epochs
does not improve the available 90-epoch audit regret because problem 135 still
changes winner after epoch 30.

## Recommended Policy

1. Enforce the existing maximum-load cap; do not continue optimizing balance
   once all candidates are feasible.
2. Generate candidates from multiple seeds and shallow refinement checkpoints.
3. Prefer a shortlist with fewer shared landmarks and a less concentrated
   Schur boundary, while retaining at least one ownership-diverse candidate.
4. Include at most one Schur-refined candidate. Total Schur cut alone is too
   weak and exact construction is expensive on high-degree graphs.
5. Run shortlisted candidates for 20 epochs and let the solver choose the
   continuation winner.

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