# Paper Experiment Status

Updated: 2026-07-26

## Current Evidence

The first diagnostic cohort compares 20 DRS outer iterations against recovered
20-iteration edge-Schur/Nesterov summaries. The DRS runs use 30 landmark-owned
clusters. These results locate useful questions, but they are not publication
comparisons because the recovered Schur implementation is missing, its records
mark cross terms as unverified, final objectives have not been independently
evaluated, and partition counts are unmatched on two of the three problems.

| Problem | DRS cost gap | DRS total/Schur time | DRS solve/Schur time | Interpretation |
|---|---:|---:|---:|---|
| 52 | +4.81% | 0.58x | 0.51x | Faster at equal iterations, worse objective |
| 871 | +1.09% | 2.20x | 1.60x | Schur better; partition counts differ 30 vs 60 |
| 3068 | +14.78% | 7.00x | 3.84x | Schur clearly better; partition counts differ 30 vs 20 |

The problem-52 row uses the recovered 30-edge-partition Schur record. The 871
and 3068 rows use the only available Schur partition counts. DRS solve time is
`overallSeconds - partitionSeconds`; Schur timing appears to exclude dataset
loading and partition setup.

Artifacts:

- `benchmark_results/drs_matched20.jsonl`
- `benchmark_results/ba_competitiveness_matched20.md`
- `analyze_ba_competitiveness.py`
- `normalize_ba_benchmark.py`

## Claim Status

We cannot currently claim that DRS is faster or more accurate than an exact
distributed Schur method. The evidence supports a narrower working hypothesis:

> Spending more local nonlinear work can be attractive when synchronization or
> high-partition overhead dominates, but the crossover depends strongly on graph
> structure, partition setup, and the target objective accuracy.

Problem 52 is the first candidate crossover regime. Problems 871 and 3068 are
important negative cases that the paper must explain rather than average away.
Likely explanatory variables are camera overlap, local problem size, partition
construction cost, convergence per outer round, and the amount of exploitable
parallelism in the exact Schur path.

The strongest defensible story remains a Pareto/regime result, not universal
solver dominance:

1. complete landmark ownership removes landmark consensus;
2. nonlinear local solves avoid collectives inside a global linear solve;
3. block-metric camera consensus trades redundant local work for fewer global
   synchronization points; and
4. the useful regime should emerge under latency, bandwidth, memory, or
   heterogeneous-worker constraints.

## Immediate Gates

### Gate 0: reproducibility and objective equality

1. Restore or replace `bae_overlay/examples/schur_megba_edge_nesterov.py`, which
   is currently empty.
2. Save final camera and landmark states from both methods.
3. Evaluate every state with one independent BAL objective implementation.
4. Verify Schur cross terms and identical initialization, normalization,
   precision, damping policy, and loss.
5. Repeat matched 20-iteration and time-to-target runs at equal partition counts.

No comparative timing claim should pass this gate without independently
verified objectives.

### Gate 1: resource accounting

Record per run:

- setup, partition, local solve, consensus, serialization, transfer, and wait
  times;
- synchronization and collective counts;
- state, metric, control, and total bytes;
- peak resident and device memory per worker and in aggregate; and
- local nonlinear iterations and rejected accelerated trials.

Run at least five timed repeats after one warm-up. Report medians and dispersion.

### Gate 2: regime experiment

Use problems 52, 871, and 3068 first, then expand only after the harness is
trusted. Sweep matched partition counts and target objective gaps. Add simulated
latency and bandwidth only after local byte accounting is correct. The main
figure should be a time/bytes/memory Pareto plot plus a latency-bandwidth
crossover heatmap.

## Generality Program

Generality is potentially valuable, but it needs two levels of evidence.

### Robust objectives

Supporting Huber, soft-L1, and Cauchy losses is a useful first demonstration
because the consensus mechanism should not change: only the local nonlinear
objective changes. Direct L1 is nonsmooth and is not the right first target for
the current GN/LM implementation. Huber or soft-L1 should come first, followed
by Cauchy for stronger outlier suppression.

Required experiment:

1. inject controlled observation outliers and use a real outlier-contaminated
   SfM dataset;
2. run L2, Huber/soft-L1, and Cauchy under identical initialization;
3. compare robust objective, inlier reprojection error, pose error where ground
   truth exists, convergence, and communication; and
4. apply the same robust weighting in every baseline.

This demonstrates residual/loss flexibility, but by itself does not establish
generality beyond bundle adjustment.

### Second application

The candidates should be prioritized by how directly they test the paper thesis:

1. **Point-and-line BA:** relatively close to the current implementation while
   adding heterogeneous residuals and landmark block types. This is the most
   practical first generality prototype.
2. **Photometric BA:** expensive local residual evaluation makes the trade of
   local computation for fewer synchronization points especially relevant.
3. **LiDAR/point-cloud BA:** strongest large-data and memory story, with BALM 3.0
   as a mandatory baseline, but substantially more implementation work.
4. **GLOMAP final BA:** best near-term pipeline integration and useful real-world
   evidence, but still BA and therefore not a strong generality claim.
5. **GLOMAP global positioning:** a genuinely different objective and a stronger
   generic-consensus test, but riskier and less aligned with the current local
   solver structure.

The recommended order is robust BA, GLOMAP final-BA integration, then a small
point-and-line BA prototype. Escalate to photometric or LiDAR BA only if the
resource-regime experiments support the synchronization thesis.

## Decision Rule

- If matched, verified experiments reveal a latency, bandwidth, or memory regime
  where DRS is Pareto-competitive, keep the solver/regime paper story.
- If DRS remains dominated after fair tuning, pivot the main contribution toward
  finite-local-solve theory, empirical analysis of nonlinear consensus, or the
  generic local-solver application result.
- If robust and second-application experiments work with minimal consensus
  changes, claim modular applicability. Do not claim genericity merely because
  the equations permit another local objective.