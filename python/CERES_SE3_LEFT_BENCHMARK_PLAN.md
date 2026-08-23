# 1DSfM/BAL Versus Ceres Left-SE3 Benchmark Plan

Phase A status: **complete**. The artifact-only report, JSON summary, per-scene
table, and two deterministic PDFs are under
`benchmark_results/ceres_se3_left_unified_benchmark/`. Phase B remains optional
and requires explicit authorization.

## Purpose

Produce one cohort-explicit benchmark of the retained distributed method against
Ceres left-SE3 on the standard Snavely pixel objective. Ceres is a centralized
reference, not a replacement for the distributed method.

This benchmark must not reopen the failed K4/K16 transfer or tune by scene. The
primary new-method row is the accepted copied-baseline K24/I120 combined stack:
direct left-SE3 tangent equations, exact tangent metric, shared-only
product-space semantics, and guarded I60+I90 Krylov camera/landmark proposals.
The established Stage-C K4/K16 rows may appear as separately labeled
resource/latency context; they are a different solver lineage.

## Cohorts And References

- SfM_Init-derived 1DSfM: all 15 scenes.
- BAL: all 29 top-level problems.
- Ceres reference: left-SE3, I90, T16, independently evaluated pixel SSE.
- Ceres artifacts:
  - `benchmark_results/1dsfm_drs_ceres_se3_all15/ceres/results.jsonl`
  - `benchmark_results/bal_ceres_se3_all29/results.jsonl`
- K24 combined artifact:
  - `benchmark_results/k1_carryover_joint_factorial/direct_shared_proposal/`
- K24 matched direct control:
  - `benchmark_results/k1_carryover_joint_factorial/direct/`

## Phase A: Artifact-Only Unified Report

Complete. It required no solver rerun.

Generate one Markdown/JSON report with, for each family:

- scene count and completion count;
- geometric and summed SSE versus Ceres;
- W/T/L versus Ceres and versus the matched direct control;
- worst per-scene ratio and complete per-scene ratio table;
- aggregate optimization time and timing class;
- coordinator/worker peak RSS where available;
- transport bytes where available;
- proposal attempted/selected counts at I60 and I90;
- recovery exhaustion and no-op counts;
- exact source artifact paths and repository commit.

Initial verified snapshot:

| Family | Combined/Ceres geometric | Summed | W/T/L | Worst | Combined/direct geometric | Complete |
|---|---:|---:|---:|---:|---:|---:|
| 1DSfM-15 | 1.169700 | 1.046518 | 3/0/12 | 2.120880 | 0.762902 | 15/15 |
| BAL-29 | 1.001181 | 0.997696 | 11/0/18 | 1.035083 | 1.000131 | 29/29 |

The BAL result must report both geometric and summed ratios: the geometric mean
is slightly above Ceres while the summed SSE is slightly below it.

## Context Rows

The unified report may include these already-authoritative rows, clearly
separated from the primary K24 combined stack:

- preserved base DRS K24/I200 for 1DSfM and K24/I90 for BAL;
- Stage-C K4 resource and K16 latency endpoints;
- K4/K16 plus one terminal correction;
- K1 BAE-style local diagnostic;
- DRS+Schur polishing rows.

Do not compare unlike CPU/GPU timing classes as speedups. Do not imply that the
failed frozen combined-stack K4/K16 development transfer invalidates the
established Stage-C K4/K16 endpoints.

## Phase B: Optional Fresh Confirmation

Run only if a fresh same-source benchmark is explicitly requested after Phase A.
Freeze before launch:

- repository commit and dirty state;
- detached worker/client revision and worker binary hash;
- exact 15-scene and 29-problem manifests;
- partition-cache policy and hashes;
- K=24, I=120, T=1/cluster configuration;
- direct and direct+shared+guarded-Krylov arms;
- Ceres left-SE3 I90/T16 reference artifacts or exact rerun command;
- independent evaluator version;
- 14 GiB per-process virtual-memory cap, `MALLOC_ARENA_MAX=2`, serialized large
  BAL cases, and `OVERWRITE=0` resume behavior.

A fresh rerun must reproduce the first 59 trajectory rows for proposal-only
comparisons where the structural arm is matched. Shared-only and direct arms are
not expected to share an identical prefix.

## Outputs

Place compact outputs under:

```text
benchmark_results/ceres_se3_left_unified_benchmark/
  report.md
  summary.json
  per_scene.md
  objective_vs_time.pdf
  sse_ratio_by_scene.pdf
```

Keep raw JSONL, logs, memory traces, and states outside ordinary Git. Commit the
analyzer, compact summaries/tables/figures, and documentation.

## Decision Language

Report this as a quality/resource benchmark, not a winner-take-all gate. The
primary questions are:

1. how close the retained distributed K24 stack is to Ceres endpoint quality;
2. where it wins or loses by scene and family;
3. what quality gain guarded Krylov adds over the matched direct control;
4. how distributed resource, communication, and timing costs differ from the
   centralized reference.
