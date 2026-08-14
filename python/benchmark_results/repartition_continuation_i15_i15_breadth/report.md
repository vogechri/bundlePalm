# Fixed-K Repartition Continuation Gate

Date: 2026-08-14

## Mechanism

Run K24/I15 with the default deterministic `landmark_scalable` partition, save
canonical geometry, then restart K24/I15 from that exact state with the
predeclared `landmark_scalable_stable` partition. Compare against both:

1. the same canonical restart with the original partition, isolating partition
   topology from process/state reset;
2. uninterrupted default-partition K24/I30, measuring the complete staged
   workflow.

All arms use the same Schur-PCG, block-Jacobi, persistent-DABA product-space DRS
policy. Canonical handoff SSE is exact.

DABA Louvain was rejected before the gate: although materially different, its
max/mean residual load is `2.76x` on Roman and `2.97x` on BAL3068. The stable
partition retains the required ±1% residual balance while changing 5.0% of
Roman and 19.2% of BAL3068 landmark ownership after optimal cluster relabeling.

## Four-Scene Sentinel

| Scene | Stable/restart SSE | Restart/direct SSE | Stable/direct SSE | Time |
|---|---:|---:|---:|---:|
| Roman Forum | 0.946383 | 1.035240 | 0.979734 | 1.659561 |
| Trafalgar | 1.014261 | 1.019038 | 1.033571 | 1.757752 |
| Montreal Notre Dame | 0.998734 | 0.992713 | 0.991456 | 1.755796 |
| BAL3068 | 0.991955 | 0.983711 | 0.975797 | 2.774028 |
| **Geometric mean** | **0.987506** | **1.007467** | **0.994879** | **1.941485** |

The alternate partition has a genuine effect beyond restart, but the complete
workflow is expensive and Trafalgar regresses.

## Breadth

| Family | Comparison | Geometric SSE | Summed SSE | W/T/L |
|---|---|---:|---:|---:|
| 1DSfM all 15 | stable / same-partition restart | 0.968615 | 0.983186 | 10/0/5 |
| 1DSfM all 15 | restart / uninterrupted I30 | 1.036495 | 1.034361 | 2/0/13 |
| 1DSfM all 15 | stable / uninterrupted I30 | 1.003965 | 1.016969 | 5/0/10 |
| BAL all 29 | stable / same-partition restart | 0.999554 | 0.998635 | 16/0/13 |
| BAL all 29 | restart / uninterrupted I30 | 0.999903 | 0.998420 | 10/0/19 |
| BAL all 29 | stable / uninterrupted I30 | 0.999457 | 0.997057 | 11/0/18 |

Total staged/direct-I30 geometric time is `2.019627x` for 1DSfM and `1.954289x`
for BAL.

## Time-Matched Control

At the staged compute scale, direct K24/I60 is decisively better:

- 1DSfM stable-I30/direct-I60: geometric/summed SSE
  `1.117241x/1.141169x`, W/T/L `1/0/14`; staged/direct time is `0.763134x`.
- BAL stable-I30/direct-I60: geometric/summed SSE `1.007659x/1.005454x`, W/T/L
  `2/0/27`; staged/direct time is `1.135416x`.

## Decision

Reject fixed-K repartition continuation as a quality or speed mechanism. The
stable partition produces useful path diversity on 1DSfM, but restarting the
optimizer is itself harmful there, and direct continuation uses the staged work
far more effectively. BAL partition effects are negligible in aggregate.

Retain the deterministic breadth runner and assignment-overlap diagnostic only.
Do not tune the repartition time or select partitions per scene.

Artifacts were produced by `serverTest/run_repartition_continuation_breadth.sh`.
