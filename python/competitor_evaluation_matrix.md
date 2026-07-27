# Competitor Evaluation Matrix

Updated: 2026-07-26

## Purpose

The first paper question is where variable-metric DRS is competitive against
existing BA methods in quality, time, memory, and communication. Published wall
times from unlike CPU and GPU systems are context, not speedups against our
implementation. The experiments must therefore use three evidence tiers:

1. **Controlled:** methods rerun on the same machine, dataset, initialization,
   loss, precision, stopping rule, and evaluator.
2. **Same-dataset contextual:** published quality, memory, or scaling values on
   identical data, with hardware and metric differences displayed.
3. **Scale context only:** proprietary or unavailable datasets and unlike
   hardware; useful for identifying claims and experiment design, not winners.

Our current machine is one Intel i9-13900K (16 physical cores, 32 logical CPUs),
32 GiB RAM, and one RTX 4080 with 16 GiB device memory. The current DRS worker
runs all clusters on one host. It does not yet demonstrate execution on multiple
networked CPU nodes.

### Resource price reference

Azure Retail Prices API values queried on 2026-07-26 for East US, Linux,
pay-as-you-go consumption:

| Resource class | Azure SKU | Configuration | USD per VM-hour |
|---|---|---|---:|
| CPU | `Standard_F16s_v2` | 16 vCPUs, 32 GiB RAM | 0.677 |
| GPU | `Standard_NV36ads_A10_v5` | 36 vCPUs, one 24 GiB A10 GPU | 3.200 |

These SKUs are not hardware-equivalent to the local i9-13900K or RTX 4080.
They define reproducible cloud experiment classes, not conversion factors for
local timings. For a run performed on a listed SKU, compute cost is

`cost_USD = wall_seconds / 3600 * VM_price_USD_per_hour`.

Multi-node cost multiplies each node's wall time by its own VM price and sums
over nodes. Spot, low-priority, Windows, storage, data-transfer, tax, and idle
cluster provisioning costs are excluded. Dollar values remain blank for local
runs and published runs on other hardware.

## Published Evidence

### LargeBA / distributed exact LM

Source: *Distributed bundle adjustment with block-based sparse matrix
compression for super large scale datasets*, Table 4, PDF pages 8--9.

Hardware for the proposed method: ten LAN-connected CPU computers at up to
1 GB/s. Compute nodes use Xeon E5-1650 v3 3.5 GHz CPUs and 64 GB RAM; the main
node uses the same CPU and 128 GB RAM. Its reported memory is the main node's
actual usage, not aggregate cluster memory. Baseline values were copied from
the MegBA paper rather than rerun on this CPU cluster.

| Dataset | Method | Memory GB | Time s | Accuracy px |
|---|---|---:|---:|---:|
| Ladybug-1723 | Ceres-CG | 0.52 | 46.7 | 1.14 |
| Ladybug-1723 | PBA | 0.30 | 12.3 | 2.22 |
| Ladybug-1723 | DeepLM | 2.10 | 3.9 | 1.12 |
| Ladybug-1723 | MegBA | 1.60 | 0.77 | 0.56 |
| Ladybug-1723 | LargeBA | 0.06 | 15.6 | 1.12 |
| Venice-1778 | Ceres-CG | 3.68 | 1992 | 0.66 |
| Venice-1778 | DeepLM | 6.20 | 24.4 | 0.66 |
| Venice-1778 | MegBA | 13.60 | 11.9 | 0.33 |
| Venice-1778 | LargeBA | 0.27 | 69.7 | 0.67 |
| Final-13682 | Ceres-CG | 16.80 | 3897 | 1.59 |
| Final-13682 | PBA | 11.90 | 340 | 3.00 |
| Final-13682 | DeepLM | 14.89 | 149 | 1.50 |
| Final-13682 | MegBA | 89.70 | 22.6 | 0.75 |
| Final-13682 | LargeBA | 4.93 | 906 | 1.24 |

Interpretation: Table 4 supports a memory-versus-time tradeoff, not a controlled
CPU-versus-GPU speed comparison. The strongest LargeBA result is memory and
capacity. Table 5 extends this to 10 million images and 5.836 billion
observations with 67.48 GB on the main node and 7.72 hours for three LM
iterations. It also reports 32 minutes of one-time tie-point transfer, 15 seconds
for sub-RCS transfer, and 0.2 seconds for camera-pose transfer on Syn2.

Availability: the public repository contains the block-sparse compression class,
not the full distributed solver. We cannot currently produce a controlled
LargeBA row.

### MegBA / exact distributed Schur LM

Source: *MegBA: A GPU-Based Distributed Library for Large-Scale Bundle
Adjustment*, Tables 2--4 and supplementary Tables 5--9.

Hardware: one server with 80 Intel Xeon 2.5 GHz CPU cores, eight V100 GPUs,
320 GB host memory, and NVLink 2.0. Results use FP64 unless labeled FP32. CPU
baselines use 16 threads. MegBA memory in the tables is summed or allocated GPU
memory for the stated GPU configuration; it is not comparable to LargeBA's main
node memory without retaining the scope label.

| Dataset | Method | MSE px | Time s | Memory GB |
|---|---|---:|---:|---:|
| Trafalgar-257 | Ceres-16 | 0.434 | 8.160 | 1.659 |
| Trafalgar-257 | MegBA-1-m manual analytical | 0.438 | 1.148 | 1.010 |
| Ladybug-1723 | Ceres-16 | 0.562 | 34.50 | 2.093 |
| Ladybug-1723 | MegBA-1-m manual analytical | 0.560 | 0.774 | 1.660 |
| Dubrovnik-356 | Ceres-16 | 0.393 | 116.0 | 2.550 |
| Dubrovnik-356 | MegBA-1-m manual analytical | 0.411 | 3.263 | 2.480 |
| Venice-1778 | Ceres-16 | 0.334 | 319.0 | 5.983 |
| Venice-1778 | DeepLM-1 GPU | 0.333 | 24.44 | 6.256 |
| Venice-1778 | MegBA-1-m manual analytical | 0.333 | 10.92 | 7.870 |
| Venice-1778 | MegBA-8-m manual analytical | 0.333 | 3.014 | 16.79 |
| Final-13682 | Ceres-16 | 0.749 | 916.0 | 26.08 |
| Final-13682 | DeepLM-1 GPU | 0.751 | 149.6 | 14.89 |
| Final-13682 | MegBA-4-m manual analytical | 0.748 | 26.46 | 47.33 |
| Final-13682 | MegBA-8-m manual analytical | 0.748 | 20.68 | 56.06 |

Supplementary tables provide useful exact overlaps with our existing suite:
Venice-52, 245, 427, 744, 951, and 1490; Ladybug-49, 646, 931, 1064, 1266,
and 1723; Final-394, 871, 961, and 3068. These rows are appropriate for quality
sanity checks after we reproduce MegBA's MSE convention, but their V100 times
must not be compared numerically with our CPU times.

Evaluation ideas to copy:

- MSE-versus-time trajectories rather than final iteration only;
- one-to-eight-device scaling;
- manual analytical (`m`) versus autodiff (`a`) Jacobians;
- FP32 versus FP64 quality, time, and memory;
- peak memory and out-of-memory boundaries;
- GPU utilization and communication fraction.

### DABA / decentralized majorization-minimization

Source: *Decentralization and Acceleration Enables Large-Scale Bundle
Adjustment*, Tables I--IV and appendix Tables V--IX, PDF pages 6--12.

Hardware: one computer with 80 Intel Xeon 2.2 GHz CPU cores and eight V100
GPUs. DABA, DR, and ADMM use CUDA and OpenMPI; each decentralized device uses
one GPU. Ceres is C++ and DeepLM is CUDA/Python. CPU Ceres versus GPU DABA is
therefore not a controlled algorithm-only comparison. DABA versus CUDA DR and
ADMM on the same device counts is controlled.

The paper labels Table I as mean reprojection error. Reproduction from the
released source shows that the numeric value is weighted 3D-ray Ceres cost
divided by observation count: Ladybug initializes at `10.479275`, matching the
published `10.48`. It is not our E0 mean Euclidean pixel error. Centralized
methods run for 40 iterations; decentralized methods run for 1000.

| Dataset | Loss | Ceres | DeepLM | DABA 4 GPUs | DABA 8 GPUs | DABA 16 GPUs | DABA 32 GPUs |
|---|---|---:|---:|---:|---:|---:|---:|
| Ladybug-1723 | trivial | 0.707 | 0.710 | 0.690 | 0.690 | 0.690 | 0.690 |
| Venice-1778 | trivial | 0.468 | 0.466 | 0.465 | 0.465 | 0.466 | 0.473 |
| Final-13682 | trivial | 0.855 | 0.848 | 0.828 | 0.828 | 0.829 | 0.833 |
| Ladybug-1723 | Huber | 0.704 | n/a | 0.690 | 0.690 | 0.690 | 0.690 |
| Venice-1778 | Huber | 0.468 | n/a | 0.465 | 0.465 | 0.465 | 0.473 |
| Final-13682 | Huber | 0.815 | n/a | 0.796 | 0.795 | 0.796 | 0.815 |

Current local values on the first Table II row. Published and local DRS quality
columns use different metrics and are not directly rankable:

| Dataset | Method | Devices/partitions | Mean error px | Overall time s | Memory scope | Payload/iteration |
|---|---|---:|---:|---:|---|---:|
| Ladybug-1723 | DR | 4 GPUs | 0.837 | published context | per GPU, Figure 4 | published context |
| Ladybug-1723 | ADMM | 4 GPUs | 0.698 | published context | per GPU, Figure 4 | published context |
| Ladybug-1723 | DABA | 4 GPUs | 0.690 | published context | per GPU, Figure 4 | published context |
| Ladybug-1723 | Variable-metric DRS | 4 logical CPU partitions | 0.758727 | 567.552 | 1.370 GiB coordinator + 0.571 GiB worker RSS | 10.289 MiB |
| Ladybug-1723 | DR | 8 GPUs | 0.846 | published context | per GPU, Figure 4 | published context |
| Ladybug-1723 | ADMM | 8 GPUs | 0.703 | published context | per GPU, Figure 4 | published context |
| Ladybug-1723 | DABA | 8 GPUs | 0.690 | published context | per GPU, Figure 4 | published context |
| Ladybug-1723 | Variable-metric DRS | 8 logical CPU partitions | 0.826987 | 526.468 | 1.366 GiB coordinator + 0.574 GiB worker RSS | 12.760 MiB |
| Ladybug-1723 | DABA | 32 GPUs | 0.690 | published context | per GPU, Figure 4 | published context |
| Ladybug-1723 | ADMM | 32 GPUs | 0.723 | published context | per GPU, Figure 4 | published context |
| Ladybug-1723 | DR | 32 GPUs | 0.859 | published context | per GPU, Figure 4 | published context |
| Ladybug-1723 | Variable-metric DRS | 30 logical CPU partitions | 0.758406 | 66.372 | 1.357 GiB coordinator + 0.590 GiB worker RSS | 17.633 MiB instrumented |

K8 is worse than K4 in its own pixel objective and communication, and its best state occurs at
iteration 20 before the safeguard rejects later unstable trials. DRS time and
memory values are not direct comparisons to GPU rows. Serialized payload counts
both ZeroMQ directions on one host.

The historical K30 cost is reproducible (`761,438` versus `761,563` recorded).
After atomically associating each cost reply with its exact landmark snapshot,
the independent state SSE agrees to `2.53e-9` relative error. Its payload
includes diagnostic landmark snapshots and is not yet the base algorithm
communication load.

On the identical canonical Snavely objective, local 40-iteration Ceres reaches
SSE `770,093.439` and mean pixel error `0.762970`, while K30 DRS reaches SSE
`761,437.918` and mean pixel error `0.758406`. DRS is slightly better in quality;
Ceres is substantially faster (`6.790 s` solve versus `66.372 s` DRS overall).

The exact CPU-only DABA Ceres source was also reproduced: its initial/final ray
metrics are `10.479275 / 0.707147`, matching Table I's `10.48 / 0.707`.
With DABA intrinsics fitted for fixed geometry, DRS K30 scores `3.413079` and
standard Snavely Ceres scores `3.591498`. DABA Ceres jointly optimizing that ray
formulation scores `0.707147`. These ray values form a separate comparison from
the standard pixel SSE table above.

In the reverse direction, the DABA-Ceres state evaluates to `0.775256 px` mean
standard reprojection error and `1.099343 px` RMSE, versus standard Ceres
`0.762970 px` and DRS `0.758406 px`; 10 of 678,718 observations have no real
inverse-ray projection. This supports a metric-misalignment caveat, not yet a
general rejection of DABA.

The most useful evaluation design is time-to-quality. For a reference objective
`F_ref`, DABA uses

`F_delta = F_ref + delta * (F_init - F_ref)`

and reports both time to `F_delta` and time to `F_ref`. The main text uses
`delta = 2.5e-4`; the appendix text states `2e-4`, so this discrepancy must be
resolved before reproducing the table. It also reports performance profiles at
`delta = 1e-4`.

The paper additionally plots maximum memory per device and total communication
load per iteration for 1, 2, 4, 8, 16, and 32 devices on Venice, Final,
Piccadilly, and Trafalgar. DABA exchanges points and cameras and therefore sends
more than camera-only ADMM. These two metrics are mandatory for our comparison:
our complete landmark ownership should avoid point communication, which is a
specific potential advantage over DABA.

Availability: the official archived `facebookresearch/DABA` repository is
complete enough to build and run BAL with CUDA, MPI, and NCCL. It is primarily
MIT licensed with an Apache-2.0 graph component. On our single RTX 4080 we can
validate one-device behavior, but cannot reproduce the paper's 4/8-GPU scaling.

DABA's ADMM source uses nonlinear LM, Schur-PCG with a block camera
preconditioner, adaptive scalar penalties, and fixed over-relaxation `1.5`. It
does not use Nesterov/Anderson acceleration, adaptive restart, coordinate
equilibration, or a full variable consensus metric. A CPU standard-Snavely ADMM
baseline can reuse our local Ceres worker; implement the coordinator update
before adding any accelerated or variable-metric ADMM ablations.

### STBA / approximate stochastic Schur decomposition

Source: *Stochastic Bundle Adjustment for Efficient and Scalable 3D
Reconstruction*, Table 2 and Figures 5--8, PDF pages 11--16.

Hardware: each compute node is an 8-core i7-4790K with 32 GB RAM; distributed
experiments use six nodes. All compared local algorithms share one C++/Eigen
codebase. The experiments use Huber loss with scale 0.5, calibrated intrinsics,
six-parameter camera extrinsics, at most 100 iterations, and tolerances of
`1e-6`. This differs from our current nine-parameter BAL model.

| Dataset | Images | Method | RPE px | Jacobian/RCS evaluations | Mean iteration s |
|---|---:|---|---:|---:|---:|
| LS-1 | 29,975 | DBACC | 0.823 | 1011 / 1080 | 912.5 |
| LS-1 | 29,975 | STBA | 0.818 | 49 / 100 | 71.0 |
| LS-2 | 33,634 | DBACC | 0.766 | 854 / 860 | 934.8 |
| LS-2 | 33,634 | STBA | 0.783 | 48 / 100 | 79.3 |
| LS-3 | 33,809 | DBACC | 1.083 | 1025 / 1100 | 1107.0 |
| LS-3 | 33,809 | STBA | 1.056 | 49 / 100 | 89.9 |
| LS-4 | 44,276 | DBACC | 0.909 | 877 / 900 | 988.1 |
| LS-4 | 44,276 | STBA | 0.882 | 49 / 100 | 71.2 |

The private LS datasets make these scale results contextual. The public STBA
repository is MIT licensed and runnable C++ with Eigen/OpenMP on COLMAP-format
data. It supports Huber and Cauchy losses. A same-machine CPU comparison is
therefore feasible after aligning the camera model and evaluator.

Evaluation ideas to copy:

- performance profiles at normalized objective gaps `0.1`, `0.01`, and `0.001`;
- number of Jacobian and reduced-system evaluations;
- time for a 90%, 99%, and 99.9% objective-gap reduction;
- public SfM and sequential SLAM datasets;
- fixed versus changing partition ablation;
- quality degradation near stationarity.

### PenBA / distributed penalty BA

Source: *Distributed Bundle Adjustment Based on Penalty Function Method*, IEEE
RA-L 2025, DOI `10.1109/LRA.2025.3558702`.

The full text is not available through the open indexes checked here. The
verified abstract reports BAL, SfM, and SLAM experiments and claims, relative to
the most accurate conventional PCG BA solver, 3.8% lower objective, 17% lower
time per iteration, and 24% lower maximum memory consumption. Hardware,
datasets, stopping rules, and memory scope are unknown from the accessible
material, so these values are requirements to investigate, not comparison rows.

## What We Can Compare Now

| Question | Status | Reason |
|---|---|---|
| Is our final quality competitive on BAL? | Instrumented, experiment pending | E0 now saves best states and reports SSE, both MSE conventions, RMSE, mean/median/tail reprojection errors, and optional Huber cost. Published metric conventions still require paper-specific confirmation. |
| Are we faster than GPU methods? | Not meaningful | One CPU host versus V100 or multi-V100 systems. |
| Are we faster than CPU Ceres/STBA? | Testable | Rerun all methods on our i9 with fixed thread count and common evaluator. |
| Do we use less per-node memory? | Testable | Add coordinator, maximum worker, aggregate RSS, and GPU memory scopes. |
| Do we communicate less than DABA? | Testable analytically and empirically | DRS sends duplicated camera states and block metrics; DABA sends points and cameras. Count payload bytes and rounds. |
| Do we scale to larger data under a memory cap? | Testable | Sweep BAL/1DSfM size under fixed per-process and aggregate memory budgets. |
| Do fewer synchronizations help on slow networks? | Testable after byte accounting | Add latency/bandwidth emulation only after transport metrics are correct. |

## First Controlled Experiment Set

### E0: common evaluator and provenance

Status: implemented and validated on Ladybug-49. The independently evaluated
DRS SSE agrees with its native worker SSE to `8.4e-8` relative error. Exact BAL
canonicalization is exportable so Ceres and DRS can use the same focal-sign and
scene-normalized initialization. This validates the harness, not competitiveness.

#### Local harness validation, not a paper result

One diagnostic run on Ladybug-49 used the identical canonicalized initial state
and L2 objective. DRS used 20 outer iterations and 30 clusters; Ceres 2.2 used
16 threads, iterative Schur, Schur-Jacobi, and at most 40 iterations. Times are
internal setup-inclusive timestamps from one run, without repetitions or
warm-up statistics.

| Method | Initial SSE | Final SSE | Final mean error px | Time to DRS final SSE s | Configuration |
|---|---:|---:|---:|---:|---|
| Variable-metric DRS | 1,701,824.921 | 27,081.306 | 0.5866 | 2.692 | 20 outer iterations, 30 clusters |
| Ceres iterative Schur | 1,701,824.921 | 26,688.552 | 0.5796 | 0.096 | iteration 3 crossing; 16 threads |

This row confirms that the common evaluator and time-to-target machinery work.
It also rejects Ladybug-49 as evidence for a DRS speed advantage under this
configuration. It does **not** compare converged methods, repeated-run medians,
memory, larger-scale behavior, distributed execution, or cloud cost. Those
remain E1 measurements.

Implement one state evaluator that reports, for both L2 and Huber where
applicable:

- summed squared residuals;
- mean squared residual per observation and per scalar residual;
- RMSE per observation and per scalar residual;
- mean Euclidean reprojection error in pixels;
- median and 90th/95th percentile reprojection error;
- robust objective with the exact loss scale;
- finite-state and observation-count checks.

Save final and best camera/point states. Record dataset hash, initialization
hash, camera model, loss, precision, thread/device count, and stopping rule.

### E1: same-machine CPU table

Start with the exact public rows shared by MegBA and DABA:

- Trafalgar-257;
- Ladybug-1723;
- Dubrovnik-356;
- Venice-1778.

Methods: DRS, Ceres with 16 physical/logical-thread settings, and STBA where its
camera model can be aligned. Use L2 first, then Huber 0.5. Report:

- time to normalized objective gaps `1e-1`, `1e-2`, `1e-3`, and `1e-4`;
- final common quality metrics;
- setup and optimization time separately;
- peak coordinator RSS, maximum worker RSS, aggregate RSS, and out-of-memory;
- Jacobian/local-solve counts;
- synchronization rounds and payload bytes.

This is the first table from which CPU runtime claims can be made.

### E1b: SfM_Init-derived 1DSfM external-validity benchmark

Generate selected 1DSfM scenes with the pinned original SfM_Init cleanup
semantics, a documented modern rotation/translation implementation, and
deterministic triangulation. Start with Union Square as the smallest target,
then add Gendarmenmarkt and one larger unordered scene after the conversion and
independent pixel-evaluation gates pass.

Run all methods from the same generated BAL files and report this as a separate
external-validity table. Include source/retained graph counts, cleanup rejection
counts, initial pixel distributions, file hashes, and the normal runtime,
memory, rounds, and bytes fields. This benchmark can support generalization and
resource claims but is not a one-to-one DABA comparison.

Exact DABA Table II reproduction remains a separate closed gate. Reopen it only
if the authors' preprocessed files or exact generation procedure are recovered;
require matching camera/point/observation counts and initial DABA-ray metric.

### E2: one-GPU controlled context

Run official DABA and the existing exact-Schur GPU implementation on the RTX
4080 using the same BAL files and common evaluator. Report these in a separate
GPU table. Do not divide their time by the CPU DRS time. Compare quality,
memory-cap feasibility, rounds, and bytes across tables; compare runtime only
within the GPU table.

### E3: partition and resource crossover

For DRS, sweep `K = 1, 2, 4, 8, 16, 30` on the four E1 datasets. The current
single-host implementation measures algorithmic partition effects, not network
scaling. A true distributed claim requires separate worker processes or hosts.
Plot quality versus time, maximum worker memory, aggregate memory, bytes, and
rounds. This identifies whether our advantage is memory, communication, or
neither.

## Claim Gates

1. **Quality gate:** common-evaluator values agree with native values and reach
   the published quality range on at least the four shared BAL datasets.
2. **CPU competitiveness gate:** DRS is Pareto-competitive with Ceres/STBA in at
   least one quality/time/memory regime on the same CPU.
3. **Distributed-value gate:** measured state/metric bytes and synchronization
   rounds predict a crossover under realistic latency or bandwidth.
4. **Scale gate:** DRS solves a larger case or uses materially less maximum
   worker memory at comparable quality, without hiding aggregate memory.

If these gates fail, the paper should not present DRS as a competitive BA
system. It can still focus on finite-prox theory, consensus behavior, or a
different application where independent nonlinear local solves are valuable.