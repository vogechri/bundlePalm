# Manuscript and Cumulative Evaluation Spine

Last reviewed: 2026-08-23

## Proposed Paper Story

The paper studies how to solve the original nonconvex pixel bundle-adjustment
objective with independent nonlinear local solves and few global
synchronizations. The central method is landmark-owned, variable-metric,
safeguarded Douglas--Rachford splitting. Each contribution should enter where
it becomes mathematically necessary, and the evaluation should add the same
components cumulatively.

The recommended method order is:

1. landmark-owned nonlinear decomposition;
2. full block-metric camera consensus;
3. coordinate normalization and preconditioning;
4. safeguarded outer acceleration with adaptive regularization;
5. efficient finite local solves using Schur power/Nesterov iterations;
6. partition construction as the mechanism that controls local conditioning,
   load balance, duplication, and communication.

Landmark ownership is not necessarily a headline innovation, but it must appear
first because it defines which variables are local and which cameras require
consensus.

## Method Sections

### 1. Landmark-Owned Nonlinear BA

Assign every complete landmark track to exactly one worker. Worker $i$ solves

$$
\min_{u_i,\ell_i}
f_i(u_i,\ell_i)
+\frac12\|u_i-s_i\|_{D_i}^2.
$$

Landmarks never enter global consensus. Only camera copies shared by multiple
landmark owners are communicated. This preserves the original nonlinear local
BA objective and avoids collectives inside the local Schur solve.

### 2. Full Block-Metric Camera Consensus

For camera $c$ and reflected local states $r_{i,c}=2u_{i,c}-s_{i,c}$,

$$
v_c=
\left(\sum_{i:c\in i}D_{i,c}\right)^{-1}
\sum_{i:c\in i}D_{i,c}r_{i,c}.
$$

The formula is a standard metric projection. The BA-specific contribution is
using regularized local $9\times9$ camera curvature blocks to reconcile
duplicated cameras. The initial 1723/K10 evidence is strong:

| Projection | Best pixel SSE | Hard rejections | Final hard `Be` |
|---|---:|---:|---:|
| Arithmetic | 124,050,155 | 19 | 0.5 |
| Scalar | 866,306 | 14 | 0.5 |
| Diagonal | 805,805 | 8 | 0.0128 |
| Full block | **764,872** | **7** | **0.0064** |

Full blocks improve SSE by 5.35% over diagonal and 13.26% over scalar, while
arithmetic projection is numerically unstable. The five-scene breadth gate is
now complete: full blocks achieve the lowest pixel SSE on every scene, with
geometric-mean SSE `1,076,627`, 6.70% below diagonal and 12.36% below scalar.
They also require the fewest hard rejections. This supports promoting full
block consensus to the first primary method contribution.

### 3. Coordinate Normalization and Preconditioning

Explain physical scene normalization and camera coordinate equilibration as a
consistent linear coordinate change. States, centers, metrics, and consensus
must transform together. Compare:

- no scaling;
- initial Jacobi geometric-mean normalization;
- symmetric Ruiz, if retained;
- optional full block coordinate transform.

The claim is not that one named scaling always wins. The claim is that the BA
coordinates require equilibration for a meaningful and stable metric prox.

### 4. Safeguarded Outer Acceleration

Present plain variable-metric DRS first, then accelerated trial centers. The
practical method evaluates an accelerated proposal and a plain-DRS fallback.
A final trial is rejected using relative splitting merit and physical primal
quality:

$$
E_t>\alpha_k E_r
\quad\land\quad
F_t>\beta_k F_r.
$$

Recovery restores the best consistent camera/landmark state, resets momentum,
and increases the local metric regularization `Be`. Nonfinite rejection is
unconditional. The adaptive regularization schedule still needs refinement:
after stable progress, test a cautious decrease so a difficult local region
does not permanently over-damp the remainder of the path.

For exposition, acceleration, fallback, restart, state restoration, and
regularization adjustment form one stability package. In the ablation they
should be introduced separately:

1. plain DRS;
2. unsafeguarded acceleration;
3. accelerated/plain fallback;
4. fallback plus momentum restart;
5. best-state restoration;
6. restoration plus adaptive `Be`;
7. optional `Be` relaxation after stable steps.

The first broad outer-Nesterov experiment is complete: 29 BAL scenes,
K10/K20/K30, and 87 paired accelerated/control configurations. Acceleration
improves best SSE in 70 cases, ties 9, and loses 8; median improvement grows
from 6.4% at K10 to 12.0% at K30. Runtime overhead is 26.6%. Stable endpoint
counts are almost unchanged, because accelerated fallback cannot repair a plain
step that itself leaves the good basin. This motivates keeping acceleration and
accepted-state safeguarding as separate cumulative contributions.

The failure cohort for safeguard development is now broader than 1723:
646 (all steps rejected), 931 (consensus around a bad basin), 1064 and 1266
(strong partition sensitivity), and 1723 (gradual K10 basin loss).

### 5. Finite Local Schur Solves: Power To Nesterov

Once the outer method is stable, explain the local linear-solver contribution.
For the preconditioned Schur system,

$$
A=I-U^{-1}WV^{-1}W^T,
$$

BA gives a spectrum suited to fixed-point/power iterations. Interpret the
PowerBA iteration as gradient descent on the Schur quadratic and apply
Nesterov acceleration. The claim should be empirical efficiency for the
finite-work regime, not universal superiority over PCG.

Required comparisons at matched outer method and local nonlinear policy:

- zero/very few power iterations;
- 25, 50, 100 power iterations;
- Nesterov at the same matrix-vector-product budgets;
- Schur-PCG at matched residual tolerance and product count;
- Ceres iterative Schur as a robustness reference.

Report local residual reduction, accepted nonlinear decrease, wall time, and
the downstream outer objective. A faster linear residual that produces worse
outer progress is not a win.

### 6. Partitioning As A Constrained Systems Problem

Partitioning should be described by the requirements it must satisfy, not only
by the implementation heuristic.

Primary targets:

1. **Completeness:** each landmark track has exactly one owner and no
   observation is lost.
2. **Local rank support:** minimize camera-cluster copies with fewer than 5,
   10, and 20 distinct landmarks; especially avoid fewer than 5 for a
   9-parameter camera.
3. **Work balance:** balance observation factors and estimated local nonlinear
   solve cost, not merely cameras or landmarks.
4. **Memory balance:** control maximum cameras, landmarks, observations, and
   Schur workspace per worker.
5. **Communication:** minimize duplicated camera copies and full metric/state
   payloads.
6. **Graph locality:** keep strongly coupled camera-landmark communities
   together where this does not violate rank or balance constraints.
7. **Determinism and setup cost:** report partition hash, construction time,
   and amortization over solves.

Useful diagnostics:

- observation min/max/CV per worker;
- additional camera copies and maximum cameras;
- degree histogram per camera-cluster incidence;
- weak incidences below 5/10/20 landmarks;
- predicted and measured peak worker memory;
- bytes per outer trial;
- downstream best/final objective and safeguard count.

DABA's Louvain partition is a graph-locality baseline, not automatically a
better optimizer partition. A promising hybrid is Louvain initialization
followed by constrained repair for rank support, balance, and communication.
Expensive partitioning is acceptable only when its cost is amortized or it
enables a problem/hardware regime that cheaper partitions cannot solve.

Partition selection should therefore be presented as a constrained frontier:

$$
\min\;(
	ext{weak-camera risk},
	ext{max worker work},
	ext{camera copies},
	ext{payload bytes},
	ext{partition time})
$$

subject to complete landmark ownership and a prescribed observation-balance
slack. Use lexicographic priorities only where failure is categorical: missing
tracks and severe rank-deficient camera copies should dominate modest load or
communication improvements. For the remaining quantities, report a Pareto
frontier and let the target hardware/network select the operating point.

## Cumulative Evaluation Ladder

Use one plain baseline and add one component at a time where the dependency is
coherent; use matched factorials for interacting coordinate and finite-solver
choices. The conceptual table should follow this order:

| Rung | Method | Question |
|---:|---|---|
| 0 | Arithmetic/scalar consensus, no scaling, plain DRS, robust local reference | Does the decomposition work at all? |
| 1 | + full block consensus | Does local curvature improve stability and quality? |
| 2 | + coordinate normalization/preconditioning | Does equilibration improve success and time-to-quality? |
| 3 | + acceleration only | What speed is gained and what instability is introduced? |
| 4 | + plain fallback and restart | Can acceleration failures be repaired cheaply? |
| 5 | + best-state restoration and adaptive damping | Does the complete safeguard make failures recoverable? |
| 6 | + Nesterov local Schur iteration | Does finite local work become faster at matched quality? |
| 7 | + constrained partitioner | Does partition quality improve the performance/resource frontier? |

Because some components depend on earlier ones, also include targeted
factorial controls where interactions matter. In particular:

- full block metric with and without scaling;
- acceleration with and without safeguard;
- partitioner with scalar and full consensus;
- Nesterov and PCG under the same stable outer method.

## Datasets

### BAL

Use the fixed five-scene diagnostic cohort first, then expand to the existing
ten-scene set and finally the broader BAL suite. Include K10/K20/K30 because
partition count changes overlap and local conditioning.

### 1DSfM

Use a benchmark named **SfM_Init-derived 1DSfM** after the BAL protocol is
frozen. It provides larger and less uniformly structured view graphs, which are
valuable for external validity, partition, conditioning, and communication
claims. It is not a DABA Table II reproduction.

The benchmark must preserve the original SfM_Init graph construction and 1DSfM
cleanup semantics, then apply a declared deterministic triangulation and degree
cleanup. For every generated scene, publish:

- the pinned SfM_Init revision and all cleanup/triangulation thresholds;
- source and retained camera, point, edge, track, and observation counts;
- coordinate conversion, gauge choice, and camera-model convention;
- cheirality, parallax, reprojection, and degree rejection counts;
- initial pixel-error distribution and independent BAL round-trip agreement;
- a content hash of the generated BAL file.

Results may appear in the paper as a separate external-validity table once
these provenance fields pass. They may compare our methods, Ceres, DeepLM, and
other solvers on the same generated files. They must not be compared directly
to DABA Table II values unless camera/point/observation counts and DABA's initial
ray metric also match.

## Pixel Objective Versus DABA Ray Objective

DABA optimizes a regularized 3D-ray objective, while the primary method targets
standard pixel reprojection. The ray objective avoids division by camera depth
and is likely numerically more stable, but its final state can be worse under
the metric users actually want.

The objective-conditioning ablation should hold solver, initialization,
partition, consensus, local work, and safeguard fixed, changing only the
residual model. Cross-evaluate both final states in both metrics:

| Optimized objective | Final pixel SSE/mean px | Final ray cost | Stability/rejections |
|---|---:|---:|---:|
| Pixel | required | required | required |
| DABA ray | required | required | required |

Existing centralized evidence suggests a pixel-quality gap, but the expected
size should not be generalized as `2%` until this matched distributed ablation
is complete. On Ladybug-1723, the existing DABA-Ceres ray-optimized state has
mean pixel error `0.775256 px`, compared with `0.758406 px` for the prior
full-block DRS state: a 2.22% gap in mean error. This is useful motivation, but
it changes both objective and solver and comes from one scene. If the matched
ray optimization is more stable but pixel quality is worse, that is an
important result: it explains part of DABA's robustness without changing the
paper's target objective.

The matched distributed ablation is now complete. With full block consensus
and all other settings fixed, DABA-ray optimization reduces hard rejections
from 12 to 4, confirming better numerical conditioning. However, geometric-mean
mean pixel error is 2.48% worse and pixel RMSE is 35.34% worse. The loss is not
uniform: central quantiles are often comparable or better, but scene 245 has a
maximum error of `889.7 px` and 1723 has 14 observations with no real inverse
projection. This supports the paper's decision to target standard pixel error
and to report DABA ray results with pixel-tail cross-evaluation.

Combining this cohort with the four-scene failure cohort gives nine matched BAL
scenes. Ray optimization has higher pixel SSE on eight of nine, with a 42.54%
geometric-mean SSE penalty and a 1.98% geometric-mean mean-error penalty. Scene
245 dominates summed SSE; excluding it still leaves a 2.60% geometric-mean SSE
penalty. As external supporting evidence, DABA-Ceres states on all six
SfM_Init-derived 1DSfM scenes have higher pixel SSE than DRS, by 8.00x geometric
mean, and contain 514 noninvertible observations. The matched BAL ablation is
the causal objective comparison; the 1DSfM result changes solver as well and
must be labeled supporting evidence.

## Main Claims If Evidence Holds

1. Full camera-block consensus is necessary for stable, high-quality nonlinear
   landmark-owned BA splitting; scalar and diagonal approximations lose useful
   local observability information.
2. Coordinate equilibration and relative DRE/primal safeguarding convert the
   variable-metric decomposition into a robust practical method.
3. Nesterov-accelerated finite Schur iterations reduce local work in the regime
   needed by outer splitting, without claiming universal dominance over PCG.
4. A constrained partitioner exposes a tunable computation/communication/
   conditioning tradeoff and can make more workers useful without uncontrolled
   weak camera copies.
5. Optimizing the original pixel objective yields the metric required by SfM
   users; DABA's ray objective is better conditioned but must be cross-evaluated
   in pixels.

## Claim Discipline

- Do not claim the metric projection formula itself is new.
- Do not claim convergence for refreshed metrics and finite GN solely from the
  fixed-metric exact theorem.
- Do not call a saved best state a stable endpoint.
- Do not compare DABA ray cost directly with pixel SSE.
- Do not claim the partitioner is better without downstream solver and resource
  evidence.
- Do not claim Nesterov universally beats PCG.