# Paper Plan: Variable-Metric Consensus Splitting for Large-Scale Bundle Adjustment

## 1. Decision in one paragraph

The paper should not claim the first decentralized, accelerated, asynchronous, or communication-efficient bundle-adjustment method. DABA already combines decentralized majorization-minimization, Nesterov acceleration, adaptive restart, and a first-order convergence result; PenBA supplies a recent distributed penalty alternative; earlier consensus and lazy-communication methods cover asynchronous execution and selective communication. The defensible paper is instead an evidence-driven study of a different operating point: solve the original nonlinear local BA objectives with complete landmark ownership, reconcile only duplicated cameras through full block-metric Douglas--Rachford consensus, stabilize the resulting method with coordinate equilibration and safeguarded outer acceleration, and quantify when this design beats synchronized exact Schur methods and competing decentralized methods in time, communication, memory, and solution quality.

## 2. Central research question and thesis

**Research question.** When is it preferable to spend more computation in independent nonlinear local BA solves in exchange for fewer global synchronization points?

**Proposed thesis.** A BA-specific variable-metric DRS decomposition is competitive in communication- and memory-constrained regimes because it:

1. assigns each complete landmark track to one worker, eliminating landmark consensus;
2. communicates only duplicated camera states and compact camera metrics once per outer trial;
3. uses full $9\times9$ camera blocks for geometrically meaningful consensus;
4. makes finite local solves numerically useful through scene normalization and coordinate equilibration; and
5. recovers much of the iteration cost through safeguarded outer acceleration without relying on acceleration for correctness.

This is an empirical systems-and-optimization thesis. It becomes a stronger algorithmic paper only if the finite-local-solve theory in Section 8 is completed.

The method/evaluation order and cumulative ablation ladder are specified in
`paper_method_evaluation_spine.md`. In brief: landmark ownership, full block
consensus, coordinate equilibration, safeguarded acceleration with adaptive
regularization, finite Schur power/Nesterov solves, and constrained partition
construction. The manuscript should introduce components in this dependency
order. Experiments should add components cumulatively where the dependency is
coherent and use matched factorials where coordinate and finite-solver effects
interact; C2 and C4 are not claimed as automatic cumulative rungs.

## 3. Contribution hierarchy

### Terminology

Use the Stage-C labels consistently in the manuscript and artifact names:

- **C1:** safeguarded fast DRS;
- **C2:** coordinate equilibration;
- **C3:** variable-metric consensus;
- **C4:** finite local Schur solver;
- **C5:** adaptive local nonlinear work;
- **C6:** optional distributed globalization.

Landmark ownership and full block-metric camera consensus define the base
method and precede this labeled ablation stack. Earlier drafts used C1--C5 for
the prose contribution hierarchy below; those labels are superseded by the
Stage-C convention to avoid assigning the same symbol to two mechanisms.

### Base method: variable-metric nonlinear consensus BA

Formulate BA as landmark-owned nonlinear subproblems with duplicated cameras. Each worker approximately solves

$$
\min_{u_i,\ell_i}\; f_i(u_i,\ell_i)
+\frac{1}{2}\lVert u_i-s_i\rVert_{U_i}^2,
$$

and the coordinator computes the block-metric camera projection

$$
v_c=\left(\sum_{i:c\in i}U_{i,c}\right)^{-1}
\sum_{i:c\in i}U_{i,c}(2u_{i,c}-s_{i,c}).
$$

The claimed distinction must be demonstrated, not asserted: original nonlinear local objectives, camera-only consensus, full block metrics, and no collective inside an inner Schur iteration.

The projection formula is standard variable-metric proximal mathematics and is
not itself a novelty claim. The candidate contribution is its BA-specific use:
full regularized $9\times9$ local camera curvature blocks reconcile duplicated
cameras in a complete-landmark-owned nonlinear decomposition. Until a dedicated
prior-art search is complete, describe this as a "BA-specific full block-metric
consensus construction," not as the first weighted or variable-metric
consensus method. The clean derivation and sign conventions are recorded in
`block_metric_consensus_derivation.md`.

Initial evidence is no longer single-scene only. On the fixed five-scene K10,
30-iteration cohort, full blocks have the lowest independently evaluated pixel
SSE on all five scenes. Geometric-mean SSE is `1,076,627`, versus `1,148,814`
for diagonal and `1,209,725` for scalar projection; arithmetic is unstable on
1723. Treat this as a passed breadth gate, while retaining a larger BAL/1DSfM
confirmation requirement for the final paper claim.

### C1. Safeguarded fast DRS

Show that coordinate equilibration, metric regularization, DRE-guided fallback, restart, and best-primal restoration transform a brittle decomposition into a robust solver. The package matters only if ablations show materially better success rate, objective, or time-to-quality across the full suite.

The safeguarded outer acceleration is a distinct algorithmic contribution, not
optional debugging machinery. Its clean form is a Themelis-style safeguarded
direction method: form an accelerated center proposal, evaluate it with the DRE
merit, backtrack or fall back to the nominal plain-DRS center, and restart the
acceleration history after failed proposals. The readable coordinator must
expose this through an opt-in outer-method flag while retaining plain DRS as the
control. Do not carry over the legacy exceptions that disable rejection at the
block-regularization ceiling or replace the accepted DRE reference with a
rejected trial; those are separate heuristics and are not part of the claimed
line-search contribution.

### Theory target: finite-local-solve convergence

Prove convergence under a checkable inexact-prox condition compatible with a finite number of damped GN/LM steps. DABA is especially relevant here: its theorem assumes local surrogate minimizers, while its implementation uses one successful LM step. A rigorous bridge from finite local solves to convergence would be valuable beyond this implementation.

Acceptable forms include:

- summable proximal optimality errors;
- a relative error bound tied to the DRS fixed-point residual;
- sufficient envelope decrease with a summable perturbation;
- a forcing sequence for local stationarity residuals; or
- eventual exactness under adaptive inner tolerances.

Do not claim this contribution unless every assumption is both proved and connected to quantities the implementation computes.

### Supporting systems contribution: partitioning and resource-aware execution

The exact complete-track partitioner and optimized worker implementation support scale and reproducibility. They are engineering contributions, not the headline, unless experiments establish a new partition-quality/runtime frontier with downstream solver consequences.

### Optional companion: PALM for evolving or bounded-memory problems

PALM is a different operating regime: sequential dirty-region optimization with ephemeral derivative workspaces. Include it only if changing-graph experiments demonstrate work proportional to affected edges and a meaningful memory advantage. Do not force PALM into the DRS convergence story.

## 4. Explicit non-claims

- Not the first decentralized or distributed BA method.
- Not the first accelerated decentralized BA method.
- Not the first asynchronous or low-communication consensus BA method.
- Not a claim that Nesterov universally dominates PCG.
- Not an accelerated $O(1/\epsilon)$ complexity result without a proof.
- Not a proof for changing metrics, finite GN, and heuristic acceleration merely by citing exact fixed-metric DRS.
- Not a claim of exact LM equivalence for the decomposed method.
- Not a claim that fewer communicated variables automatically means fewer bytes; full metric blocks and repeated outer trials must be counted.

## 5. Related-work map and required comparisons

| Family | Representative work | What it already establishes | Required positioning |
|---|---|---|---|
| Decentralized MM | DABA, RSS 2023 / IJRR 2025 | Decoupled BA surrogates, peer-to-peer communication, Nesterov acceleration, adaptive restart, first-order convergence under local-minimizer assumptions | Mandatory primary baseline; compare objective versus time, rounds, bytes, memory, and local-solve effort |
| Distributed penalty | PenBA, RA-L 2025 | Unconstrained penalty formulation and distributed PCG with bounded-complexity hyperparameters | Mandatory recent baseline or, if code is unavailable, reproduce published regime and state limitation |
| Global camera consensus | Zhang et al., TPAMI 2020 | Distributed BA through camera consensus | Explain camera-only overlap and compare convergence/communication where implementation is available |
| Asynchronous consensus | Liu et al., FITEE 2020 | Partial-barrier execution to avoid waiting for slow workers | Baseline for straggler experiments; no novelty claim for asynchrony |
| Lazy communication | Tian et al., IROS 2022 | Triggered communication with first-order convergence and up to 78% reported reduction | Compare rounds and bytes or discuss why the model differs |
| Approximate distributed Schur | STBA, ECCV 2020 | Stochastic decomposition of the reduced camera system | Compare time-to-quality and scale |
| Exact distributed Schur | MegBA, ECCV 2022 | GPU distributed Schur/PCG with exact global LM behavior | Mandatory synchronized baseline at matched nonlinear policy and target objective |
| Exact extreme-scale LM | Zheng et al., ICCV 2023 | Block sparse compression and million-image distributed LM | Scale/context baseline; distinguish data regime and hardware |
| Alternative Schur solvers | PowerBA, multidirectional CG, RootBA | Faster centralized or distributed linear solves | Use to contextualize inner-solver comparisons, not as DRS analogues |
| Generic nonlinear DRS | Themelis, Stella, Patrinos | Fixed-metric nonconvex DRS/DRE descent and safeguarded directions | Foundation for the exact idealization only |
| Point-cloud BA | BALM 3.0 and related LiDAR BA | Distributed MM on 70 GB point clouds and consumer laptops | Strong candidate second application and practical resource comparison |

Before submission, repeat the literature search by title/abstract and forward citations for DABA, PenBA, MegBA, global camera consensus, and nonconvex inexact DRS/ADMM.

## 6. Experimental claims and measurements

Every solver run should record:

- original reprojection objective versus wall time and outer iteration;
- best objective reached, final objective, and failure/divergence status;
- number of nonlinear local solves and rejected accelerated trials;
- synchronization rounds and collective operations;
- payload bytes sent and received, separated into states, metrics, and control traffic;
- local compute, coordinator/consensus, serialization, transfer, and barrier wait time;
- peak host and device memory per worker and total aggregate memory;
- worker utilization and straggler idle fraction;
- hardware, process placement, precision, stopping rule, and initialization;
- final gradient/stationarity proxy and camera-consensus residual;
- cloud cost or normalized device-seconds when comparing unlike hardware.

Report distributions over repeated runs for wall time. A single aggregate runtime is insufficient because synchronized methods amplify host variance.

## 7. Experiment matrix

### E0. Correctness and reproducibility gate

**Purpose:** establish that all methods optimize the same robustified BA objective from the same initialization.

- Use the complete 29-problem BAL suite already exercised by the implementation.
- Verify observation, camera, and point conventions after format conversion.
- Evaluate all final states with one independent objective evaluator.
- Report deterministic partition checksums and objective agreement for repeated runs.
- Require finite states and no missing observations on every problem.

**Gate:** no headline timing result until this passes for every included baseline.

### E1. Main solver comparison

**Methods:** proposed DRS, DABA, PenBA, global camera consensus, STBA, MegBA-PCG, MegBA-Nesterov, and a strong single-node Ceres/RootBA reference where feasible.

**Plots:** objective gap versus wall time, rounds, bytes, and peak memory. Use both absolute objective and gap to the best objective found by any trusted solver.

The authoritative complete-cohort objective/time figure is generated by
`serverTest/build_stage_c_publication_plots.py` from
`benchmark_results/stage_c_publication_comparison/summary.json` and written to
`benchmark_results/stage_c_publication_comparison/objective_vs_optimization_time.pdf`.
It separates DRS-only, terminal-correction, Schur-polishing, K1-diagnostic, and
Ceres timing roles rather than treating unlike execution boundaries as one
speedup claim. The same builder combines the frozen repeat summary and complete
terminal-correction summary in `k16_over_k4_resources.pdf`, separating sentinel
DRS optimization/CPU/transport/worker-RSS ratios from complete-cohort corrected
quality/time/process-RSS ratios. Accepted and rejected displayed outer
iterations for the matched plain/C1/C5/C1+C5 breadth are generated from the
reproducibility manifest in `outer_iteration_outcomes.pdf`.

**Targets:** small, medium, and largest feasible BAL/1DSfM problems; include sequential and unordered graphs. Report results by graph density and camera-overlap ratio, not only by dataset name.

**Decision rule:** the method needs a visible regime where it is Pareto-competitive. Winning only at a fixed iteration count is not enough.

### E2. Component ablations

Run matched cohorts for:

1. arithmetic versus diagonal-metric versus full $9\times9$ block-metric consensus;
2. no coordinate scaling, Jacobi, and symmetric Ruiz;
3. scene normalization off/on;
4. plain DRS, unsafeguarded momentum, fallback only, fallback plus restart, and full restoration;
5. fixed versus adaptive proximal regularization;
6. one, two, five, and converged local GN/LM steps;
7. fixed metric versus refreshed metric, if both are supported;
8. random/view-graph partitioning versus complete landmark-track ownership.

Report time-to-target as well as final objective. The existing near-tie between Jacobi and Ruiz should be presented honestly; the broader claim is that equilibration matters relative to no scaling, not that Ruiz is universally superior.

### E3. Scale and partition count

Use $K\in\{1,2,4,8,16,30\}$ where problem size permits. Measure:

- local compute balance;
- duplicated-camera fraction;
- metric and state bytes;
- outer iterations to target;
- total wall time and parallel efficiency;
- partition construction time and quality.

The expected curve is non-monotone: more workers reduce local work but increase overlap and consensus burden. The paper should identify the crossover, not hide it.

### E4. Network operating regimes

Use traffic control or a reproducible proxy to sweep:

- round-trip latency: local, 1, 5, 20, 50, and 100 ms;
- bandwidth caps: 10 Gbit/s, 1 Gbit/s, 100 Mbit/s, and 20 Mbit/s;
- optional packet jitter.

Compare DRS, DABA, PenBA, lazy communication if available, and MegBA. Plot time-to-objective against latency and bandwidth. This is the most direct test of the central thesis.

### E5. Heterogeneous workers and stragglers

Inject controlled delays into 0%, 10%, and 25% of workers at 1.5x, 2x, and 4x slowdown. Compare full barrier, partial barrier/asynchronous policy if implemented, and stale-result rejection behavior.

Measure useful work, wait time, objective trajectory, and reproducibility. If no asynchronous update is implemented, present this as a limitation rather than claiming straggler tolerance from out-of-order reply handling.

### E6. Resource and deployment value

Run at least one matched CPU/commodity-node configuration and one GPU configuration. Report:

- maximum solvable problem under fixed per-node memory;
- total and per-node peak memory;
- energy if reliable counters exist, otherwise device-seconds and cloud price;
- setup/partition amortization;
- failure recovery cost.

The practical question is whether several modest nodes solve a problem that does not fit one accelerator, or achieve a lower cost at a target accuracy.

### E7. Finite-local-solve diagnostics

For each local solve record:

$$
e_{i,k}=\nabla f_i(u_{i,k},\ell_{i,k})
+U_i(u_{i,k}-s_{i,k}),
$$

plus local model decrease, accepted LM ratio, step norm, and local iterations. Test whether observed errors are:

- summable;
- decreasing to zero;
- bounded by $\eta_k\lVert u_k-v_k\rVert$ with $\eta_k<\bar\eta$;
- or compatible with a perturbed DRE decrease inequality.

This experiment decides which inexact theorem is realistic. It must precede choosing the proof assumptions.

### E8. PALM changing-graph experiment

Construct chronological or synthetic update streams that add observations, landmarks, and cameras. Compare:

- full global re-solve;
- full PALM sweep;
- dirty-region PALM with periodic consistency sweeps.

Measure update latency, work versus $|E_{\mathrm{dirty}}|$, peak memory, objective drift, and recovery after a large loop closure. Include PALM only if it yields a clear bounded-memory or incremental advantage.

### E9. Global SfM application

**Immediate experiment:** replace the final BA stage in GLOMAP with the proposed solver while keeping matching, view graph, rotation averaging, global positioning, triangulation, robust loss, and initialization fixed.

Use standard Global SfM datasets already supported by GLOMAP, plus at least one large collection. Report registered images, pose accuracy where ground truth exists, reprojection objective, final BA time, total pipeline time, memory, and communication.

**Exploratory experiment:** apply consensus decomposition to GLOMAP's global positioning objective. Keep this out of the main claim unless it works robustly across datasets; it changes the optimization model and would dilute a BA paper.

**Interpretation:** final-BA replacement demonstrates integration value but is not a second algorithmic application. A stronger generality claim requires another nonlinear least-squares family with different residuals or state blocks.

### E10. Stronger second application candidates

Ranked options:

1. **LiDAR/point-cloud BA:** strongest demonstration of generic nonlinear local solvers and large local data; BALM 3.0 is a mandatory baseline.
2. **Point-and-line BA:** exposes heterogeneous residuals and nonstandard landmark blocks that a BA-specific surrogate may not handle directly.
3. **Photometric BA:** stresses expensive residuals, making fewer synchronization points potentially more valuable.
4. **GLOMAP global positioning:** closest integration, but less independent from the main SfM story.

Choose one only after a small feasibility prototype. The best candidate is the one that reuses the consensus machinery while requiring genuinely different local residuals.

## 8. Theory work plan

### T1. Exact fixed-metric theorem

Define the reduced local camera objective

$$
\bar f_i(u_i)=\inf_{\ell_i} f_i(u_i,\ell_i),
$$

stack camera copies, and write $F(u)=\sum_i\bar f_i(u_i)$ and $G=\iota_{\mathcal C}$ for the consensus subspace. For a constant positive-definite block metric $M$, transform $z=M^{1/2}u$ and apply Euclidean nonconvex DRS to $F(M^{-1/2}z)+G(M^{-1/2}z)$.

State all assumptions explicitly: differentiability and Lipschitz gradient of the selected first term on the relevant level set, proper lsc second term, existence of a solution, exact proximal selections, fixed metric, admissible step size, and boundedness/level-boundedness where needed.

### T2. Finite-GN bridge

Derive the local proximal optimality defect after a finite LM solve. Seek a bound of the form

$$
\lVert e_k\rVert_{M^{-1}}
\leq \eta_k\lVert r_k\rVert_M+\delta_k,
\qquad
\sum_k\delta_k<\infty,
$$

or prove directly

$$
\mathcal E_{k+1}\leq \mathcal E_k-c\lVert r_k\rVert_M^2+\epsilon_k,
\qquad
\sum_k\epsilon_k<\infty.
$$

Connect LM acceptance and adaptive inner stopping to this bound. A fixed number of GN steps with fixed tolerance will not generally imply summable error; do not assume it does.

### T3. Metric updates

Either freeze the metric for the theorem, update it only finitely many times, or prove uniform spectral bounds plus controlled metric variation such as

$$
mI\preceq M_k\preceq MI,
\qquad
\sum_k\lVert M_{k+1}-M_k\rVert<\infty.
$$

Without a common Lyapunov argument, refreshed curvature metrics remain heuristic.

### T4. Acceleration safeguards

Prove convergence for accelerated proposals only through an accepted sufficient-decrease condition and plain DRS fallback. The proposal generator itself may be Nesterov-like, Anderson, or quasi-Newton; correctness should depend on acceptance, not on naming the momentum scheme.

The current implementation's maximum of a DRE-like merit and primal objective is not automatically the exact DRE. Either align the check with the theorem or label the safeguard empirical.

### T5. Rates

From summability of squared residuals, claim only the standard best-iterate residual bound $O(k^{-1/2})$ unless stronger structure is proved. Do not claim an accelerated rate from empirical momentum.

## 9. Manuscript story and figure plan

### Suggested title

**Trading Synchronization for Nonlinear Local Solves in Distributed Bundle Adjustment**

Alternative if the inexact theorem succeeds:

**Inexact Variable-Metric Douglas--Rachford Splitting for Distributed Bundle Adjustment**

### Section outline

1. **Introduction:** synchronization versus redundant nonlinear work; precise contributions and non-claims.
2. **Related work:** exact distributed Schur, decentralized/MM/penalty BA, consensus/asynchronous methods, and nonconvex splitting.
3. **Problem and decomposition:** complete landmark ownership, duplicated cameras, communication model.
4. **Variable-metric DRS:** local prox, block-metric consensus, coordinate transport, safeguards.
5. **Convergence:** exact theorem; inexact theorem only if completed; status of changing metrics and acceleration.
6. **Implementation:** partitioning, workers, message contents, failure recovery, instrumentation.
7. **Experiments:** correctness, main comparison, ablations, network/straggler/resource regimes, Global SfM.
8. **Limitations:** finite solves, metric variation, coordinator design, partition dependence, hardware fairness.

### Essential figures

- Decomposition diagram showing owned landmarks and duplicated cameras.
- Objective gap versus wall time for all primary baselines.
- Time-to-target heatmap over latency and bandwidth.
- Pareto plot of time, communicated bytes, and peak memory.
- Ablation plot for metric/scaling/safeguards.
- Scaling plot with compute, communication, and barrier-wait breakdown.
- Finite-solve residual diagnostic supporting the chosen theory assumption.
- GLOMAP final-BA integration result.

## 10. Execution order and go/no-go gates

### Phase A: instrumentation and fair baselines

1. Add byte, round, phase-time, memory, and local-optimality instrumentation.
2. Reproduce DABA on overlapping datasets and hardware where possible.
3. Establish MegBA-PCG and MegBA-Nesterov target-objective curves.
4. Determine PenBA code availability or implement a narrowly faithful baseline.

**Gate A:** if DRS has no Pareto-competitive regime against DABA and MegBA after fair tuning, stop framing it as a solver paper and pivot to an empirical analysis or the finite-GN theory result.

### Phase B: central evidence

1. Run E0--E4.
2. Run component ablations E2 before large deployment experiments.
3. Identify the latency, bandwidth, overlap, and memory crossover regimes.

**Gate B:** proceed only if gains survive repeated runs and are not explained by unequal stopping rules, hardware, or objective evaluation.

### Phase C: theory

1. Instrument E7.
2. Select the weakest realistic inexactness assumption.
3. Complete and independently audit the proof.
4. Test the assumption on all main experiments.

**Gate C:** if the proof or empirical condition fails, retain only the exact idealized theorem and state finite GN as a limitation.

### Phase D: application and optional PALM

1. Integrate final BA into GLOMAP.
2. Prototype one stronger second application, preferably point-cloud or point-and-line BA.
3. Run changing-graph PALM only if it remains a manuscript contribution after the main DRS results are known.

### Minimum publishable package

- DABA, MegBA, and at least one other distributed baseline.
- Full objective/time/round/byte/memory accounting.
- Network-regime and component-ablation results.
- Exact fixed-metric convergence statement with honest implementation boundary.
- One real pipeline integration, preferably GLOMAP final BA.

### Strong package

All of the above, plus a proved and empirically verified finite-local-solve theorem and a second nonlinear least-squares application.
