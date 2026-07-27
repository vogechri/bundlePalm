# Global Safeguard Interpretation and Recovery Design

Date: 2026-07-26

## Purpose

The global safeguard is numerical-failure handling around the distributed
outer map. It is not a monotone line search for ADMM. Normal ADMM transients
must remain possible: on Ladybug-1723, the first global pixel SSE rises from
`1.24e8` to `2.71e11`, then falls to `1.67e7` on the next iteration. Guards at
`1.1x` and `2x` rejected this useful transient and stalled at initialization.

The validated catastrophic guard rejects only a nonfinite candidate or a
one-step global pixel-SSE increase greater than `1e6`. This threshold is an
engineering numerical-failure detector, not a convergence condition.

The mature BA implementation uses relative thresholds against accepted
references and rejects only when both its sandwiched DRE diagnostic and global
primal cost deteriorate:

$$
E_{\mathrm{trial}}>\alpha_k E_{\mathrm{ref}}
\quad\text{and}\quad
F_{\mathrm{trial}}>\beta_k F_{\mathrm{ref}},
\qquad \alpha_k,\beta_k>1.
$$

In `client_acc.py`, these factors are `maxPct` and `maxPctV`. They tighten over
the run. This relative conjunction permits nonmonotone accelerated progress
while rejecting trials for which both splitting merit and physical BA quality
become worse. Nonfinite rejection remains unconditional. This is an engineering
BA safeguard, not the line search in Themelis, Stella, and Patrinos.

## Official Douglas--Rachford Line Search

The accelerated paper points to `ProximalAlgorithms.jl`, now maintained at
`JuliaFirstOrder/ProximalAlgorithms.jl`. Its `DRLS` implementation chooses a
fixed DRS stepsize from the smoothness constant. In the nonconvex case,

$$
\gamma=\alpha\frac{2-\lambda}{2L_f},
\qquad \alpha=0.95 \text{ by default}.
$$

For $\lambda=1$, this gives $\gamma L_f=0.475$. The implementation computes
the positive theoretical decrease coefficient

$$
C=\frac{\lambda}{(1+a)^2}
\left(\frac{2-\lambda}{2}-a\right),
\qquad a=\gamma L_f,
$$

and uses $c=\beta C$ with default $\beta=0.5$. A trial is accepted when

$$
E_{\mathrm{DR}}(s_{\mathrm{trial}})
\le E_{\mathrm{DR}}(s_k)-\frac{c}{\gamma}\lVert u_k-v_k\rVert^2.
$$

Backtracking halves only the interpolation parameter $\tau$ between an
arbitrary accelerated direction and the nominal DRS point. The final
$\tau=0$ trial is the nominal DRS fallback. The official `DRLS` implementation
does not increase the proximal coupling or adapt $\gamma$ inside this line
search; it assumes a valid $L_f$ and hence an admissible fixed $\gamma$.

For the block-metric BA formulation with $\gamma=1$ absorbed into the metric,
the corresponding smoothness quantity is

$$
L_M=\lambda_{\max}
\left(M^{-1/2}\nabla^2F\,M^{-1/2}\right).
$$

The default DRLS margin targets $L_M\le0.475$ for $\lambda=1$. A global
multiplier on the complete proximal metric is therefore the clean analog of
reducing $\gamma$. The weak-direction diagonal floor is a separate numerical
regularizer and should not also serve as the Lipschitz multiplier.

The current worker entangles these roles:

$$
M=\min\left(1.005,0.1\sqrt{\mathrm{be}/\mathrm{be}_0}\right)J^\top J
+10\,\mathrm{be}\,\operatorname{blockdiag}(J^\top J).
$$

Even at the `be=0.5` ceiling, the curvature coefficient is only `1.005`. Under
the optimistic Gauss--Newton Schur bound $S\preceq J^\top J$, this gives only
$L_M\lesssim0.995$ before accounting for the diagonal term, not the DRLS target
$0.475$. This explains why increasing only `be` need not establish the
stepsize condition.

The theoretical DRE must also be distinguished from the implementation's
`max(DRE_model, f(v))` sandwich. On scene 931, iteration 6 to 7 decreases the
model DRE from approximately `556783` to `548423`. With the official defaults,
the required decrease is only about `117`, so this trial passes the formal
DRLS test even though pixel $f(v)$ rises from `661402` to `984722`. Conversely,
an unguarded 90-iteration probe showed that model-DRE decrease can coexist with
pixel SSE excursions of $10^{16}$--$10^{21}$ under the present finite-GN,
refreshed-metric oracle. Therefore the model DRE cannot simply replace the
physical primal guard until the local proximal defect and metric assumptions
are controlled.

## What Is Being Safeguarded

Each worker approximately solves a local augmented problem

$$
\min_{x_i,l_i}
f_i(x_i,l_i)
+\frac{\rho_k}{2}\|x_i-c_i^k\|^2,
\qquad c_i^k=z^k-u_i^k.
$$

Local acceptance only controls this cluster objective. The coordinator then
averages duplicated cameras and evaluates the resulting consensus together
with the assembled landmark state. This global state can have catastrophic
pixel SSE even when every worker returns a nonincreasing local augmented
objective. The Ceres experiment demonstrates this separation: Ceres LM rejects
bad local trials, yet unguarded global pixel SSE reaches `1.29e25` on 1723 and
later recovers.

The safeguard therefore protects the composition

$$
\text{local solves}\;\longrightarrow\;\text{camera consensus}
\;\longrightarrow\;\text{assembled global BA state},
$$

not merely the local nonlinear solver.

## Current Recovery: Increase the ADMM Penalty

The current implementation snapshots local cameras, landmarks, consensus,
scaled duals, and penalties. On rejection it restores the snapshot and applies

$$
\rho_{k+1}=2\rho_k,
\qquad
u_{k+1}=\frac{u_k}{2}.
$$

The unscaled multiplier is preserved:

$$
y_{k+1}=\rho_{k+1}u_{k+1}
=(2\rho_k)\frac{u_k}{2}=y_k.
$$

The larger penalty adds camera curvature and pulls local copies more strongly
toward consensus. This resembles `Be *= 2` in `client_acc.py`, but it also
changes the ADMM dual/consensus penalty. Ordinary DABA residual balancing then
resumes and may multiply the penalty by `1.5`, `0.8`, or `1.005` each accepted
iteration.

For the custom DABA trust policy, restoring the previous trust radius currently
does not help because the worker resets the radius to `100` at each local solve.
Thus the effective current recovery is rollback plus penalty doubling.

## Relation to `Be` and iPALM

`Be` in the mature DRS path is a local metric or majorization parameter. A
rejected final trial doubles it, restores the best camera/landmark state, and
resets acceleration. `LipJ` is currently unused. There is no active schedule
that later decreases `Be`; an apparent `sqrt(2)` increase in the accepted path
is overwritten by restoration from the trial snapshot.

iPALM motivates block-local partial Lipschitz estimates rather than one global
worst-case constant. When these constants are unknown, the paper permits a
standard descent-lemma backtracking estimate. The useful interpretation here
is qualitative: strengthen a local majorizer where the current path requires
it, retain it long enough to avoid immediate recurrence, and relax it after
moving into a region with lower local curvature. The iPALM paper does not
prescribe a particular geometric decay schedule for this application.

## Cleaner Variable-Metric ADMM Recovery

Separate the two roles currently assigned to `rho`:

- $\rho_k$: ADMM consensus and dual penalty;
- $\eta_k$: temporary recovery majorizer.

After a failed outer candidate, solve

$$
\min_{x_i,l_i}
f_i(x_i,l_i)
+\frac{\rho_k}{2}\|x_i-c_i^k\|^2
+\frac{\eta_k}{2}\|x_i-x_i^k\|^2,
$$

where $x_i^k$ is the last accepted local camera state. The two quadratics
combine exactly:

$$
q_k=\rho_k+\eta_k,
\qquad
\widetilde c_i^k=
\frac{\rho_k c_i^k+\eta_k x_i^k}{\rho_k+\eta_k}.
$$

The worker can therefore receive the existing scalar proximal interface with
penalty $q_k$ and center $\widetilde c_i^k$. No BA objective or wire protocol
change is required. The coordinator must continue to use $\rho_k$, not $q_k$,
for consensus weighting, scaled-dual interpretation, residual balancing, and
dual rescaling.

Initial schedule:

$$
\eta_0=0.
$$

After a catastrophic rejection:

$$
\eta\leftarrow
\begin{cases}
\rho, & \eta=0,\\
2\eta, & \eta>0.
\end{cases}
$$

After two consecutive accepted outer steps:

$$
\eta\leftarrow\frac{1}{2}\eta.
$$

Set $\eta=0$ when $\eta<0.05\rho$. This hysteresis avoids switching off the
majorizer after one lucky step while guaranteeing eventual return to ordinary
ADMM. Start with one shared scalar $\eta$; split extrinsic/intrinsic values are
a later ablation only if the scalar design succeeds.

This is cleaner for a paper because the ADMM penalty remains an ADMM parameter,
while recovery is explicitly a vanishing variable metric. Claims should still
be conservative: convergence with a state-dependent safeguard and inexact
local solves requires a separate argument. A finite number of recoveries and
$\eta_k\to0$ reduce the asymptotic method to the baseline ADMM iteration.

## Three Recovery Ablations

Use the same catastrophic trigger and exact rollback in every mode.

### A1: Penalty Doubling

- Restore the accepted state.
- Set $\rho\leftarrow2\rho$ and preserve the unscaled dual.
- Resume ordinary residual balancing.
- This is the current validated implementation and control.

### A2: Temporary Proximal Majorizer

- Keep $\rho$ unchanged on rejection.
- Set or increase $\eta$ using the schedule above.
- Send $(q,\widetilde c)$ to workers, but perform ADMM updates with $\rho$.
- Decay $\eta$ after two accepted steps and report when it returns to zero.

### A3: Persistent Trust-Radius Recovery

- Keep $\rho$ unchanged and use no additional $\eta$.
- On rejection, restore the accepted worker state and shrink the saved local
  trust radius, initially by `0.5`.
- Stop resetting the DABA radius to `100` on every call for this mode only.
- Allow accepted local steps to update the persistent radius normally.

A3 isolates whether the failure can be handled entirely inside the local
quadratic model. The Ceres results suggest that it cannot guarantee safe global
consensus, but it remains an important control.

For the first implementation, keep rejected attempts visible as consumed outer
attempts to preserve the existing trajectory and communication accounting.
After the recovery action is selected, separately test same-iteration retries
with a strict retry cap; otherwise comparisons can hide additional local work.

## Experimental Stages

### Stage 1: Five-Scene Mechanism Gate

- scenes: 1723, 52, 245, 394, 871;
- K20;
- 30 accepted/attempted outer indices under the current accounting;
- exactly one local nonlinear update;
- one CPU thread per cluster;
- Nesterov Schur solve with DABA trust policy;
- initial Jacobi variable substitution and `alpha = 1`;
- standard Snavely pixel SSE.

Compare unguarded, A1, A2, and A3. Report best and endpoint SSE, maximum
candidate excursion, recovery count, runtime, communication, $\rho_k$,
$\eta_k$, $\eta_k/\rho_k$, trust radius, and consecutive stable steps.

Acceptance gate for A2 or A3:

- no best-SSE regression above run noise on the four previously stable scenes;
- finite, recoverable 1723 endpoint;
- fewer than five recoveries in 30 iterations;
- no material runtime increase beyond explicit repeated work;
- saved state independently reproduces recorded pixel SSE.

### Stage 2: Partition-Stress Sweep

Only mechanisms passing Stage 1 proceed to K10, K20, and K30 on the same five
scenes. Larger K generally creates smaller local subproblems and more weak
camera-cluster incidences; K30 is therefore the first stress setting. K10 is a
lower-fragmentation control. Record partition diagnostics alongside solver
metrics:

- observation counts per cluster;
- camera copies and maximum cameras per cluster;
- camera-cluster landmark-count histogram;
- counts below 5, 10, and 20 landmarks;
- partition time and deterministic partition hash.

Do not mix recovery and partition innovations in one claim. First compare
recovery modes under the same partition; then compare partitioners with the
chosen recovery fixed.

### Stage 3: Broader Scene Set

Expand to the existing ten-scene list only after Stage 1 and the K30 stress
gate pass. Use repeated runs for timing claims. Keep Ceres with and without the
same global guard as contextual controls because the guard improved Ceres on
1723 as well.

## What DABA Partitions

DABA constructs an unweighted bipartite graph whose vertices are all cameras
and landmarks and whose edges are observations. It runs GPU Louvain community
detection with modularity optimization, bounded cluster scores, and merge or
decompose steps until the requested cluster count is reached.

Both camera and landmark vertices initially receive Louvain labels. DABA then
reassigns each landmark to the cluster containing the largest total weight of
its observing cameras. Consequently every camera and landmark has one owner
cluster. An observation is stored in the owner cluster of each endpoint; when
camera and landmark owners differ, that observation is replicated into both
clusters. This differs from saying that DABA simply partitions cameras or
shares every landmark globally.

DABA's default non-memory-efficient balance score counts one unit per camera,
with an approximate maximum of `number_of_cameras / K` per cluster. Its
memory-efficient mode instead scores camera degree and caps approximate edge
load. The structural objective is modularity/internal edge weight plus these
score limits. DABA does not explicitly optimize weak camera-cluster rank,
minimum landmarks per local camera, or exact residual balance.

Our active `cluster_by_landmark_scalable_stable` path assigns every landmark to
exactly one cluster and places all of that landmark's observations there.
Cameras are copied into every cluster containing one of their landmarks. The
partition objective lexicographically prioritizes severe weak-camera counts,
weak-camera counts and penalties, maximum cameras, camera copies, and residual
imbalance. It is deterministic and CPU-based.

Neither partitioner is uniformly better from its objective alone:

- DABA may produce communities with fewer cross-owner observations and better
  graph locality, and its GPU Louvain search is less locally greedy.
- Our partitioner directly targets the weak camera blocks that caused custom
  Schur instability and explicitly balances observations.
- DABA balances owned cameras or edge score, while our actual worker memory and
  runtime depend on replicated cameras and owned observations. These are
  different load models.

A fair partition comparison must convert both outputs into the same local BA
representation and measure the resulting observation balance, camera copies,
weak-camera histogram, worker memory, communication, convergence, and
partition time. Comparing modularity with our lexicographic objective is not
sufficient.

## Partition Comparison Plan

1. Export DABA's camera and final landmark owner labels for the five scenes and
   K10/K20/K30.
2. Construct two explicit local-problem interpretations:
   DABA's endpoint-owner observation replication and our landmark-owned
   observation assignment.
3. Evaluate both with common diagnostics before running an optimizer.
4. If DABA ownership is numerically viable for our worker contract, run the
   selected recovery mode with all other solver settings fixed.
5. Report partition quality separately from optimization quality.

The likely useful hybrid is not to copy DABA wholesale, but to use Louvain
community labels as an initialization for our constrained landmark refinement.
That could retain graph locality while repairing weak-camera and load-balance
violations explicitly.

## Stage 1 Results

The K20, five-scene, 30-iteration A1/A2/A3 experiment completed all 15 cases.

| Recovery | Geomean best SSE | Geomean endpoint SSE | Recoveries | Total time s | Gate |
|---|---:|---:|---:|---:|---|
| A1: double ADMM penalty | 3,110,876 | 3,518,342 | 1 | 99.47 | pass |
| A2: temporary proximal majorizer | 3,110,876 | 3,805,414 | 1 | 100.28 | pass |
| A3: persistent trust radius | 3,110,876 | 5,360,211 | 10 | 99.84 | fail |

All three methods reproduced the same best SSE and best iteration on every
scene. Scenes 52, 245, 394, and 871 were identical across modes and required no
recovery.

On 1723:

- A1 recovered once and ended at `30,842,287` pixel SSE.
- A2 recovered once and ended at `45,652,681` pixel SSE. The temporary
  majorizer was positive for four recorded iterations and then returned to
  exactly zero under the predeclared decay schedule.
- A3 recovered ten times and ended at `253,143,019` pixel SSE. It fails the
  fewer-than-five recovery gate.

A3 was carefully isolated: its trajectory is bit-for-bit identical to A1
through the first rejected candidate. Trust-radius persistence activates only
after that rejection, and repeated shrinkage is cumulative. Its failure is
therefore evidence that local trust-radius control alone does not repair the
global consensus incompatibility.

A2 validates the cleaner algorithmic separation: the ADMM penalty remains on
its ordinary residual-balance path while a vanishing metric handles recovery.
It is not yet superior numerically to A1 on 1723. The next test should compare
A1 and A2 at K10 and K30 before tuning A2's decay; tuning the schedule only on
1723 would weaken the ablation.

Artifacts:

- `benchmark_results/admm_recovery_ablation_i30_k20/report.md`
- `benchmark_results/admm_recovery_ablation_i30_k20/status.tsv`
- `benchmark_results/admm_recovery_ablation_i30_k20/*.jsonl`

## Stage 2 Results: K10 and K30

The predeclared A1/A2 stress matrix completed all 20 cases. A3 was excluded.

| Recovery | K | Geomean best SSE | Geomean endpoint SSE | Recoveries | Total time s |
|---|---:|---:|---:|---:|---:|
| A1: double ADMM penalty | 10 | 3,025,954 | 24,722,276 | 0 | 97.30 |
| A2: temporary proximal majorizer | 10 | 3,025,954 | 24,722,276 | 0 | 97.49 |
| A1: double ADMM penalty | 30 | 2,578,232 | 2,674,900 | 0 | 102.47 |
| A2: temporary proximal majorizer | 30 | 2,578,232 | 2,674,900 | 0 | 101.92 |

A1 and A2 are bit-for-bit identical because neither guard fires. The important
counterexample is 1723/K10. It improves from `1.24e8` to `2.16e7`, then jumps
to `1.51e13` at iteration 2. This is `6.99e5` times the best state, just below
the `1e6` one-step trigger. Subsequent one-step ratios are much smaller, so the
bad state remains accepted and endpoint SSE is `7.85e11`.

Meanwhile the ADMM primal residual falls from `24.9` to `4.1e-8`, the dual
residual falls from `19.9` to `5.7e-7`, and the adaptive penalty rises from
`788` to `1.01e8`. Thus the algorithm is successfully enforcing camera
agreement around a catastrophically bad BA state. Residual convergence and
penalty growth cannot diagnose objective-basin loss.

At K30, 1723 has large early transients but recovers to best SSE `5.16e6` and
endpoint `6.20e6`. K30 also has more weak camera-cluster incidences than K10,
so weak incidence count alone does not predict optimization quality. Graph
locality and the identities of weak camera copies matter.

All 20 saved states independently reproduce their recorded standard pixel SSE.

## Why Mature DRS Does Not Expose The Explosion

The supplied 1723/K10 `client_acc.py` trace does contain catastrophic trial
states: approximately `1.41e10`, `1.10e10`, `1.67e12`, and `1.97e8` pixel SSE.
They do not become accepted outer states because mature DRS has four layers
absent from the current ADMM safeguard:

1. It evaluates both the Douglas-Rachford envelope and the global primal pixel
  objective. The envelope is sandwiched from below by the primal value.
2. It tries an accelerated center and then a plain-DRS fallback. A failed
  accelerated proposal can therefore be retried before abandoning the state.
3. If the final trial makes both merit and primal cost sufficiently worse, it
  restores globally best cameras and their associated landmarks, not merely
  the immediately previous state.
4. It doubles the block metric `Be` and resets Nesterov momentum. In the trace,
  `Be` progresses from `5e-5` to `1e-4`, `2e-4`, `4e-4`, and `8e-4` after
  rejected trials.

DRS also uses a full block-metric consensus

$$
v=\left(\sum_i D_i\right)^{-1}\sum_i D_i(2u_i-s_i),
$$

whereas the current scalar-penalty ADMM uses arithmetic camera consensus. The
block metric downweights weak camera directions and is likely an additional
stabilizer. It should be isolated separately from the safeguard.

Therefore `client_acc.py` does not prove that its local solver never explodes;
it proves that its two-trial merit safeguard catches and repairs those
explosions before acceptance.

## Why Published DABA Does Not Use An ADMM Penalty Safeguard

The published DABA solver in `third_party/DABA` is not the consensus ADMM
implemented in `client_admm.py`. It has no `rho`, scaled dual, residual-balance
rule, or dual rescaling. It is decentralized majorization-minimization with
gossip communication, local trust-region refinement, and Nesterov acceleration.
The separate `examples/admm` directory is not the published DABA BA solver.

It also does not optimize the standard Snavely pixel objective used by our
primary experiments. DABA evaluates a weighted regularized 3D ray error. Its
projection-free residual subtracts the component of a rotated observation ray
along the camera-to-point displacement and regularizes the denominator with
`delta = 1e-6`. It never divides by camera depth as pixel projection does.
Consequently a camera approaching zero depth can create enormous standard
pixel SSE while remaining much less singular in DABA's ray objective. This
objective-conditioning difference is a major stability confounder and must not
be attributed to partitioning or the optimizer alone.

DABA has its own safeguards:

- each local LM call accepts only actual surrogate decrease and shrinks its
  trust radius after rejected or invalid trials;
- proximal operators include a block regularizer of `1e-3` for extrinsics,
  intrinsics, and points;
- the nonaccelerated majorization update compares its proximal candidate with
  the maintained cost and starts trust-region refinement from the better state;
- the accelerated candidate is accepted only when its distributed surrogate
  cost is below an exponential moving-average threshold; otherwise DABA calls
  the nonaccelerated `Update()` path, which is an adaptive restart.

The example initializes that moving-average threshold at `250` times the
initial maintained cost and uses `eta=1e-3`, increasing eta by `5e-4` relative
per iteration. This is intentionally permissive early and gradually tightens,
similar in spirit to an armed basin safeguard rather than a monotone objective
test.

DABA's default `1000` outer iterations are not caused by a large ADMM penalty:
there is no ADMM penalty. They provide a long convergence horizon for a
first-order majorization/gossip method with one accepted local LM step per
outer iteration. DABA computes its maintained distributed cost each iteration,
MPI-sums it across ranks, records every value, and prints every twentieth value.

We cannot yet claim that DABA never has bad standard-pixel transient states.
Its restart tests its own distributed surrogate objective, not our independently
assembled standard-pixel consensus state, and the published evaluation does not
save an independently evaluated physical state at every iteration. That must be
measured by instrumenting a runnable DABA build.

Partition quality may help DABA by improving graph locality, but it is not the
primary explanation for the algorithmic stability difference. DABA uses a
different and better-conditioned ray objective, a different update map,
majorizing surrogates, a block regularizer, local LM rejection, and acceleration
restart.

## Next Safeguard Ablation

The current `1e6` one-step trigger catches abrupt numerical explosions but
misses persistent basin loss. It should remain only a last-resort numerical
check. A more faithful analogue of mature DRS requires a relative splitting
merit together with relative global primal cost:

$$
	ext{reject}
\iff
E_{\mathrm{trial}}>\alpha_kE_{\mathrm{accepted}}
\ \land\ 
F_{\mathrm{trial}}>\beta_kF_{\mathrm{accepted}}.
$$

For DRS, $E$ is the Douglas--Rachford envelope. For ADMM, the corresponding
merit must be derived or selected carefully; primal and dual residual norms
alone are insufficient because K10/1723 converges in both residuals around a
bad BA state. Candidates include an augmented-Lagrangian merit evaluated at a
consistent global state or a variable-metric envelope for the equivalent DRS
formulation. This is the clean theoretical direction.

Before that merit is available, a two-scale stateful engineering ablation can
test basin protection:

1. remain permissive until an iterate improves on initialization and thereby
  establishes a good basin;
2. retain the `1e6` candidate/previous-state catastrophic test;
3. additionally reject candidates above `1e4` times the best accepted pixel
  SSE after the guard is armed.

Counterfactual screening on all five scenes at K10/K20/K30 produced no trigger
on 52, 245, 394, or 871. On 1723 it would reject K10 at iteration 2, while
allowing the useful K20/K30 iteration-2 transients (`1.87e3` and `4.09e3` times
best) and rejecting their larger iteration-3 excursions. This is only a
screening result. The best-state ratio is still relative, but it is not a
substitute for the relative two-merit acceptance test. K30 currently recovers
naturally, so the armed guard must be run as an ablation before promotion.

## Source Anchors

- `serverTest/client_admm.py`: current outer guard and penalty recovery.
- `serverTest/admm_consensus.py`: residual-balanced ADMM penalty schedule.
- `serverTest/client_acc.py`: mature DRS rollback and `Be` update.
- `block_metric_consensus_derivation.md`: paper-ready metric projection,
  gradient sign convention, novelty scope, and relative safeguard statement.
- `serverTest/main.cpp`: scalar proximal term and custom trust policies.
- `iPalm.pdf`: block-local partial Lipschitz constants and backtracking context.
- `third_party/DABA/sfm/ba/clustering/clustering.cu`: Louvain, score bounds,
  decomposition, and merging.
- `third_party/DABA/sfm/ba/dataset.cu`: final ownership and observation
  replication.
- `third_party/DABA/sfm/ba/clustering/common_methods.cu`: bipartite graph
  construction with unit observation edges.
- `landmark_partitioning.cpp`: our constrained landmark partition objective.
- `serverTest/clustering.py`: active Python partition wrapper.