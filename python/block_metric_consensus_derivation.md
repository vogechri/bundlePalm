# Block-Metric Camera Consensus for Distributed Bundle Adjustment

Date: 2026-07-27

## Paper Position

The weighted projection formula itself is not new. It is the metric projection
onto a consensus subspace and follows directly from variable-metric proximal
splitting. The paper should not claim invention of weighted consensus or
variable-metric DRS.

The candidate BA-specific contribution is the complete design:

1. assign each complete landmark track to one worker;
2. duplicate only cameras needed by several landmark owners;
3. solve the original nonlinear local BA objectives;
4. retain each local camera's full regularized $9\times9$ curvature block;
5. use those blocks in the global camera-consensus projection; and
6. communicate only camera states and compact camera metrics between outer
   nonlinear solves.

This is a defensible application and systems-algorithm contribution if
arithmetic, diagonal, and full-block ablations show a material stability,
quality, or time-to-target advantage. A claim that no prior BA method uses this
construction requires a dedicated literature search. Until that search is
complete, use wording such as "a BA-specific full block-metric consensus
construction" rather than "the first block-metric consensus BA method."

## Consensus Subproblem

For a camera $c$, let $\mathcal I_c$ be the workers containing a local copy.
Worker $i$ has local camera $u_{i,c}$, splitting center $s_{i,c}$, and symmetric
positive-definite metric block $D_{i,c}\in\mathbb R^{9\times9}$.

The reflected point is

$$
r_{i,c}=2u_{i,c}-s_{i,c}.
$$

Projection onto camera consensus solves

$$
v_c
=\arg\min_v
\frac12\sum_{i\in\mathcal I_c}
\|v-r_{i,c}\|_{D_{i,c}}^2,
\qquad
\|x\|_D^2=x^TDx.
$$

Its first-order condition is

$$
\sum_{i\in\mathcal I_c}D_{i,c}(v_c-r_{i,c})=0,
$$

and therefore

$$
\boxed{
v_c=
\left(\sum_{i\in\mathcal I_c}D_{i,c}\right)^{-1}
\sum_{i\in\mathcal I_c}D_{i,c}(2u_{i,c}-s_{i,c})}.
$$

Only one $9\times9$ system is solved per shared camera. The formula is
coordinate invariant under a consistent linear camera reparameterization.

## Equivalent Gradient Form

Consider the more general consensus objective

$$
\phi_c(v)=
\sum_{i\in\mathcal I_c}
\left[
\frac12\|u_{i,c}-v\|_{D_{i,c}}^2
-\langle g_{i,c},u_{i,c}-v\rangle
\right].
$$

Differentiating with respect to $v$ gives

$$
\nabla_v\phi_c(v)=
\sum_{i\in\mathcal I_c}
\left[D_{i,c}(v-u_{i,c})+g_{i,c}\right].
$$

Hence

$$
\boxed{
v_c=
\left(\sum_iD_{i,c}\right)^{-1}
\sum_i\left(D_{i,c}u_{i,c}-g_{i,c}\right)}.
$$

The sign is important. In the current DRS notation,

$$
g_{i,c}=D_{i,c}(s_{i,c}-u_{i,c}),
$$

so

$$
D_{i,c}u_{i,c}-g_{i,c}
=D_{i,c}(2u_{i,c}-s_{i,c}),
$$

which recovers the reflected-state formula. If a paper defines the stored
gradient with the opposite sign, the displayed general formula must change
accordingly. The reflected-state expression is unambiguous and should be the
primary presentation.

For scalar metrics $D_{i,c}=\rho_{i,c}I$, this reduces to

$$
v_c=
\frac{\sum_i\rho_{i,c}(2u_{i,c}-s_{i,c})}
     {\sum_i\rho_{i,c}}.
$$

If all scalar weights are equal, it reduces further to arithmetic averaging.

## Why A Single Scalar Is Weak For BA

The nine Snavely camera coordinates have different units, scales, and local
observability. More importantly, one camera copy may constrain a direction
strongly while another copy is nearly rank deficient in that same direction.
A scalar weight treats all directions and all local geometries equally.

With full blocks, a local copy contributes strongly in well-observed camera
directions and weakly in poorly observed directions. Off-diagonal entries also
retain coupling between rotation, translation, focal length, and radial
distortion. This is not achieved by:

- arithmetic averaging;
- one shared scalar ADMM penalty;
- separate extrinsic/intrinsic scalar penalties; or
- diagonal coordinate scaling alone.

The metric must be regularized and expressed in the same coordinates on every
worker. If $y=Tx$ is a camera coordinate change, then states and blocks must be
transformed consistently by congruence; otherwise the consensus has no coherent
physical interpretation.

## Relationship To ADMM

Standard consensus ADMM with scalar penalty solves an arithmetic or
scalar-weighted $z$ update. A matrix-penalty ADMM can use positive-definite
penalties $P_{i,c}$ and obtains

$$
z_c=
\left(\sum_iP_{i,c}\right)^{-1}
\sum_iP_{i,c}(x_{i,c}+u_{i,c}),
$$

with the precise sign determined by the scaled-dual convention. Thus
matrix-weighted consensus is also known in preconditioned or variable-metric
ADMM. What must be tested here is whether BA-derived $9\times9$ blocks remain
stable when used as ADMM penalties, particularly when blocks are refreshed.

The current mature implementation is more naturally described as
variable-metric DRS because its projection acts on reflected points
$2u_i-s_i$ and its DRE safeguard uses the same metric.

## Paper-Ready Contribution Statement

Suggested wording:

> We introduce a BA-specific variable-metric consensus construction for a
> landmark-owned nonlinear decomposition. Each worker retains a regularized
> full camera curvature block from its local BA solve, and shared cameras are
> reconciled by the induced metric projection. Unlike arithmetic or scalar
> penalty consensus, this projection accounts for direction-dependent local
> observability while requiring no collective operation inside the local Schur
> solve.

This statement does not claim that metric projection itself is new.

## Required Ablation

Hold partition, objective, local solver, scaling, safeguard, and outer method
fixed. Compare:

1. arithmetic consensus;
2. one scalar weight per camera copy;
3. diagonal $9$-parameter metric;
4. full regularized $9\times9$ block metric.

Report best and endpoint pixel SSE, accepted and rejected trials, weak-camera
failures, metric condition numbers, time, payload bytes, and outer rounds.
The innovation is supported only if the full metric provides a repeatable
advantage beyond coordinate scaling and additional communicated bytes.

## Relative Safeguard

The mature DRS acceptance test is relative, not an absolute global-cost
threshold. For trial merit $E_t$, reference merit $E_r$, trial primal cost
$F_t$, and reference primal cost $F_r$, reject only when both

$$
E_t>(1+\delta_E)E_r
\quad\text{and}\quad
F_t>(1+\delta_F)F_r.
$$

In `client_acc.py`, the factors are `maxPct` and `maxPctV`; they tighten over
the run. This conjunction is important: DRE may be nonmonotone under
acceleration, and the primal objective may transiently rise, but simultaneous
relative deterioration is evidence that the trial left the useful basin.

Nonfinite-state rejection remains unconditional. Any coarse catastrophic cap
used in diagnostics should be presented as a last-resort numerical check, not
as the primary acceptance rule.