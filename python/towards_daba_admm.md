# Towards a DABA-Style ADMM Baseline

## Scope

This note explains DABA's consensus variables, dual variables, adaptive
penalties, and local Levenberg-Marquardt (LM) solve. It also states what those
choices mean for our CPU ADMM implementation.

Our primary optimization and evaluation objective remains the standard
Snavely pixel reprojection SSE:

$$
F(c,p)=\sum_i\left[(\hat x_i-x_i)^2+(\hat y_i-y_i)^2\right].
$$

DABA's weighted-ray objective is not proposed as a replacement. The ADMM
mechanics below can be applied while preserving our pixel objective.

## Short Answers

### What does separate extrinsic and intrinsic consensus mean?

DABA splits each camera into two consensus groups:

- extrinsics $e\in\mathbb{R}^6$: angle-axis rotation and translation;
- intrinsics $q\in\mathbb{R}^3$: focal/ray scale and two radial parameters.

For every camera copy, it stores separate local values, consensus references,
and scaled duals:

$$
(e_k,z_e,u_{e,k}),\qquad(q_k,z_q,u_{q,k}).
$$

The consensus operation itself is the same arithmetic averaging for both
groups. DABA sums the copies of each shared camera and divides by its occurrence
count. "Separate consensus" therefore does not mean a special averaging rule.
It means that extrinsics and intrinsics are separate state blocks and can have
different penalties and convergence behavior.

### Does that allow different strengths?

Yes. The local augmented objective contains two quadratic terms:

$$
\min_{e_k,q_k,p_k}
F_k(e_k,q_k,p_k)
+\frac{\rho_e}{2}\|e_k-z_e+u_{e,k}\|^2
+\frac{\rho_q}{2}\|q_k-z_q+u_{q,k}\|^2.
$$

Here $\rho_e$ controls extrinsic consensus strength and $\rho_q$ controls
intrinsic consensus strength.

- Larger $\rho_e$: rotation/translation copies are forced toward consensus
  more strongly, but local reprojection improvement in those coordinates is
  more restricted.
- Smaller $\rho_e$: local extrinsics can move more freely, but agreement is
  restored more slowly.
- The same interpretation holds for $\rho_q$ and intrinsics.

The released BAL executable initializes both to the same value:

$$
\rho_e^0=\rho_q^0=2.5\frac{N_{\mathrm{observations}}}{N_{\mathrm{cameras}}}.
$$

They can subsequently become different because they are adapted independently.
Thus this is not initially hand-tuned as "intrinsics weaker" or "extrinsics
stronger," although the automatic process may produce that outcome.

## Separate Duals and Separate Penalties

### What is a dual here?

DABA stores a scaled dual value for every duplicated coordinate:

$$
u_{e,k}\in\mathbb{R}^6,\qquad u_{q,k}\in\mathbb{R}^3.
$$

After consensus, its over-relaxed update is

$$
u_{e,k}^{+}=u_{e,k}+1.5(e_k-z_e),$$

$$
u_{q,k}^{+}=u_{q,k}+1.5(q_k-z_q).$$

"Separate duals" means extrinsic disagreement and intrinsic disagreement
accumulate in different arrays. It does **not** mean that every coordinate has
an independently tuned scalar weight. The dual is vector-valued, while the
penalty is scalar within each group.

The corresponding unscaled multipliers are

$$
y_{e,k}=\rho_e u_{e,k},\qquad y_{q,k}=\rho_q u_{q,k}.
$$

When a penalty changes by a factor $r$, DABA divides the scaled dual by $r$:

$$
u^+\leftarrow u^+/r.$$

This preserves the unscaled multiplier $y=\rho u$. Without this rescaling, a
penalty update would also produce an unintended jump in dual force.

### Can one penalty be smaller than the other?

Yes. After adaptation, any of the following can occur:

$$
\rho_e>\rho_q,\qquad \rho_e<\rho_q,\qquad \rho_e=\rho_q.
$$

A smaller penalty is less restrictive for that block during the local solve.
It is not "ignored": its dual still accumulates disagreement, and later
adaptation can increase it.

## Independently Adapted Penalties

DABA computes separate residual statistics for extrinsics and intrinsics.
For a group $g\in\{e,q\}$, its code uses

$$
r_g^2=\|x_g^{\mathrm{previous}}-z_g^{\mathrm{previous}}\|^2,
$$

$$
s_g^2=\|z_g^{\mathrm{current}}-z_g^{\mathrm{previous}}\|^2.
$$

It then selects a multiplier independently for each group:

$$
a_g=
\begin{cases}
1.5,
& \rho_g^0 r_g^2>2.5s_g^2,\\
0.8,
& s_g^2>10\rho_g^0r_g^2,\\
1.005,
& \text{otherwise}.
\end{cases}
$$

and updates

$$
\rho_g\leftarrow a_g\rho_g,
\qquad
u_g\leftarrow u_g/a_g.
$$

This is automatic residual balancing, not per-dataset hand tuning. The
constants are fixed algorithm choices in the released executable:

- initial scale $2.5N_{\mathrm{obs}}/N_{\mathrm{cam}}$;
- increase factor $1.5$;
- decrease factor $0.8$;
- default drift factor $1.005$;
- threshold constants $2.5$ and $10$.

"Independently adapted" means that the extrinsic test uses only extrinsic
residuals and changes only $\rho_e$, while the intrinsic test uses only
intrinsic residuals and changes only $\rho_q$. One can increase while the
other decreases.

This differs from our first CPU baseline, which applied one penalty update to
all nine raw Snavely camera parameters. Rotation, translation, focal length,
and radial distortion have very different numerical scales, so one combined
residual norm can be dominated by only part of the camera vector.

## Relationship to Jacobi Prescaling

Jacobi prescaling is related to, but more fine-grained than, DABA's two scalar
penalties. Let scaled coordinates be

$$
y=Dx,$$

where $D$ is diagonal. A scalar penalty in $y$ is equivalent in physical
coordinates to

$$
\frac{\rho}{2}\|D(x-z)+u_y\|^2
=
\frac12\|x-z+u_x\|_{P}^2,
\qquad
P=\rho D^TD.
$$

Therefore diagonal Jacobi scaling induces a different effective penalty for
every coordinate:

$$
P_{jj}=\rho d_j^2.
$$

DABA instead has the coarser block metric

$$
P_{\mathrm{DABA}}=
\operatorname{diag}(\rho_e I_6,\rho_q I_3).
$$

So:

- DABA: two automatically adapted strengths;
- diagonal Jacobi: up to nine curvature-derived strengths per camera;
- full block scaling: a $9\times9$ positive-definite metric that may also
  couple coordinates.

Prescaling is not the same as separate dual arrays. It becomes equivalent to a
metric penalty only when the primal state, consensus state, and scaled dual
are all transformed consistently.

## Does DABA Perform Several LM Steps?

Yes. In the released BAL ADMM executable, one ADMM outer iteration calls a
nonlinear local solve configured with:

- at most 20 LM attempts;
- at most 15 accepted LM steps;
- Schur-PCG with at most 400 iterations;
- PCG relative reduction tolerance $0.1$.

During this local solve, the ADMM quantities $z_e,z_q,u_e,u_q,\rho_e,\rho_q$
remain fixed. The same local proximal subproblem is being solved more
accurately; DABA is not changing the ADMM objective between LM steps.

### Why relinearize if the penalty is quadratic?

The consensus penalty is already quadratic and exact:

$$
\frac{\rho}{2}\|x-z+u\|^2.
$$

Its gradient and Hessian are

$$
\nabla=\rho(x-z+u),\qquad \nabla^2=\rho I,
$$

so that part does not need approximation. But the reprojection term
$F_k(e,q,p)$ is nonlinear. At LM step $j$, DABA linearizes the pixel/ray
residual at the current local state $x_j$:

$$
r(x_j+\Delta)\approx r(x_j)+J_j\Delta.
$$

It solves the damped linearized system, accepts or rejects the proposed state,
and, after an accepted move, relinearizes $F_k$ at the new state. The fixed
quadratic penalty is simply added exactly to every linear system:

$$
(H_j+\rho I+\lambda_j D_j)\Delta
=-(g_j+\rho(x_j-z+u)).
$$

Thus multiple LM steps matter because they track the nonlinear reprojection
geometry while approximately solving one fixed proximal subproblem.

### Is this also true for DRS?

Yes. For a fixed DRS center $s$ and metric $P$, the local prox is

$$
\operatorname{prox}^{P}_{F}(s)
=
\arg\min_x F(x)+\frac12\|x-s\|_P^2.
$$

Several LM steps may be used to solve this same nonlinear prox problem. The
quadratic term remains identical while $F$ is relinearized after accepted
moves. The outer DRS state/center must not change until that local prox call is
finished.

There is therefore no mathematical conflict between a fixed DRS penalty and
multiple LM steps. It is an inexact-versus-more-exact prox choice. More local
steps increase work and can improve outer stability, but excessive accuracy
may waste computation early in the outer solve.

### Why can several local LM steps ever be faster?

They are not faster per local solve. Every accepted nonlinear LM step normally
requires another Jacobian evaluation, linearization, and linear solve. The
possible speedup is only at the complete distributed-algorithm level:

1. All workers perform their local solves in parallel, so wall time is closer
   to the slowest worker than to the sum of worker times.
2. A DABA worker uses custom GPU Jacobian, Schur, and PCG kernels on a smaller
   partition. Local arithmetic may be relatively cheap compared with global
   communication and synchronization.
3. Several local steps can reduce the number of ADMM outer iterations and
   communication rounds required to reach a target quality.
4. The configured 20 LM iterations and 15 accepted iterations are maxima.
   Gradient, update-norm, relative-decrease, or trust-region tests may stop the
   local solve earlier.
5. Its PCG solve is deliberately inexact (`relative_reduction_tol = 0.1`), so
   an LM iteration is not necessarily a high-accuracy linear solve.

The tradeoff is therefore

$$
T_{\mathrm{total}}
\approx N_{\mathrm{outer}}
\left(T_{\mathrm{communication}}
      +\max_k T_{\mathrm{local},k}\right).
$$

Increasing local work is beneficial only if the reduction in outer iterations
and communication outweighs the extra local linearizations. DABA's design does
not imply that 15 accepted local steps are optimal on a CPU or on every
dataset.

### Why one local step can be better for our DRS implementation

With one inexact LM/Gauss-Newton step, the current outer iteration uses a local
quadratic model at $x_j$. The next outer iteration or line-search trial then
linearizes again at a new point. This interleaves nonlinear progress with fresh
consensus information rather than accurately solving a local prox whose center
may soon change.

That can be more efficient when:

- linearization is expensive;
- communication is local and relatively cheap;
- the outer method is stable with inexact prox evaluations;
- acceleration and line search already trigger additional prox trials;
- high local accuracy early in the solve does not reduce outer iterations
  enough to repay its cost.

There is also a proof-level distinction. A one-step GN/LM update constructs
one quadratic model at the current outer/local state $x_k$ and applies its
step once. Its model error can be related directly to derivatives at $x_k$.
With several accepted inner steps,

$$
x_k^{(0)}=x_k,
\qquad
x_k^{(j+1)}=x_k^{(j)}+\Delta_k^{(j)},
$$

the later Jacobians are evaluated at worker-local points $x_k^{(j)}$ while the
outer consensus reference and dual remain those of outer iteration $k$. A
multi-step proof must control the accumulated model errors and cross terms
between all $\Delta_k^{(j)}$; it is not obtained by simply repeating a
one-step inequality.

The current convergence draft therefore says:

- exact fixed-metric DRS is covered when the local prox is exact, regardless
   of how many numerical iterations compute it;
- finite GN/LM solves are covered only conditionally through a vanishing,
   summable, relative, or sufficient-decrease proximal-defect condition;
- an earlier ADMM--GN derivation provides a useful one-step bound;
- the corresponding multi-step cross terms and connection to the implemented
   metric still require a fresh proof.

Thus one local step is especially attractive here, but it is not yet proved
convergent merely because it is one step. The missing bridge is to show that
its proximal optimality defect

$$
e_k=\nabla F(\widehat u_k)+\gamma^{-1}M(\widehat u_k-s_k)
$$

satisfies the chosen inexact-DRS forcing condition. Empirically, our K20 tests
also show that one step is substantially more stable and faster than 20 local
LM iterations.

Our mature DRS code uses `innerIts = 1`. Its two-trial line search can still
perform almost two local prox batches per outer iteration, but each batch uses
one local nonlinear step. The historical count of 30 DRS outer iterations can
therefore represent about 58 local linearization batches, rather than 30.

In contrast, the recent plain CPU ADMM baseline was launched with
`localSteps = 20`. The Ceres worker consequently allows up to 20 local LM
iterations inside every worker request before the next consensus update. This
matches DABA's maximum-attempt count more closely, but it is expensive on the
CPU and must not be assumed optimal. A fair work comparison should report at
least:

- outer iterations and communication rounds;
- actual accepted/rejected LM steps or Jacobian evaluations;
- total local linear solves/PCG iterations;
- wall time and time to a common global pixel-SSE target.

The correct conclusion is not that multiple LM steps are intrinsically fast.
They are a communication-versus-local-compute tradeoff. Our observation that
one local step was more effective for mature DRS is fully compatible with
DABA choosing more local work on distributed GPUs.

## DABA Source Behavior

The released code establishes the following:

1. Camera extrinsics are represented as six values: SO(3) logarithm/angle-axis
   plus translation. Intrinsics are three separate values.
2. Extrinsic and intrinsic consensus references are formed by componentwise
   arithmetic averaging over camera occurrences.
3. The local surrogate adds separate Euclidean quadratic penalties and adds
   $\rho_e I$ or $\rho_q I$ to the corresponding camera Hessian coordinates.
4. Scaled duals are updated with over-relaxation $1.5$.
5. The two penalties are adapted from separate residual statistics and scaled
   duals are rescaled to preserve unscaled multipliers.
6. Local LM repeatedly relinearizes only the nonlinear reprojection term; the
   ADMM quadratics remain fixed during the local solve.

Relevant implementation files:

- `third_party/DABA/examples/mpi_admm_bal_dataset.cu`
- `third_party/DABA/examples/admm/admm_problem.h`
- `third_party/DABA/examples/admm/admm_problem.cu`
- `third_party/DABA/sfm/ba/macro.h`

## Consequences for Our CPU Pixel-Objective ADMM

To move toward DABA's ADMM mechanics without changing our standard pixel
objective:

1. Split camera consensus into raw Snavely extrinsics `[0:6]` and intrinsics
   `[6:9]`.
2. Maintain separate scaled-dual blocks $u_e$ and $u_q$ (they may share one
   array in memory, but updates and diagnostics must be block-specific).
3. Maintain $\rho_e$ and $\rho_q$ separately.
4. Compute and apply residual balancing separately for the two blocks.
5. Rescale only the matching scaled dual when its penalty changes.
6. Keep the standard Snavely pixel residual as the data objective and as the
   independently reported global metric.
7. Use several LM steps to solve each fixed local augmented subproblem, with
   the consensus references, duals, and penalties held constant throughout the
   local solve.
8. Compare this two-penalty baseline with a consistently transformed Jacobi
   version. The latter is a distinct innovation because it induces
   coordinate-level curvature-aware weights beyond DABA's two scalar groups.

## Recommended Ablation Order

A clean sequence is:

1. one raw-coordinate scalar penalty over all nine camera parameters;
2. separate extrinsic/intrinsic scalar penalties and adaptation;
3. diagonal Jacobi prescaling with consistently transformed primal,
   consensus, and dual states;
4. full $9\times9$ block metric;
5. adaptive local LM accuracy/work.

All rows must continue to report only independently evaluated standard pixel
reprojection SSE and matching mean pixel error at the agreed checkpoints.
