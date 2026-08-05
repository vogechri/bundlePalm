# Plain DRS Coordinator

`client_drs.py` is a separate, readable product-space Douglas-Rachford
coordinator. It reuses the existing worker transport and local proximal solver,
but it does not use ADMM dual variables, ADMM consensus updates, residual
balancing, or ADMM outer acceleration.

## Iteration and notation

The implementation follows `drs_gn_convergence_draft.tex` and Themelis's
fixed-metric convention:

\[
 u_k \in \operatorname{prox}_{\gamma F}(s_k),\qquad
 v_k = P_{\mathcal C}(2u_k-s_k),\qquad
 s_{k+1}=s_k+\lambda(v_k-u_k),\quad \lambda\in(0,2).
\]

The coordinator supports two local proximal metrics. With
`--proximal-metric scalar`, the worker uses

\[
 \rho=\gamma^{-1}=m\,2.5\,N_{\mathrm{obs}}/N_{\mathrm{cam}},
\]

where `m` is `--penalty-multiplier`. On a rejected catastrophic candidate the
coordinator restores the previous accepted `(s,u,v,landmarks)` state and
multiplies `rho` by `--recovery-penalty-ratio`. This recovery changes the DRS
stepsize and is reported explicitly; it is not part of exact fixed-step DRS.

With `--proximal-metric block`, the custom Nesterov/Schur worker constructs
dense Gauss-Newton `9x9` camera blocks. The regularization floor is controlled
by `--block-regularization`; recovery multiplies it by
`--recovery-penalty-ratio` up to `--maximum-block-regularization`.

The curvature coefficient and weak-direction floor can be separated with
`--block-curvature-multiplier`. Its default value `0` preserves the legacy
coefficient `min(1.005, 0.1*sqrt(be/be_initial))`; a positive value fixes the
coefficient of the camera Gauss-Newton block directly. Recovery defaults to
increasing `block_regularization`. The opt-in
`--block-recovery-mode curvature` instead keeps `block_regularization` fixed
and increases the curvature multiplier up to
`--maximum-block-curvature-multiplier`.

Curvature recovery can optionally use hysteresis: after
`--curvature-decay-after N` accepted iterations without a curvature change,
the multiplier is reduced by `--curvature-decay-ratio`, but never below its
initial value. The default wait `0` disables decay.

The custom DABA trust policy normally starts every local solve with radius 100.
`--persistent-trust-region` instead carries each cluster's accepted radius to
the next solve. A global rejection restores the last accepted radius and
multiplies it by `--trust-region-recovery-ratio`. This is experimental and not
a default: it improved scene 646 but substantially degraded scene 931.

`--metric-diagnostic-iterations N` enables diagnostic-only generalized power
iteration for the transformed reduced-camera smoothness constant, plus a
returned-state proximal defect evaluation. Each trajectory row stores all
per-cluster estimates, eigen-residuals, camera/landmark defect components, the
defect-to-fixed-point ratio, and the official Themelis metric/decrease checks.
These values do not alter acceptance. Sixty iterations gave approximately 2%
eigen-residual on the first validation scene; 30 is a cheaper exploratory
setting.

`--block-recovery-mode measured_curvature` uses the measured maximum to jump
toward `--target-transformed-lipschitz` (default 0.475), with geometric recovery
as fallback. This mode is experimental and performed worse than ordinary
curvature doubling on scenes 646 and 931. The diagnostics showed that once the
camera metric became admissible, the remaining proximal defect on scene 931
was dominated by unregularized landmark stationarity.

## Consensus metric

Block proximal mode supports four reductions of each active camera-copy block

`--consensus-metric arithmetic|scalar|diagonal|full`:

- `arithmetic`: identity weighting;
- `scalar`: geometric mean of the positive block diagonal, times identity;
- `diagonal`: the floored positive diagonal;
- `full`: the symmetrized dense `9x9` block with a floored diagonal.

For a selected copy metric `D[i,c]`, projection computes

\[
 v_c=\left(\sum_i D_{i,c}\right)^{-1}
   \sum_iD_{i,c}(2u_{i,c}-s_{i,c}).
\]

The selected metric is used consistently for projection, residuals, and the
DRE splitting term. Scalar proximal mode intentionally permits only arithmetic
consensus: its worker reply is `rho I`, not a Gauss-Newton block, so exposing
the weighted names there would not represent distinct algorithms.

`--metric-proposal-disagreement-scale alpha` optionally filters the local
proposal disagreement before reflection:

\[
 u^{(\alpha)}=[P_D+\alpha(I-P_D)]u.
\]

This preserves both `P_D u` and the current reflected projection
`P_D(2u-s)`, but changes the center update to

\[
 s_\alpha^+=s_1^++\lambda(1-\alpha)(u-P_Du).
\]

It is therefore disagreement-mode damping rather than uniform step damping.
The full derivation and normalized disagreement statistic are in
`block_metric_consensus_derivation.md`.

## Relative safeguard

The default safeguard matches the relative conjunction used by `client_acc.py`:

\[
\operatorname{reject}_k =
\neg\operatorname{finite}(E_k,f(v_k))
\ \lor\
\left[E_k>r_{E,k}E_{\mathrm{ref}}
\ \land\ f(v_k)>r_{f,k}f_{\mathrm{ref}}\right].
\]

The ratios are

\[
r_{E,k}=1+0.01\frac{(1-k/K)^4}{(1-\min(5,K-1)/K)^4},
\qquad
r_{f,k}=\max(1.001,\sqrt{r_{E,k}}).
\]

Thus the DRE and physical primal objective must both exceed their last accepted
references before a finite candidate is rejected. Debug output prints the two
ratios and two Boolean comparisons on every iteration. `SAFEGUARD_MODE=none`
disables finite relative rejection while retaining nonfinite-state protection;
it is used only for compatibility with old unsafeguarded reference runs.

The coordinator keeps the last accepted consensus and landmarks separately
from the lowest-primal-cost state used for final reporting. On rejection,
recovery atomically constructs the coherent restart
`s = u = v = v_accepted` and supplies the accepted landmarks in the
synchronized worker update. It does not restore the old non-consensus accepted
`(s, u)` pair: doing so changes the next proximal center and caused a severe
scene-931 regression. Scalar proximal mode then increases `rho`; block proximal
mode instead increases `block_regularization`, because scalar `rho` is inactive
in that worker path.

If another trial is rejected after the active recovery parameter has reached
its configured maximum, the deterministic restart can no longer change the DRS
map. The coordinator terminates with `terminationReason=recovery_exhausted` and
returns the accepted state instead of replaying an accept/reject cycle. Decreasing
the same parameter after an accepted step is deliberately not used: it can
create a parameter cycle around the same accepted-state restart and, in block mode,
changing scalar `rho` has no effect at all.

## Debug output

With `--debug-output`, each outer iteration reports:

- `f(v)`: accepted physical pixel SSE;
- `candidate`: candidate physical pixel SSE before restoration;
- `|u-v|^2`: DRS fixed-point residual in the selected metric;
- `|u-s|^2`: local proximal displacement in the selected metric;
- `|2u-s-v|^2`: reflection-to-projection residual in the selected metric;
- `|s_next-s|^2`: relaxed center step in the selected metric;
- `dre_split`: the quadratic splitting contribution
  \(\rho\langle u-v,u-v+2(u-s)\rangle/2\);
- `prox_obj`: worker-returned proximalized local objective
  \(F(u)+\rho\lVert u-s\rVert^2\);
- `F(u)`: data-only local objective recovered by subtracting the exact scalar
  proximal term;
- `DRE_model`: \(F(u)+\text{dre_split}\);
- `DRE`: \(\max(\text{DRE_model},f(v))\), matching the sandwich convention in
  `client_acc.py`;
- `DRE_ref` and `DRE_gain`: accepted reference envelope and candidate decrease;
- `rho`, `gamma`, `lambda`, rejection count, and transport bytes.

For the scalar prior, the worker source sets the proximal residual
matrix to \(\sqrt{\rho}I\). Since its reply is twice Ceres cost, subtraction of
\(\rho\lVert u-s\rVert^2\) recovers `F(u)` in the same SSE convention as
`f(v)`. The complete scalar-metric envelope identities are unit-tested and also
checked from persisted working-run trajectories. In block mode the custom
worker returns the data-only `F(u)` and the camera blocks separately.

## Run the failure cohort

```bash
cd /home/vogechri/bundlePalm/python
./serverTest/run_drs_failure_top3_live.sh
```

The runner uses isolated ports 6656/6657, fresh workers, live human-readable
output, per-case logs, exact-key resumability, and optional `OVERWRITE=1`.
Useful overrides are:

```bash
OVERWRITE=1 PROBLEM_FILTER="646" CLUSTERS_LIST="10" \
  ITERATIONS=30 RELAXATION=1.0 PENALTY_MULTIPLIER=1.0 \
  ./serverTest/run_drs_failure_top3_live.sh
```

The runner defaults to block/full. A projection ablation is:

```bash
for metric in arithmetic scalar diagonal full; do
  PROXIMAL_METRIC=block CONSENSUS_METRIC="$metric" \
    LOCAL_SOLVER=nesterov TRUST_REGION_POLICY=daba \
    ./serverTest/run_drs_failure_top3_live.sh
done
```

A curvature-only recovery run is:

```bash
BLOCK_CURVATURE_MULTIPLIER=0.1 \
BLOCK_RECOVERY_MODE=curvature \
MAXIMUM_BLOCK_CURVATURE_MULTIPLIER=128 \
BLOCK_REGULARIZATION=5e-5 \
TRUST_REGION_POLICY=daba \
  ./serverTest/run_drs_failure_top3_live.sh
```

The current two-scene continuation candidate additionally uses:

```bash
CURVATURE_DECAY_AFTER=10 CURVATURE_DECAY_RATIO=0.5 \
MINIMUM_PRIMAL_RATIO=1.0 \
  ./serverTest/run_drs_failure_top3_live.sh
```

For compatibility with `client_acc.py`, the runner also accepts
`BUNDLE_PALM_DRS_CONSENSUS_METRIC`. `CONSENSUS_METRIC` is the runner shorthand
and takes precedence if both variables are set. For example:

```bash
BUNDLE_PALM_DRS_CONSENSUS_METRIC=diagonal \
  ./serverTest/run_drs_failure_top3_live.sh
```

The exact scalar arithmetic compatibility path is:

```bash
PROXIMAL_METRIC=scalar CONSENSUS_METRIC=arithmetic \
  LOCAL_SOLVER=ceres_pcg TRUST_REGION_POLICY=ceres \
  ./serverTest/run_drs_failure_top3_live.sh
```

## Current scope

The separate coordinator currently provides:

- scalar and dense block local proximal metrics;
- arithmetic, scalar, diagonal, and full block consensus projection;
- metric-consistent residual and DRE evaluation;
- fixed `lambda`;
- no momentum or line search;
- relative, catastrophic, or disabled safeguards;
- coherent camera, center, and landmark restoration before regularization
  escalation.

Safeguarded acceleration remains deliberately absent. The mature
`client_acc.py` remains the reference for that feature.

## Planned safeguarded acceleration

The next outer-method addition is an opt-in accelerated line search, with plain
DRS retained as the baseline. Its required control flow is:

1. construct a Nesterov-like proposal for the DRS center;
2. evaluate accelerated line-search trials with the same local oracle,
  projection metric, and DRE merit as the nominal method;
3. fall back to the nominal plain-DRS center when the accelerated proposal does
  not satisfy the acceptance test; and
4. restart momentum after failed accelerated proposals.

This proposal/merit/fallback/restart mechanism is the intended Themelis-style
safeguarded acceleration contribution. Two behaviors in the legacy
`client_acc.py` are explicitly excluded from the clean port: disabling finite
rejection when `block_regularization` reaches its ceiling, and replacing the
accepted DRE reference with a rejected trial merely to avoid another rejection.
Those behaviors can hide deterioration and are not needed to define the
accelerated line search.

The eventual interface should distinguish at least `plain` and
`accelerated_linesearch` outer methods. Component ablations must separately
measure plain DRS, unguarded acceleration, safeguarded fallback, restart, and
adaptive metric recovery.
