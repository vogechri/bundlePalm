# Guard and Ceres Comparison

Date: 2026-07-26

## Protocol

- datasets: 1723, 52, 245, 394, 871
- 20 clusters, 30 outer iterations
- exactly one requested local nonlinear update
- one CPU thread per cluster
- initial Jacobi camera substitution, `alpha = 1`, adaptive scalar penalty
- standard BAL/Snavely pixel SSE
- catastrophic guard: reject nonfinite or greater than `1e6` one-step global
  pixel-SSE growth, restore coordinator and worker state, double ADMM penalty

Four methods were rerun: unguarded and guarded Nesterov + DABA trust region,
and unguarded and guarded Ceres iterative Schur + Ceres LM.

## Results

| Method | Geomean best SSE | Geomean endpoint SSE | Recoveries | Total time s |
|---|---:|---:|---:|---:|
| Nesterov + DABA TR | 3,110,876 | 915,648,534 | 0 | 99.26 |
| Nesterov + DABA TR + guard | 3,110,876 | 3,518,342 | 1 | 99.54 |
| Ceres + Ceres LM | 2,241,156 | 2,251,203 | 0 | 143.63 |
| Ceres + Ceres LM + guard | 2,152,077 | 2,166,427 | 4 | 141.39 |

For Nesterov, the guard triggered once on 1723 and zero times elsewhere. Best
SSE and best iteration were identical to the unguarded run on all five scenes.
The 1723 endpoint changed from `3.682e19` to `3.084e7`.

For Ceres, the guard triggered four times on 1723 and zero times elsewhere.
The four unaffected scenes reproduced the unguarded trajectory and best SSE.
On 1723, best SSE improved from `2,805,429` to `2,290,486`; endpoint SSE
improved from `2,809,739` to `2,319,061`. Runtime did not increase.

Every saved physical state independently reproduced its recorded pixel SSE
exactly.

## What Ceres + Ceres LM Means

The worker uses Ceres Solver 2.2.0. Each cluster problem contains standard
pixel reprojection residuals plus the ADMM proximal residual

$$
\sqrt{\rho}\,(x-s),
$$

with the current split intrinsic penalty when enabled. Variables use the same
initial Jacobi substitution as the custom path.

The configured Ceres path is:

- `ITERATIVE_SCHUR` linear solver;
- `SCHUR_JACOBI` preconditioner;
- Ceres monotonic Levenberg-Marquardt trust-region strategy;
- one requested minimizer update per cluster call;
- trust-region radius persisted between outer calls;
- Ceres default LM diagonal clamp `[1e-6, 1e32]`;
- minimum relative decrease `1e-3`;
- iterative linear solver limit 500 and forcing tolerance `eta = 0.1`;
- Ceres Jacobian column scaling enabled.

Ceres summaries contain the initial-state entry plus at most one trial entry,
so `successful=2` means one accepted update; `successful=1, unsuccessful=1`
means the trial was rejected and the input state was returned unchanged.

On unguarded 1723, 28 of 600 cluster calls rejected their one LM trial. Every
cluster call returned a nonincreasing augmented local objective. The other four
scenes had no rejected Ceres trials.

## Why Ceres Looked Stable

The premise that catastrophic excursions never occur for Ceres was false.
Unguarded Ceres reached global pixel SSE `1.290e25` at outer iteration 0 and
`1.890e15` at iteration 24 on 1723. It later recovered to the low millions.

Ceres is more recoverable because LM clamps weak normal-equation diagonals,
rejects locally invalid/nondecreasing trials, and carries the adjusted trust
radius into the next outer call. The custom Nesterov path directly inverts
weak camera blocks and, after its outer excursion, became trapped near
`3.68e19`.

However, local LM acceptance cannot guarantee a safe global consensus state.
Each cluster decreases its own augmented objective at its local cameras and
landmarks. The coordinator then averages duplicated cameras and evaluates that
consensus with the assembled landmark state. This combination can have much
higher global pixel SSE even when every local solve is valid. The outer guard
therefore addresses a failure mode that is distinct from Ceres's local LM
safeguards and benefits both solver families.

## Reproducibility Note

The first guarded Nesterov scene-245 run failed while assigning a protobuf
repeated-field view into NumPy. Converting camera and landmark replies with an
owning `np.array` copy removed the lifetime hazard. The isolated rerun completed
with best/final SSE `2,906,068.937797963`, zero recoveries, and 17.09 seconds.