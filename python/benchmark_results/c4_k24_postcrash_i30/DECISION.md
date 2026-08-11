# C4 post-crash K24 transfer gate

This repeats the frozen K2 comparison at `K24/I30/L1` without changing the
inner tolerance, trust policy, proximal metric, coordinate system, local depth,
or outer algorithm. C1 and C5 remain disabled.

| Scene | Solver | Final SSE | Optimization s | Rejections | Median inner iterations | Maximum residual |
|---|---:|---:|---:|---:|---:|---:|
| Roman Forum | Nesterov | 6,982,024.57 | 5.03 | 8 | 12.00 | 0.1171 |
| Roman Forum | Schur-PCG | 6,834,464.25 | 4.39 | 8 | 5.25 | 0.01000 |
| Trafalgar | Nesterov | 22,419,955.99 | 12.23 | 11 | 9.00 | 0.3702 |
| Trafalgar | Schur-PCG | 20,271,345.55 | 10.61 | 11 | 5.00 | 0.00998 |

Schur-PCG reaches `0.9789x` the Nesterov SSE on Roman and `0.9042x` on
Trafalgar. Its optimization time is `0.8732x` and `0.8672x`, respectively.
The accepted local solve count is identical, while aggregate inner work is
substantially lower. The inner-only policy therefore passes the K24 transfer
gate. The next check is unchanged six-scene K24 breadth, not tolerance tuning.