# Separate DRS Coordinator Compatibility Gate

Date: 2026-07-27

## Reference

The previous embedded `plain_drs` path is reproduced on the known-working
Venice-52 problem with:

- K5 and two outer iterations;
- two local Ceres iterations;
- no camera scaling;
- fixed scalar penalty `16691.009615384617`;
- canonical relaxation `lambda = 1`;
- `SAFEGUARD_MODE=none`, matching the old unsafeguarded plain-DRS path.

## Result

| Iteration | Previous SSE | Separate DRS SSE | `|u-v|^2` | Proximal objective |
|---:|---:|---:|---:|---:|
| 0 | 30,733,521.94808663 | 30,733,521.94808665 | 21.4704778101 | 2,758,556.03125 |
| 1 | 12,848,961.85649521 | 12,848,961.85649522 | 6.38336342863 | 1,931,643.046875 |

The saved points are bit-identical. Maximum camera difference is
`1.07e-14`; final SSE differs by `3.73e-9`. This passes strict numerical
equivalence and confirms that the new coordinator preserves the old plain DRS
operator while exposing it in the notation used by `client_acc.py` and the
method writeup.

## DRE diagnostics

The scalar worker objective satisfies

`prox_obj = F(u) + rho * |u-s|^2`.

The separate coordinator reports the complete sandwich envelope

`DRE = max(F(u) + dre_split, f(v))`.

| Iteration | `F(u)` | `dre_split` | `DRE_model` | `f(v)` / DRE | DRE gain |
|---:|---:|---:|---:|---:|---:|
| 0 | 2,400,192 | -85,058 | 2,315,134 | 30,733,522 | -8,429,396 |
| 1 | 1,770,007 | -3,363 | 1,766,644 | 12,848,962 | 17,884,560 |

The `f(v)` sandwich is active on both iterations. All envelope identities were
recomputed from the persisted JSON trajectory and pass at floating-point
tolerance.

## Relative safeguard behavior

The restored default safeguard compares the trial against the last accepted
DRE and physical primal objective, rejecting only when both relative thresholds
fail. On this two-iteration compatibility setup its annealed ratios are:

| Iteration | DRE ratio | Primal ratio | Decision |
|---:|---:|---:|---|
| 0 | 1.16 | 1.07703 | reject: both fail |
| 1 | 1.01 | 1.00499 | reject: both fail after recovery rerun |

Thus the relative guard suppresses the initial nonmonotone transient that the
old plain-DRS run needs before improving at iteration 1. This is not a defect in
the reproduction: it distinguishes the unsafeguarded algorithm-compatibility
gate from the guarded debugging configuration. For a normal 30-iteration run,
the same schedule is much less unusual at the start (`r_DRE` is about 1.0207)
and reaches 1.01 at iteration 5.