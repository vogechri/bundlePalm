# Relative DRS Recovery Diagnosis on Venice-52

Date: 2026-07-27

## Question

After a relative DRE/primal rejection, why did increasing the scalar proximal
penalty make subsequent trial `f(v)` progressively worse?

## Root cause

The initial recovery restored the previous accepted DRS tuple but retained its
non-consensus splitting center `s`. Repeated penalty doubling therefore forced

`u = prox_F(s) -> s`

while the consensus operation still computed

`v = P_C(2u-s) -> P_C(s)`.

Because `s` was not a coherent consensus state, stronger proximal coupling
pinned the local solves to the wrong center. The observed signature was:

- `|u-s|^2` decreased toward zero;
- `|u-v|^2` increased toward 3.69;
- trial `f(v)` increased from 9.80M at iteration 6 to 35.21M at iteration 29;
- `rho` saturated at `1e12`;
- only 4 of 30 iterations were accepted.

This showed that the correction did not implement the recovery used by
`client_acc.py`.

## Correct recovery

On rejection, reset the complete best state coherently:

`s_i = u_i = v_best` for every local camera copy,

restore the physical landmarks associated with the same best state, and only
then increase `rho`. Best landmarks are sent in the synchronized cluster update
and acknowledged by the normal solve reply; the earlier asynchronous
`best_cost_proto` experiment was removed because it caused a transport timeout.

## Controlled comparison

All three runs use scene 52, K5, 30 outer iterations, one local Ceres step,
no camera scaling, `lambda=1`, and the same relative DRE/primal safeguard.

| Recovery state after rejection | Best SSE | Mean px | Accepted / rejected | Final rho | Final `|u-v|^2` |
|---|---:|---:|---:|---:|---:|
| Preserve non-consensus tuple | 6,346,535.53 | 2.9905 | 4 / 26 | `1.00e12` | 3.69145 |
| Reset cameras and centers | 1,798,094.10 | 1.5703 | 19 / 11 | `3.42e7` | `2.78e-5` |
| Reset cameras, centers, and landmarks | **1,733,413.98** | **1.5378** | **20 / 10** | **`1.71e7`** | `3.89e-5` |

With complete reset, the iteration-6 rejection is followed by trial SSE falling
from 5.62M to 3.04M at iteration 7, rather than growing monotonically. The run
continues improving through iteration 29.

The final saved state independently reproduces SSE `1,733,413.9761142628`
exactly. Its RMSE is 2.2345 px and maximum error is 50.21 px.

## Conclusion

Penalty escalation is useful only after a coherent basin reset. A recovery that
changes the DRS metric while retaining a non-consensus center can make the next
proximal solve more faithful to the wrong state. Future variable-metric or block
metric safeguards must treat the accepted state, splitting center, local copies,
consensus, landmarks, and DRE reference as one atomic object.
