# C2 Diagonal Symmetric Ruiz Gate

## Scope

This is the first, low-risk subgate of idea 2. It computes each camera's full
initial `9x9` Gauss--Newton block, applies symmetric Ruiz iterations to absolute
row norms, and retains the existing diagonal camera-scaling transport. It does
not implement a full matrix-valued camera coordinate transform.

The new global mode is `CAMERA_SCALING=ruiz_initial`; the existing default
remains `jacobi_initial`.

## Roman/Trafalgar I5

Matched K24/I5 quality-stack DRS changes only camera scaling.

| Scene | Ruiz/Jacobi SSE | Ruiz/Jacobi time | Rejections |
|---|---:|---:|---:|
| Roman Forum | 0.999999998 | 1.216221 | 0 / 0 |
| Trafalgar | 1.000000000 | 0.948740 | 0 / 0 |
| **Geometric mean** | **0.999999999** | **1.074187** | - |

Every displayed trajectory row agrees to approximately `1e-9` relative. Ruiz
coordinate scales differ from Jacobi by at most `1.26%` on Roman and `1.46%` on
Trafalgar, but the scaling extrema and DRS behavior are effectively unchanged.

## BAL1778 I5

Matched raw K24/I5 Jacobi and Ruiz runs also agree:

- final SSE ratio: `0.999999999998x`;
- overall time ratio: `1.061371x`;
- optimization time ratio: `1.022937x`;
- identical median and maximum local linear iterations at every outer step;
- no safeguard rejections in either arm.

The Ruiz/Jacobi coordinate-scale difference is at most `0.813%` on BAL1778.
It does not improve the current-source conditioning failure or trajectory.

## Decision

Reject diagonal symmetric Ruiz as a quality or solver-work mechanism. Keep the
mode default-off as validated C2 diagnostic infrastructure. This does not close
full block-coordinate C2: a true `9x9` transform requires consistent matrix
state conversion, worker transform transport, and physical-metric congruence in
the current DRS coordinator. Evaluate that as a separate implementation gate;
do not tune Ruiz iterations, tolerance, clipping, or scale caps from these
scenes.
