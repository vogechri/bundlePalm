# Mature C1/Nesterov Transient Schur Majorizer

The crash-lost off-diagonal Frobenius-majorizer transport was restored and
tested with one global historical policy: startup observability threshold
`0.55`, Frobenius scale `0.5`, 32 buckets, active through I10, then raw block
DRS. The local proximal solve and consensus projection use the same block
metric. The uncoupled branch reproduces the prior Roman I3 trajectory and state
bitwise.

| Scene | Horizon | Selected | Candidate/control SSE | Control rejections | Candidate rejections | Time ratio |
|---|---:|---:|---:|---:|---:|---:|
| Roman | 30 | yes | 0.993173 | 1 | 1 | 1.0472 |
| BAL52 | 30 | no | 1.000000 | 0 | 0 | 1.0709 |
| Roman | 90 | yes | 1.007679 | 7 | 10 | 1.0338 |
| BAL52 | 90 | no | 1.000000 | 0 | 0 | 1.0345 |

BAL52 declines the selector and reproduces every behavioral field, endpoint
camera, and endpoint point exactly. Roman recovers the early I10 lag by I30 but
loses that gain by I90. Do not broaden or retune this component on the mature
backbone.

Projection-only full coupling was catastrophic, while this consistent diagonal
majorizer is too weak to repair late direction quality. The next diagnostic is
one behavior-neutral Jacobi-preconditioned Schur residual correction of the
nominal shared DRS tangent. This tests whether one distributed cross-camera
operator action can recover useful direction without replacing DRS by a global
Schur solve.