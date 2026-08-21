# K24 Safeguarded One-Step Schur Proposal

Roman Forum and BAL52 were run through I120. One proposal is permitted at I90.
It starts from nominal DRS, applies one shared-camera Schur matvec/Jacobi
correction, and evaluates eight fixed geometric scales from `1` through
`1/128`. The lowest precise worker-SSE candidate is selected only if it beats
ordinary consensus, followed by atomic DRS center/residual rebuild.

| Scene | Scale | Immediate I90 ratio | I100 ratio | I120 ratio | Delivered endpoint ratio | Control/candidate rejections |
|---|---:|---:|---:|---:|---:|---:|
| Roman | 0.5 | 0.888157 | 0.821864 | 0.800360 | 0.797016 | 11 / 8 |
| BAL52 | 0.0625 | 0.999999 | 0.999999 | 1.000000 | 1.000000 | 0 / 0 |

Both proposals pass actual SSE selection and the outer safeguard. Roman has one
rejected continuation step at I91, then improves further; total rejections are
lower than control. BAL52 is effectively neutral and has no rejection.

The same frozen I90 proposal should next run on the seven-scene direction
cohort through I120. No scale, timing, or damping changes are allowed. Report
selection, immediate and delivered ratios, completion, and losses before any
all-15/all-29 breadth.