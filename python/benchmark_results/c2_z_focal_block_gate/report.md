# C2 Translation-Z/Focal Restricted Block Gate

## Motivation

Worker-derived initial camera metrics show near-singular coupling between
physical translation-z (camera index 5) and focal length (index 6):

| Scene | Median correlation | P95 absolute | Maximum absolute |
|---|---:|---:|---:|
| Roman Forum | 0.9106 | 0.9971 | 0.99999 |
| Trafalgar | 0.8892 | 0.9950 | 1.00000 |
| BAL1778 | 0.9835 | 0.9993 | 1.00000 |

Diagonal scaling cannot remove this direction. Dense `9x9` whitening mixes all
camera coordinates and is unsafe. The default-off
`CAMERA_SCALING=worker_z_f_block_jacobi_initial` mode therefore uses diagonal
Jacobi on all coordinates except one `2x2` inverse-square-root block on
`(translation-z, focal)`. It uses the same aggregated post-proximal worker
metric and relative floor `1e-6`.

## Smoke

BAL49 K2/I1 completes and accepts its first candidate at `141,223.541` SSE. It
is `0.29%` worse than worker-derived diagonal scaling but `5.1%` better than the
dense worker-derived transform.

## Frozen I3 Gate

| Scene | Restricted / diagonal SSE | Rejections | Restricted / dense SSE |
|---|---:|---:|---:|
| Roman Forum | 0.997010 | 0/3 | 1.024324 |
| Trafalgar | 0.998699 | 0/3 | 0.715515 |
| BAL1778 | 92.442181 | 3/3 | 1.054910 |

The restricted block gives small consistent 1DSfM gains: `0.299%` on Roman and
`0.130%` on Trafalgar. BAL1778 rejects every candidate and preserves its initial
state; its transformed scale ratio is `1.46e21`. The pair block changes
BAL1778's local linear work but does not produce an admissible nonlinear step.

## Decision

The focal/depth coupling is real and the restricted transform is better behaved
than dense whitening on 1DSfM. It is not a globally safe cross-family C2 policy.
Correlation magnitude cannot serve as an activation selector because BAL1778
has the strongest measured coupling and the worst outcome. Retain the mode and
correlation telemetry default-off; do not tune floors, pair strength, or scene
thresholds from this gate. Idea 2 remains closed for the current production
formulation.
