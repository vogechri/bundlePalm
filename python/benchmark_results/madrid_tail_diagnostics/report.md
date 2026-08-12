# Madrid I60+16 Tail Diagnosis

Date: 2026-08-12

## References

The similarity-aligned I60+16 Madrid endpoint is compared with base-I200,
left-SE3 Ceres I90, and the verified BAE-style K1/I90 endpoint.

| Reference | Candidate/reference SSE | Low-8 center energy | Low-32 center energy |
|---|---:|---:|---:|
| Base-I200 | 1.251524 | 0.000453 | 0.244221 |
| Ceres | 1.314266 | 0.000877 | 0.075441 |
| BAE-style K1 | 1.150461 | 0.000954 | 0.055310 |

The first eight graph modes explain below `0.1%` of the camera-center gap for
all references. Madrid is therefore not a retained low-mode error of the kind
seen on Tower.

## Residual Concentration

The candidate is heavy-tailed: its top 1%/5%/10% observations carry
`45.65%/75.82%/87.21%` of SSE. This is not distinctive. The top 1% observation
shares are `52.01%` for base, `49.31%` for Ceres, and `46.26%` for BAE. The
candidate top-5% camera share is `34.08%`, versus `33.94%/39.42%/40.30%` for
the three references.

The excess is nevertheless localized consistently. After normalizing by each
camera's observation count, cameras 177, 171, and 173 are the three largest
candidate excess contributors against base, Ceres, and BAE. Their rotation
differences vary strongly by reference, so they do not define one global
rotation correction.

## Camera Subspaces

Raw camera-coordinate comparisons are ill-conditioned. Similarity-aligned
left-SE3 tangents show large coupled translation differences, while weak radial
coordinates show enormous relative changes because their reference scales are
small. Swapping cameras or points between independently optimized states gives
catastrophic pixel objectives. These facts prevent a defensible attribution to
one scalar camera subspace: the camera and point states compensate each other.

## Decision

Madrid is a coupled non-low-mode camera/point basin problem, not primarily:

- insufficient Schur budget (corrections 17--20 improve only `0.93%`),
- a low graph-mode error,
- globally excessive residual concentration, or
- one clean rotation/translation/intrinsics defect.

Future Madrid work should inspect the recurring cameras 177/171/173 and their
owned tracks jointly, preserving camera-point coupling. Do not tune a global
pose prior or camera-subspace scale from Madrid.
