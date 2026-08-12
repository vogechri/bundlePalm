# Three-Correction Global Schur Sentinel Gate

Date: 2026-08-12

## Configuration

All four cases use the matched K24/I30 base-backbone hybrid: local Nesterov,
persistent DRS trust, curvature 0.4 recovery/decay, camera metric 75,
shared-only proximal terms, proposal damping 0.5 on duplicated cameras, and
safeguarded Themelis outer acceleration with binary line search `{1,0}`.

The final distributed global Schur loop starts from the best DRS checkpoint and
uses one global policy on every scene:

- at most three accepted corrections;
- camera/landmark damping 3/3 initially;
- damping multiplied by 0.5 after acceptance and 2 after rejection;
- matrix-free Python CG with relative tolerance `1e-6`;
- accepted only for finite lower independently evaluated SSE and converged CG;
- stop after an accepted correction below `1e-4` relative improvement;
- three landmark-refinement steps after each camera correction.

## Result

| Case | I30 pre-Schur SSE | One-correction SSE | Up-to-three SSE | Three/pre | Three/one | Accepted | Termination | Schur seconds |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Roman | 4,373,429.666 | 4,151,469.258 | 3,652,406.208 | 0.835135 | 0.879786 | 3 | maximum corrections | 1.671 |
| Trafalgar | 16,371,979.074 | 15,844,380.951 | 15,013,453.151 | 0.917021 | 0.947557 | 3 | maximum corrections | 10.675 |
| BAL52 | 496,270.466 | 496,190.299 | 496,186.770 | 0.999831 | 0.999993 | 2 | minimum relative decrease | 0.446 |
| BAL3068 | 3,348,759.114 | 3,347,166.814 | 3,346,886.900 | 0.999441 | 0.999916 | 2 | minimum relative decrease | 87.901 |

The geometric endpoint ratio to the uncorrected I30 state is `0.875121x` on
the 1DSfM pair, `0.999636x` on the BAL pair, and `0.935309x` over all four.
Relative to one correction, up to three reaches `0.913043x` on 1DSfM and
`0.999955x` on BAL.

All DRS prefixes are identical to the one-correction artifacts. Every accepted
CG solve converged, and worker/evaluator relative SSE disagreement remains
below `1.1e-8`.

## Decision

Retain the bounded repeated correction as a strong 1DSfM polishing candidate,
not as a DRS-core gain. The unchanged global stopping rule correctly stops both
BAL cases after correction two, but their second corrections are not
cost-effective. A predeclared `1e-3` progress stop would preserve the observed
three-correction 1DSfM endpoints while stopping BAL after correction one; this
is a trace-derived candidate and must be frozen before broader confirmation.

Before an all-15/all-29 run, compare the existing `bsr_low_memory` operator to
the Python operator on these four cases. BAL3068 correction time rises from
42.18 seconds for one correction to 87.90 seconds for two, and coordinator peak
RSS reaches 7,301,240 KiB. After operator equivalence is established, freeze
one operator and one stopping policy, then evaluate them unchanged on all 15
1DSfM and all 29 BAL scenes without retuning.
