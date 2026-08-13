# Collective Trust Trial/Commit Gate

## Mechanism

During a fixed startup window, each ordinary K24 local solve is compared with
one alternate solve that starts every worker at a shared trust radius. The
shared radius is the geometric mean of the nominal workers' completed radii,
multiplied by the existing trust recovery ratio `0.5`. Both candidates use the
same DRS centers and pass the same global DRE and physical-SSE safeguards.
The alternate is selected only for a material physical-SSE decrease.

The selected trial's per-worker trust radii are carried into the next outer
iteration. Coordinator-owned cameras and the existing nominal landmark snapshot
restore the selected geometry. This closes the hidden-state defect in an initial
probe where a losing alternate left its trust state resident.

The feature is default-off via `--collective-trust-trial-until 0` and requires
persistent trust with coordinator-owned cameras.

## Roman/Trafalgar I5 Gate

Matched K24/I5 runs use the quality stack's shared-only block metric, direct
left-SE3 assembly, proposal damping `0.5`, safeguarded Themelis acceleration,
and enhanced local solve through I5. The candidate evaluates collective trust
alternatives on all five iterations.

| Scene | Candidate/control SSE | Candidate/control time | Selected trials | Oracle calls candidate/control |
|---|---:|---:|---:|---:|
| Roman Forum | 0.999999999 | 1.046572 | 0/5 | 13/8 |
| Trafalgar | 1.000000000 | 1.103407 | 0/5 | 13/8 |
| **Geometric mean** | **0.999999999** | **1.074614** | **0/10** | - |

The candidate and control physical-SSE trajectories agree to approximately
`1e-9` relative. Shared alternatives are identical to nominal early and worse
later. No candidate is materially better, so the corrected commit policy keeps
nominal geometry and radii throughout.

## Decision

Reject collective shared-radius trial/commit as the next quality mechanism.
The experiment demonstrates that synchronizing the starting radius and choosing
by global SSE does not alter the early basin on the sentinels; local solves
already reach the same or better trial. Do not run I60 or broaden this policy.
Retain the default-off implementation only as explicit trial-state consistency
and diagnostic infrastructure.

Artifacts:

- `benchmark_results/collective_trust_rt_i5/`;
- `benchmark_results/collective_trust_rt_i5_control/`.
